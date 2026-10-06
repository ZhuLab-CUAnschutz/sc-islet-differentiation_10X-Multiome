#!/usr/bin/env python
"""Soft/hard synergy calls on the full pair x CT matrix, via the matrix's OWN in-matrix empirical null
+ Benjamini-Hochberg. Retires the small, contaminated 42-pair lineage-mismatched null.

Per cell type, the null for Delta (and for the arrangement sharpness maxZ) is the CENTRAL BULK of that
CT's own distribution across all pairs (most pairs are non-synergistic background). We fit center/scale
robustly (median / MAD) so the synergistic tail does not inflate its own bar, then:
  soft_call = Delta clears the per-CT empirical bar  AND  BH q < QMAX
  hard_call = soft_call AND arrangement is sharply peaked (maxZ clears its per-CT bar)
BH is applied to the per-(pair,CT) Wilcoxon p across ALL tests.

Usage: python call_syntax_full.py <long_matrix.tsv> [out_prefix]
  long_matrix needs columns: idA idB ct delta maxZ wilcoxon_p   (MT matrix works as a prototype)
"""
import sys, os
import numpy as np, pandas as pd

Z_SOFT = 3.0     # Delta must exceed median + Z_SOFT*robust_sd of the per-CT null bulk
Z_HARD = 3.0     # maxZ (arrangement sharpness) must exceed its per-CT null bulk by this much
QMAX   = 0.05    # BH q threshold

def bh(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out

def robust_z(x):
    x = x.astype(float); med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad

def main(path, outpre):
    df = pd.read_csv(path, sep="\t")
    need = {"idA", "idB", "ct", "delta", "maxZ", "wilcoxon_p"}
    assert need <= set(df.columns), f"missing {need - set(df.columns)}"
    df["pair"] = df.idA + "__" + df.idB

    # BH across ALL pair x CT tests
    df["q"] = bh(df.wilcoxon_p.values)
    # per-CT in-matrix empirical null (robust central-bulk z)
    df["delta_z"] = df.groupby("ct")["delta"].transform(robust_z)
    df["maxZ_z"]  = df.groupby("ct")["maxZ"].transform(robust_z)

    df["soft_call"] = (df.delta_z > Z_SOFT) & (df.q < QMAX)
    df["hard_call"] = df.soft_call & (df.maxZ_z > Z_HARD)

    df.to_csv(f"{outpre}_calls_long.tsv", sep="\t", index=False)
    # pair x CT boolean call matrix (soft)
    call_mat = df.pivot_table(index="pair", columns="ct", values="soft_call", aggfunc="first").fillna(False)
    call_mat.to_csv(f"{outpre}_soft_call_matrix.tsv", sep="\t")

    breadth = df.groupby("pair").soft_call.sum()
    print(f"pairs={df.pair.nunique()}  CTs={df.ct.nunique()}  tests={len(df)}")
    print(f"soft (pair,CT) calls: {int(df.soft_call.sum())}   hard: {int(df.hard_call.sum())}")
    print(f"pairs soft in >=1 CT: {int((breadth>0).sum())}   hard in >=1 CT: "
          f"{int(df[df.hard_call].pair.nunique())}")
    print("\nper-CT soft counts:")
    print(df[df.soft_call].groupby("ct").size().sort_values(ascending=False).to_string())
    print(f"\nwrote {outpre}_calls_long.tsv + {outpre}_soft_call_matrix.tsv")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mt/mt_calls_long.tsv"
    outpre = sys.argv[2] if len(sys.argv) > 2 else "PROTO_mt"
    main(path, outpre)
