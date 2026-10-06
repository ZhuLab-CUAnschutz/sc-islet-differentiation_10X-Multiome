#!/usr/bin/env python
"""Stage 0: CPU-only checks on the login node, before any GPU allocation is spent.

Verifies every path, motif, and table row that marginalize_tracks.py will touch.
Exits non-zero on the first failure.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marginalize_tracks import cwm_consensus_ohe, model_paths, pair_starts  # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


def main(a):
    pairs = pd.read_csv(a.pairs, sep="\t")
    singles = pd.read_csv(a.singles, sep="\t")
    cts = sorted(pairs["ct"].unique())

    print(f"=== cell types ({len(cts)}) ===")
    check("22 cell types in pair table", len(cts) == 22, f"got {len(cts)}")

    print("\n=== model + background paths ===")
    for ct in cts:
        mp, nonpk, pk = model_paths(a.models_root, ct)
        check(f"{ct} model", os.path.exists(mp))
        check(f"{ct} nonpeaks", os.path.exists(nonpk))
        check(f"{ct} peaks (dinuc fallback)", os.path.exists(pk))

    print("\n=== genome ===")
    check("hg38.fa", os.path.exists(a.genome))
    check("hg38.fa.fai", os.path.exists(a.genome + ".fai"))

    print("\n=== CWMs and consensus widths ===")
    check("cwms.npz", os.path.exists(a.cwms))
    if os.path.exists(a.cwms):
        cwms = np.load(a.cwms)
        want = list(dict.fromkeys(
            singles["short_id"].tolist() + pairs["idA"].tolist() + pairs["idB"].tolist()))
        for st in want:
            present = st in cwms.files
            check(f"{st} in cwms.npz", present)
            if present:
                w = cwm_consensus_ohe(cwms[st]).shape[-1]
                # widths for pair members are recorded in the published table
                pub = None
                ra = pairs[pairs["idA"] == st]
                rb = pairs[pairs["idB"] == st]
                if len(ra):
                    pub = int(ra["wa"].iloc[0])
                elif len(rb):
                    pub = int(rb["wb"].iloc[0])
                if pub is None:
                    cat = singles[singles["short_id"] == st]
                    pub = int(cat["width"].iloc[0]) if len(cat) else None
                    check(f"{st} consensus width vs catalog", pub is None or w == pub,
                          f"consensus={w} catalog={pub}")
                else:
                    check(f"{st} consensus width vs published wa/wb", w == pub,
                          f"consensus={w} published={pub}")

    print("\n=== pair geometry reproduces published center-center ===")
    bad = 0
    for _, r in pairs.iterrows():
        _, _, cc = pair_starts(int(r["wa"]), int(r["wb"]), int(r["opt_gap"]))
        if abs(cc - float(r["opt_center_center"])) > 0.51:
            bad += 1
            print(f"    mismatch {r['idA']}x{r['idB']}/{r['ct']}: {cc} vs {r['opt_center_center']}")
    check("all 66 rows reproduce opt_center_center", bad == 0, f"{len(pairs)} rows, {bad} bad")
    check("66 pair x cell-type rows", len(pairs) == 66, f"got {len(pairs)}")
    check("all rows nbg==32", (pairs["nbg"] == 32).all(),
          f"nbg values {sorted(pairs['nbg'].unique())}")
    check("no opt_gap at the maxdist=50 boundary", (pairs["opt_gap"] < 50).all(),
          f"max opt_gap={pairs['opt_gap'].max()}")

    print("\n=== env ===")
    try:
        import torch
        import tangermeme
        import bpnetlite
        print(f"    torch {torch.__version__}  tangermeme {tangermeme.__version__}  "
              f"bpnetlite {bpnetlite.__version__}  pandas {pd.__version__}")
        check("torch/tangermeme/bpnetlite import", True)
    except Exception as e:  # noqa: BLE001
        check("torch/tangermeme/bpnetlite import", False, str(e))

    print()
    if FAIL:
        print(f"PREFLIGHT FAILED: {len(FAIL)} check(s)")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models_root", required=True)
    p.add_argument("--cwms", required=True)
    p.add_argument("-g", "--genome", required=True)
    p.add_argument("--singles", required=True)
    p.add_argument("--pairs", required=True)
    main(p.parse_args())
