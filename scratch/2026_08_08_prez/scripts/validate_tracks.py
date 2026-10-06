#!/usr/bin/env python
"""Verification for the marginalization track run.

1. Scalar reproduction -- recomputed dS and dJ_opt from the OWN-background set must match the
   published calls_long values. This is the decisive check that the new script reproduces the
   published pipeline rather than something adjacent to it.
2. Area identities -- the whole point of the log-space averaging + renormalization.
3. Profile-head sanity -- the motif insert must produce a visible central bump, not flat noise.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import tracks_io as T


def main(a):
    pairs = pd.read_csv(a.pairs, sep="\t")
    files = sorted(glob.glob(os.path.join(a.tracks, "tracks__*.npz")))
    print(f"{len(files)} track files in {a.tracks}\n")

    scal, ident, bump = [], [], []
    for f in files:
        cond, meta = T.load(f)
        ct = meta["ct"]
        for bgset, cs in cond.items():
            bg = cs["bg"]
            for _, r in pairs[pairs["ct"] == ct].iterrows():
                idA, idB = r["idA"], r["idB"]
                key = f"pair__{idA}__{idB}"
                dA_c = T.dlog(cs[f"single__{idA}"], bg)   # centered, forward: published convention
                dB_c = T.dlog(cs[f"single__{idB}"], bg)
                dJ = T.dlog(cs[f"{key}__AB"], bg)
                dA_m = T.dlog(cs[f"{key}__A"], bg)        # in-pair position AND orientation
                dB_m = T.dlog(cs[f"{key}__B"], bg)
                scal.append(dict(ct=ct, bgset=bgset, pair=f"{idA}__{idB}",
                                 dS_pub=r["dS"], dS_new=dA_c + dB_c,
                                 dJ_pub=r["dJ_opt"], dJ_new=dJ,
                                 delta_pub=r["delta"], delta_pub_recomp=dJ - (dA_c + dB_c),
                                 delta_matched=dJ - (dA_m + dB_m),
                                 dA_cen=dA_c, dA_matched=dA_m, dB_cen=dB_c, dB_matched=dB_m))

                Tb, Ta = T.track(bg), T.additive_track(cs[f"{key}__A"], cs[f"{key}__B"], bg)
                To = T.track(cs[f"{key}__AB"])
                ident.append(dict(ct=ct, bgset=bgset, pair=f"{idA}__{idB}",
                                  e_add=abs(np.log(Ta.sum() / Tb.sum()) - (dA_m + dB_m)),
                                  e_obs=abs(np.log(To.sum() / Ta.sum()) - (dJ - dA_m - dB_m))))

            for k in [c for c in cs if c.startswith("single__")]:
                lr = T.log_ratio_track(cs[k], bg)
                bump.append(dict(ct=ct, bgset=bgset, motif=k.split("__")[1],
                                 d=T.dlog(cs[k], bg),
                                 peak_central=float(T.window(lr, 25).max()),
                                 flank=float(np.abs(np.r_[lr[:200], lr[-200:]]).max())))

    s = pd.DataFrame(scal)
    own = s[s.bgset == "own"]
    print("=== 1. scalar reproduction vs published calls_long (own backgrounds) ===")
    for col, pub in [("dS_new", "dS_pub"), ("dJ_new", "dJ_pub"), ("delta_pub_recomp", "delta_pub")]:
        e = (own[col] - own[pub]).abs()
        print(f"  {col:>18s} vs {pub:<12s} max|err|={e.max():.2e}  mean={e.mean():.2e}  "
              f"{'OK' if e.max() < 1e-3 else 'MISMATCH'}")

    print("\n=== 2. area identities (all background sets) ===")
    i = pd.DataFrame(ident)
    print(f"  area(T_add)/area(T_bg) == exp(dA+dB)   max|err|={i.e_add.max():.2e}  "
          f"{'OK' if i.e_add.max() < 1e-6 else 'FAIL'}")
    print(f"  area(T_obs)/area(T_add) == exp(delta)  max|err|={i.e_obs.max():.2e}  "
          f"{'OK' if i.e_obs.max() < 1e-6 else 'FAIL'}")

    print("\n=== 3. profile head: central bump vs flank noise (log enrichment) ===")
    b = pd.DataFrame(bump)
    g = b[b.bgset == "own"].groupby("motif").agg(
        d=("d", "max"), peak=("peak_central", "max"), flank=("flank", "median")).sort_values("peak", ascending=False)
    g["ratio"] = g.peak / g.flank
    print(g.to_string())
    print(f"\n  verdict: {'profile head is informative' if g.ratio.max() > 2 else 'FLAT -- switch --background_mode'}")

    print("\n=== 4. centered-forward vs position/orientation-matched singles ===")
    m = own.assign(dA_gap=(own.dA_cen - own.dA_matched).abs(),
                   dB_gap=(own.dB_cen - own.dB_matched).abs())
    print(m.groupby("pair")[["dA_gap", "dB_gap", "delta_pub_recomp", "delta_matched"]]
          .agg(["max", "mean"]).round(3).to_string())

    if a.out:
        s.to_csv(a.out, sep="\t", index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default=None)
    main(p.parse_args())
