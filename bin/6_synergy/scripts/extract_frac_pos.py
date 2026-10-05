#!/usr/bin/env python
"""frac_pos per (pair, CT): fraction of backgrounds where the joint effect at the optimal
arrangement exceeds the additive expectation, i.e. mean(best_dj_seq > indep_seq) from the
per-pair npz written by marginalize_pairs.py (<idA>__<idB>__<ct>.npz).

Reported alongside the calls by downstream.py; not used as a gate.
Output: TSV with idA, idB, ct, frac_pos, nbg.
"""
import os, glob, argparse
import numpy as np, pandas as pd


def main(a):
    files = sorted(glob.glob(os.path.join(a.results, "chunk_*", "*__*__*.npz")))
    rows = []
    for f in files:
        idA, idB, ct = os.path.basename(f)[:-4].split("__")
        z = np.load(f)
        j, s = z["best_dj_seq"], z["indep_seq"]
        rows.append((idA, idB, ct, round(float(np.mean(j > s)), 4), int(len(j))))
    pd.DataFrame(rows, columns=["idA", "idB", "ct", "frac_pos", "nbg"]).to_csv(a.out, sep="\t", index=False)
    print(f"[frac_pos] {len(rows)} pair x CT -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="sweep dir with chunk_*/ subdirs")
    p.add_argument("-o", "--out", required=True)
    main(p.parse_args())
