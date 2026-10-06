#!/usr/bin/env python
"""Per-figure number tables, so every panel in the deliverable is backed by a readable TSV."""
import argparse
import glob
import os

import pandas as pd

import tracks_io as T


def main(a):
    os.makedirs(a.out, exist_ok=True)
    singles = pd.read_csv(a.singles, sep="\t")
    pairs = pd.read_csv(a.pairs, sep="\t")
    md = pd.read_csv(a.ct_metadata, sep="\t")
    order = dict(zip(md.cell_type, md.display_order))

    rows_s, rows_p = [], []
    for f in sorted(glob.glob(os.path.join(a.tracks, "tracks__*.npz"))):
        cond, meta = T.load(f)
        ct = meta["ct"]
        for bgset, cs in cond.items():
            bg = cs["bg"]
            for _, s in singles.iterrows():
                c = cs[f"single__{s.short_id}"]
                rows_s.append(dict(short_id=s.short_id, tf=s.curator_tf,
                                   category=s.curator_category, width=s.width,
                                   cell_type=ct, display_order=order[ct], background_set=bgset,
                                   log_counts_background=round(float(bg["c"].mean()), 4),
                                   log_counts_with_motif=round(float(c["c"].mean()), 4),
                                   delta_log_counts=round(T.dlog(c, bg), 4),
                                   n_backgrounds=int(len(bg["c"]))))
            for _, r in pairs[pairs.ct == ct].iterrows():
                key = f"pair__{r.idA}__{r.idB}"
                m = cs[key]["meta"]
                dA_m, dB_m = T.dlog(cs[f"{key}__A"], bg), T.dlog(cs[f"{key}__B"], bg)
                dJ = T.dlog(cs[f"{key}__AB"], bg)
                rows_p.append(dict(
                    idA=r.idA, idB=r.idB, tf_A=r.tf_A, tf_B=r.tf_B,
                    cell_type=ct, display_order=order[ct], background_set=bgset,
                    orientation=m["orient"], gap_bp=m["gap"], center_to_center_bp=m["center_center"],
                    width_A=m["wa"], width_B=m["wb"],
                    dA_centered=round(T.dlog(cs[f"single__{r.idA}"], bg), 4),
                    dB_centered=round(T.dlog(cs[f"single__{r.idB}"], bg), 4),
                    dA_in_pair=round(dA_m, 4), dB_in_pair=round(dB_m, 4),
                    dS_additive=round(dA_m + dB_m, 4), dJ_observed=round(dJ, 4),
                    synergy_delta=round(dJ - dA_m - dB_m, 4),
                    published_dS=r.dS, published_dJ=r.dJ_opt, published_delta=r.delta,
                    published_maxZ=r.maxZ, published_q_delta=r.q_delta, published_call=r.call,
                    n_backgrounds=int(len(bg["c"]))))

    s = pd.DataFrame(rows_s).sort_values(["short_id", "background_set", "display_order"])
    p = pd.DataFrame(rows_p).sort_values(["idA", "idB", "background_set", "display_order"])
    s.to_csv(os.path.join(a.out, "single_tf_marginalization.tsv"), sep="\t", index=False)
    p.to_csv(os.path.join(a.out, "pair_synergy_tracks.tsv"), sep="\t", index=False)
    print(f"wrote single_tf_marginalization.tsv ({len(s)} rows)")
    print(f"wrote pair_synergy_tracks.tsv ({len(p)} rows)\n")

    print("=== headline single-TF panels (own backgrounds, strongest cell type) ===")
    own = s[s.background_set == "own"]
    print(own.loc[own.groupby("short_id").delta_log_counts.idxmax()]
          [["short_id", "tf", "category", "cell_type", "log_counts_background",
            "log_counts_with_motif", "delta_log_counts"]].to_string(index=False))

    print("\n=== headline pair panels (own backgrounds, max published delta) ===")
    ownp = p[p.background_set == "own"]
    idx = ownp.groupby(["idA", "idB"]).published_delta.idxmax()
    print(ownp.loc[idx][["idA", "idB", "tf_A", "tf_B", "cell_type", "orientation", "gap_bp",
                         "dA_in_pair", "dB_in_pair", "dS_additive", "dJ_observed",
                         "synergy_delta", "published_delta", "published_call"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--singles", required=True)
    ap.add_argument("--ct_metadata", required=True)
    ap.add_argument("-o", "--out", required=True)
    main(ap.parse_args())
