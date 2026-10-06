#!/usr/bin/env python
"""Build the input TSVs for a higher-order (3-way) synergy run from a finished pairwise run.

Given a small set of focus motifs (the TFs whose combinatorics you want to dissect) and the
pairwise calls for them, this writes the four manifests consumed by marginalize_tracks.py and
marginalize_triples.py:

  single_motifs.tsv     one row per focus motif (for the centered single-motif tracks)
  pair_arrangements.tsv the pairwise optimum per pair, locked as the anchor geometry
  triples.tsv           every C(n,3) triple x each of its 3 anchor-pair choices
  null_triples.tsv      the same anchors with a lineage-mismatched third motif

The null third motifs are drawn from the full catalog, excluding anything annotated to one of the
focus TFs (that exclusion is derived from the focus set, not hard-coded), and are spread evenly
across the width distribution so the null is not confounded by motif length.

Originally the DE-specific build_inputs.py of the 2026_09_20_de_higher_order sandbox (EOMES,
MIXL1, SOX17, FOXH1); the focus set is now an argument.
"""
import os, argparse, itertools
import numpy as np, pandas as pd


def _pair_key(a, b):
    return frozenset((a, b))


def main(a):
    os.makedirs(a.out, exist_ok=True)

    focus = pd.read_csv(a.motifs, sep="\t")
    for col in ("short_id", "curator_tf"):
        if col not in focus.columns:
            raise SystemExit(f"--motifs needs a {col} column; got {list(focus.columns)}")
    ids = focus.short_id.tolist()
    tf_of = dict(zip(focus.short_id, focus.curator_tf))
    print(f"[focus] {len(ids)} motifs: {', '.join(f'{s} ({tf_of[s]})' for s in ids)}")

    with np.load(a.cwms) as npz:
        widths = {k: npz[k].shape[0] for k in npz.files}
    missing = [s for s in ids if s not in widths]
    if missing:
        raise SystemExit(f"--cwms is missing focus motifs: {missing}")

    calls = pd.read_csv(a.calls, sep="\t")
    if a.set and "set" in calls.columns:
        calls = calls[calls["set"] == a.set]
    calls = calls[calls.idA.isin(ids) & calls.idB.isin(ids)].copy()
    n_het = int((calls.idA != calls.idB).sum())
    print(f"[pairs] {len(calls)} rows ({n_het} heterotypic, {len(calls) - n_het} homodimer)")
    pair_lookup = {_pair_key(r.idA, r.idB): r for r in calls.itertuples()}

    # ---- single_motifs.tsv ----
    cat_of = dict(zip(focus.short_id, focus.get("curator_category", pd.Series(dtype=str)))) \
        if "curator_category" in focus.columns else {}
    singles = pd.DataFrame([{"short_id": s, "curator_tf": tf_of[s],
                             "curator_category": cat_of.get(s, ""), "width": widths[s]}
                            for s in ids])
    singles.to_csv(os.path.join(a.out, "single_motifs.tsv"), sep="\t", index=False)

    # ---- pair_arrangements.tsv ----
    pair_cols = ["idA", "idB", "tfA", "tfB", "kind", "ct", "opt_orient", "opt_gap",
                 "opt_center_center", "wa", "wb", "dS", "dJ_opt", "delta", "maxZ", "call"]
    pairs = calls[[c for c in pair_cols if c in calls.columns]].reset_index(drop=True)
    pairs = pairs.rename(columns={"tfA": "tf_A", "tfB": "tf_B"})   # plot_tracks.py convention
    pairs.to_csv(os.path.join(a.out, "pair_arrangements.tsv"), sep="\t", index=False)

    # ---- triples.tsv: each triple, each of its 3 anchor choices ----
    rows = []
    for combo in itertools.combinations(ids, 3):
        for i in range(3):
            anchor = [combo[j] for j in range(3) if j != i]
            slid = combo[i]
            key = _pair_key(*anchor)
            if key not in pair_lookup:
                raise SystemExit(f"no pairwise call for anchor {anchor}; cannot lock its geometry")
            r = pair_lookup[key]
            rows.append({"idA": r.idA, "idB": r.idB, "idC": slid,
                         "tfA": tf_of[r.idA], "tfB": tf_of[r.idB], "tfC": tf_of[slid],
                         "anchor_orient": r.opt_orient, "anchor_gap": r.opt_gap,
                         "anchor_cc": r.opt_center_center,
                         "wa": widths[r.idA], "wb": widths[r.idB], "wc": widths[slid]})
    triples = pd.DataFrame(rows)
    triples.to_csv(os.path.join(a.out, "triples.tsv"), sep="\t", index=False)

    # ---- null_triples.tsv: same anchors, lineage-mismatched third motif ----
    meta = pd.read_csv(a.catalog, sep="\t")
    focus_tfs = set(tf_of.values())
    exclude = set(meta.loc[meta.curator_tf.isin(focus_tfs), "short_id"])   # derived, not hard-coded
    exclude |= set(ids)
    exclude |= {f"J_{tf}" for tf in focus_tfs}                             # JASPAR keys for the same TFs
    with np.load(a.null_cwms) as npz:
        pool = {k: npz[k].shape[0] for k in npz.files if k not in exclude}
    print(f"[null] {len(pool)} eligible motifs after excluding {len(exclude)} focus-TF entries")

    # spread the picks evenly over the width distribution
    ordered = sorted(pool.items(), key=lambda x: (x[1], x[0]))
    idx = [int(round(i * (len(ordered) - 1) / (a.n_nulls - 1))) for i in range(a.n_nulls)]
    nulls = [ordered[i] for i in idx]
    tf_lookup = dict(zip(meta.short_id, meta.curator_tf))
    for sid, w in nulls:
        print(f"  {sid} (w={w}, tf={tf_lookup.get(sid, sid)})")

    het = calls[calls.idA != calls.idB]
    rows = []
    for r in het.itertuples():
        for sid, w in nulls:
            tf = tf_lookup.get(sid, sid)
            rows.append({"idA": r.idA, "idB": r.idB, "idC": sid,
                         "tfA": r.tfA, "tfB": r.tfB, "tfC": tf,
                         "anchor_orient": r.opt_orient, "anchor_gap": r.opt_gap,
                         "anchor_cc": r.opt_center_center,
                         "wa": int(r.wa), "wb": int(r.wb), "wc": w,
                         "note": f"null: {tf} ({sid}) is lineage-mismatched"})
    null_triples = pd.DataFrame(rows)
    null_triples.to_csv(os.path.join(a.out, "null_triples.tsv"), sep="\t", index=False)

    print(f"\n[done] single_motifs {len(singles)} | pair_arrangements {len(pairs)} | "
          f"triples {len(triples)} | null_triples {len(null_triples)}  -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--motifs", required=True, help="TSV of focus motifs: short_id, curator_tf[, curator_category]")
    p.add_argument("--calls", required=True, help="pairwise calls_long.tsv covering the focus motifs")
    p.add_argument("--cwms", required=True, help="npz of CWMs for the focus motifs (widths)")
    p.add_argument("--catalog", required=True, help="catalog metadata.tsv (short_id, curator_tf)")
    p.add_argument("--null_cwms", required=True, help="npz of CWMs to draw null third motifs from")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--set", default=None, help="optional value of a 'set' column to filter calls on")
    p.add_argument("--n_nulls", type=int, default=7)
    main(p.parse_args())
