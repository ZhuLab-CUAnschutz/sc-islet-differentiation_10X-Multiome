#!/usr/bin/env python
"""Aggregate + call the 3-way synergy run (CPU, local).

Calling follows the incremental metric (delta_incr = dJ_ABC_opt - (dJ_pair + dC)):
  thresholds = 95th percentile of the matched 3-way null (lineage-mismatched triples),
  floored at the paper's 0.15 / Z 4.
  synergistic = delta_incr > d_thresh & Wilcoxon p < 1e-3
  hard = synergistic & maxZ > z_thresh
"""
import os, argparse
import numpy as np
import pandas as pd


def main(a):
    TAB = a.out
    os.makedirs(TAB, exist_ok=True)

    # ── read target and null summaries ────────────────────────────────────
    t = pd.read_csv(os.path.join(a.triples, f"summary__{a.ct}.tsv"), sep="\t")
    n = pd.read_csv(os.path.join(a.null, f"summary__{a.ct}.tsv"), sep="\t")

    # ── null thresholds ───────────────────────────────────────────────────
    d95 = float(np.nanpercentile(n.delta_incr, 95))
    z95 = float(np.nanpercentile(n.maxZ, 95))
    d_thr = max(0.15, d95)
    z_thr = max(4.0, z95)

    with open(os.path.join(TAB, "null_triple_thresholds.txt"), "w") as fh:
        fh.write(f"3-way lineage-mismatched null: n_triples={len(n)}\n"
                 f"delta_incr: mean={n.delta_incr.mean():.3f} sd={n.delta_incr.std():.3f} p95={d95:.3f}\n"
                 f"maxZ:       mean={n.maxZ.mean():.2f} sd={n.maxZ.std():.2f} p95={z95:.2f}\n"
                 f"applied: delta_incr>{d_thr:.3f} & wilcoxon_p<1e-3 -> synergistic; "
                 f"& maxZ>{z_thr:.2f} -> hard\n")
    print(open(os.path.join(TAB, "null_triple_thresholds.txt")).read())

    # ── calling ───────────────────────────────────────────────────────────
    t["synergistic"] = (t.delta_incr > d_thr) & (t.wilcoxon_p < 1e-3)
    t["hard"] = t.synergistic & (t.maxZ > z_thr)
    t["call"] = np.select([t.hard, t.synergistic], ["hard", "syn"], "none")

    # ── per-triple: pick the anchor that gives the best incremental delta ─
    t["triple"] = t.apply(
        lambda r: "×".join(sorted([r.idA, r.idB, r.idC])), axis=1
    )
    best_idx = t.groupby("triple")["delta_incr"].idxmax()
    best = t.loc[best_idx].copy()
    best.to_csv(os.path.join(TAB, "best_triple_arrangements.tsv"),
                sep="\t", index=False)

    # ── full calls table ──────────────────────────────────────────────────
    t.to_csv(os.path.join(TAB, "triple_calls.tsv"), sep="\t", index=False)

    # ── print summary ─────────────────────────────────────────────────────
    pd.set_option("display.width", 280)
    cols = ["idA", "idB", "idC", "dJ_pair", "dC", "dS_incr", "dJ_triple_opt",
            "delta_incr", "delta_full", "opt_layout", "opt_gap", "maxZ",
            "wilcoxon_p", "frac_pos", "call"]
    print(t[[c for c in cols if c in t.columns]].round(3).to_string(index=False))

    print(f"\nBest anchor per triple:")
    for _, r in best.iterrows():
        print(f"  {r.triple}: anchor={r.idA}×{r.idB}, C={r.idC}, "
              f"delta_incr={r.delta_incr:+.3f}, layout={r.opt_layout}@{r.opt_gap}bp, "
              f"Z={r.maxZ:.1f}, call={r.call}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True, help="dir with summary__<ct>.tsv from the target run")
    p.add_argument("--null", required=True, help="dir with summary__<ct>.tsv from the null run")
    p.add_argument("--ct", default="DE", help="cell type the triples were run in")
    p.add_argument("-o", "--out", required=True, help="output tables dir")
    main(p.parse_args())
