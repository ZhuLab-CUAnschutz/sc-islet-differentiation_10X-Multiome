#!/usr/bin/env python
"""Aggregate and call a FOCUS pairwise screen (CPU).

Calling here follows the hero-run convention, NOT the all-pairs one: thresholds are the 95th
percentile of a matched lineage-mismatched null run at the same fidelity, floored at the paper's
0.15 / Z 4.
    synergistic = delta > d_thresh & wilcoxon_p < 1e-3
    hard        = synergistic & maxZ > z_thresh
    soft        = the sweep's own flag (delta > 0.15 at some 20-150 bp arrangement)
    frac_pos    = share of backgrounds with joint > additive at the optimum (reported, not gated)

A focus screen has too few pairs to fit the in-matrix empirical null that downstream.py uses for
the all-pairs sweep, which is why the matched external null is required here.

Writes tables/calls_long.tsv in the schema build_higher_order_inputs.py consumes, so a focus screen
feeds straight into the 3-way pipeline.

Generalized from scratch/2026_09_06_de_synergy/scripts/analyze_de4.py (the DE 4-TF screen).
"""
import os, argparse
import numpy as np, pandas as pd


def main(a):
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv(os.path.join(a.results, f"summary__{a.ct}.tsv"), sep="\t")
    null = pd.read_csv(a.null, sep="\t")

    d95, z95 = float(np.nanpercentile(null.delta, 95)), float(np.nanpercentile(null.maxZ, 95))
    d_thr, z_thr = max(0.15, d95), max(4.0, z95)
    thresh_txt = (f"{a.ct} lineage-mismatched null: n_pairs={len(null)} nbg={int(null.nbg.iloc[0])}\n"
                  f"delta: mean={null.delta.mean():.3f} sd={null.delta.std():.3f} p95={d95:.3f}\n"
                  f"maxZ:  mean={null.maxZ.mean():.2f} sd={null.maxZ.std():.2f} p95={z95:.2f}\n"
                  f"applied: delta>{d_thr:.3f} & wilcoxon_p<1e-3 -> synergistic; "
                  f"& maxZ>{z_thr:.2f} -> hard\n")
    with open(os.path.join(a.out, "null_thresholds.txt"), "w") as fh:
        fh.write(thresh_txt)

    def from_npz(r):
        """dA/dB and frac_pos live only in the per-pair grid, not the summary."""
        z = np.load(os.path.join(a.results, f"{r.idA}__{r.idB}__{r.ct}.npz"))
        return pd.Series({"dA": float(z["dA"]), "dB": float(z["dB"]),
                          "frac_pos": float((z["best_dj_seq"] > z["indep_seq"]).mean())})

    s = pd.concat([s, s.apply(from_npz, axis=1)], axis=1)
    s["synergistic"] = (s.delta > d_thr) & (s.wilcoxon_p < 1e-3)
    s["hard_cal"] = s.synergistic & (s.maxZ > z_thr)
    s["call"] = np.select([s.hard_cal, s.synergistic & s.soft, s.synergistic],
                          ["hard", "soft", "syn"], "none")

    by = pd.read_csv(os.path.join(a.inputs, "pairs_by_set.tsv"), sep="\t")
    long = by.merge(s.drop(columns=["ct"]), on=["idA", "idB"], how="left")

    # optional: join the all-pairs first-pass call for the same pairs, as a cross-check
    if a.firstpass and os.path.exists(a.firstpass):
        fp = pd.read_csv(a.firstpass, sep="\t")
        fp = fp[fp.ct == a.ct][["idA", "idB", "delta", "maxZ", "q_delta", "call"]]
        fp.columns = ["idA", "idB", "fp_delta", "fp_maxZ", "fp_q_delta", "fp_call"]
        # the all-pairs sweep stores each pair in catalog order, so join on the unordered key
        fp["key"] = fp.apply(lambda r: "|".join(sorted([r.idA, r.idB])), axis=1)
        fp = fp.drop(columns=["idA", "idB"])
        long["key"] = long.apply(lambda r: "|".join(sorted([r.idA, r.idB])), axis=1)
        long = long.merge(fp, on="key", how="left").drop(columns=["key"])

    long["pair"] = long.tfA + "×" + long.tfB
    long.to_csv(os.path.join(a.out, "calls_long.tsv"), sep="\t", index=False)

    cols = ["pair", "idA", "idB", "dA", "dB", "dS", "dJ_opt", "delta", "opt_orient", "opt_gap",
            "opt_center_center", "maxZ", "wilcoxon_p", "frac_pos", "call"]
    het = long[long.kind == "heterotypic"]
    if het["set"].nunique() > 1:
        side = het.pivot(index="pair", columns="set", values=[c for c in cols if c != "pair"])
        side.columns = [f"{x}|{y.split('_')[0]}" for x, y in side.columns]
        side = side.reset_index().sort_values("pair")
        side.to_csv(os.path.join(a.out, "pairs_by_set.tsv"), sep="\t", index=False)

    long[long.kind == "homodimer"].drop_duplicates(["idA", "idB"])[cols + ["set"]] \
        .to_csv(os.path.join(a.out, "homodimers.tsv"), sep="\t", index=False)

    ins_path = os.path.join(a.inputs, "inserts.tsv")
    if os.path.exists(ins_path):
        ins = pd.read_csv(ins_path, sep="\t")
        homo = s[s.idA == s.idB].set_index("idA")
        ins[f"dA_{a.ct}"] = ins.id.map(homo.dA)
        ins.to_csv(os.path.join(a.out, "inserts.tsv"), sep="\t", index=False)

    pd.set_option("display.width", 250)
    print(thresh_txt)
    show = ["set", "pair", "dA", "dB", "dS", "dJ_opt", "delta", "opt_orient", "opt_gap",
            "maxZ", "wilcoxon_p", "frac_pos", "call"] + [c for c in ("fp_delta", "fp_call") if c in het]
    print(het[show].round(3).to_string(index=False))
    print(f"\n[done] {len(long)} rows -> {a.out}/calls_long.tsv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="sweep output dir (summary__<ct>.tsv + grids)")
    p.add_argument("--null", required=True, help="matched lineage-mismatched null summary TSV")
    p.add_argument("--inputs", required=True, help="dir with pairs_by_set.tsv [+ inserts.tsv]")
    p.add_argument("--ct", required=True)
    p.add_argument("-o", "--out", required=True, help="output tables dir")
    p.add_argument("--firstpass", default=None, help="optional all-pairs calls_long.tsv to cross-check against")
    main(p.parse_args())
