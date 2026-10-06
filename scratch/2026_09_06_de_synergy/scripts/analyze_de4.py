#!/usr/bin/env python
"""Aggregate + call the DE 4-TF synergy run (CPU, local).
Calls follow the hero-run convention (2026_07_19 plot_syntax.py): thresholds = 95th percentile of the
DE lineage-mismatched null (42 pairs, same fidelity: n=100, gaps 0-200, 1-bp), floored at the paper's
0.15 / Z 4. synergistic = delta > d_thresh & Wilcoxon p < 1e-3; hard = synergistic & maxZ > z_thresh;
soft = script flag (delta>0.15 at some 20-150 bp arrangement). frac_pos = share of backgrounds with
joint > additive at the optimum (first-pass robustness metric)."""
import os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); WD = os.path.join(HERE, "..")
IN, RES, TAB = (os.path.join(WD, d) for d in ("inputs", "results/de4", "tables")); os.makedirs(TAB, exist_ok=True)
FP = os.path.join(WD, "..", "2026_08_06_syntax_final", "tables", "calls_long.tsv")   # first pass (n=32, gaps 0-50)

s = pd.read_csv(os.path.join(RES, "summary__DE.tsv"), sep="\t")
null = pd.read_csv(os.path.join(IN, "null_DE_summary.tsv"), sep="\t")
d95, z95 = float(np.nanpercentile(null.delta, 95)), float(np.nanpercentile(null.maxZ, 95))
d_thr, z_thr = max(0.15, d95), max(4.0, z95)
with open(os.path.join(TAB, "null_thresholds.txt"), "w") as fh:
    fh.write(f"DE lineage-mismatched null: n_pairs={len(null)} nbg={int(null.nbg.iloc[0])}\n"
             f"delta: mean={null.delta.mean():.3f} sd={null.delta.std():.3f} p95={d95:.3f}\n"
             f"maxZ:  mean={null.maxZ.mean():.2f} sd={null.maxZ.std():.2f} p95={z95:.2f}\n"
             f"applied: delta>{d_thr:.3f} & wilcoxon_p<1e-3 -> synergistic; & maxZ>{z_thr:.2f} -> hard\n")

def from_npz(r):                      # dA/dB (single-insert effects) live only in the grids; frac_pos from per-seq deltas
    z = np.load(os.path.join(RES, f"{r.idA}__{r.idB}__{r.ct}.npz"))
    return pd.Series({"dA": float(z["dA"]), "dB": float(z["dB"]), "frac_pos": float((z["best_dj_seq"] > z["indep_seq"]).mean())})
s = pd.concat([s, s.apply(from_npz, axis=1)], axis=1)
s["synergistic"] = (s.delta > d_thr) & (s.wilcoxon_p < 1e-3)
s["hard_cal"] = s.synergistic & (s.maxZ > z_thr)
s["call"] = np.select([s.hard_cal, s.synergistic & s.soft, s.synergistic], ["hard", "soft", "syn"], "none")

by = pd.read_csv(os.path.join(IN, "pairs_by_set.tsv"), sep="\t")
long = by.merge(s.drop(columns=["ct"]), on=["idA", "idB"], how="left")
if os.path.exists(FP):
    fp = pd.read_csv(FP, sep="\t"); fp = fp[fp.ct == "DE"][["idA", "idB", "delta", "maxZ", "q_delta", "call"]]
    fp.columns = ["idA", "idB", "fp_delta", "fp_maxZ", "fp_q_delta", "fp_call"]
    fp["key"] = fp.apply(lambda r: "|".join(sorted([r.idA, r.idB])), axis=1); fp = fp.drop(columns=["idA", "idB"])
    long["key"] = long.apply(lambda r: "|".join(sorted([r.idA, r.idB])), axis=1)   # first pass stores pairs in catalog order
    long = long.merge(fp, on="key", how="left").drop(columns=["key"])
long["pair"] = long.tfA + "×" + long.tfB
long.to_csv(os.path.join(TAB, "calls_long.tsv"), sep="\t", index=False)

cols = ["pair", "idA", "idB", "dA", "dB", "dS", "dJ_opt", "delta", "opt_orient", "opt_gap", "opt_center_center", "maxZ", "wilcoxon_p", "frac_pos", "call"]
het = long[long.kind == "heterotypic"]
side = het.pivot(index="pair", columns="set", values=[c for c in cols if c not in ("pair",)])
side.columns = [f"{a}|{b.split('_')[0]}" for a, b in side.columns]; side = side.reset_index()
order = ["EOMES×MIXL1", "EOMES×SOX17", "MIXL1×SOX17", "EOMES×FOXH1", "MIXL1×FOXH1", "SOX17×FOXH1"]
side["pair"] = pd.Categorical(side.pair, order); side = side.sort_values("pair")
side.to_csv(os.path.join(TAB, "pairs_by_set.tsv"), sep="\t", index=False)
long[long.kind == "homodimer"].drop_duplicates(["idA", "idB"])[cols + ["set"]].to_csv(os.path.join(TAB, "homodimers.tsv"), sep="\t", index=False)

ins = pd.read_csv(os.path.join(IN, "inserts.tsv"), sep="\t")
homo = s[s.idA == s.idB].set_index("idA")
ins["dA_DE"] = ins.id.map(homo.dA); ins.to_csv(os.path.join(TAB, "inserts.tsv"), sep="\t", index=False)

pd.set_option("display.width", 250)
print(open(os.path.join(TAB, "null_thresholds.txt")).read())
print(het[["set", "pair", "dA", "dB", "dS", "dJ_opt", "delta", "opt_orient", "opt_gap", "maxZ", "wilcoxon_p", "frac_pos", "call"] + [c for c in ("fp_delta", "fp_call") if c in het]].round(3).to_string(index=False))
print("\nhomodimers"); print(long[long.kind == "homodimer"].drop_duplicates(["idA", "idB"])[["pair", "idA", "dA", "delta", "opt_orient", "opt_gap", "maxZ", "wilcoxon_p", "call"]].round(3).to_string(index=False))
print("\ninserts"); print(ins[["id", "tf", "consensus", "width", "dA_DE"]].round(3).to_string(index=False))
