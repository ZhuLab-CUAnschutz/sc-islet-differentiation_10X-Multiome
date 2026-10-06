#!/usr/bin/env python
"""DE-only analysis of the full-fidelity CRESTED multitask target and null runs."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.path.join(HERE, "..")
TAB = os.path.join(WD, "tables")
FIG = os.path.join(WD, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

targets = pd.read_csv(os.path.join(WD, "results", "mt_de4", "summary__all.tsv"), sep="\t")
null = pd.read_csv(os.path.join(WD, "results", "mt_null_full", "summary__all.tsv"), sep="\t")
assert len(targets) == 19 * 22 and len(null) == 42 * 22
assert targets.groupby(["idA", "idB"]).ct.nunique().eq(22).all()
assert null.groupby(["idA", "idB"]).ct.nunique().eq(22).all()

targets = targets[targets.ct == "DE"].copy()
null = null[null.ct == "DE"].copy()
d95 = float(np.percentile(null.delta, 95))
z95 = float(np.percentile(null.maxZ, 95))
dthr, zthr = max(0.15, d95), max(4.0, z95)
targets["synergistic"] = (targets.delta > dthr) & (targets.wilcoxon_p < 1e-3)
targets["hard"] = targets.synergistic & (targets.maxZ > zthr)
targets["call"] = np.select([targets.hard, targets.synergistic], ["hard", "syn"], "none")
targets["delta_over_null95"] = targets.delta / dthr

sets = pd.read_csv(os.path.join(WD, "inputs", "pairs_by_set.tsv"), sep="\t")
long = sets.merge(targets.drop(columns="ct"), on=["idA", "idB"], how="left")
long["pair"] = long.tfA + "×" + long.tfB
long.to_csv(os.path.join(TAB, "mt_calls_long.tsv"), sep="\t", index=False)

with open(os.path.join(TAB, "mt_null_thresholds.txt"), "w") as out:
    out.write(
        f"Full-fidelity multitask DE null: n_pairs={len(null)}, n=100 backgrounds, gaps=0-200 by 1 bp\n"
        f"delta raw-count p95={d95:.4f}; maxZ p95={z95:.4f}\n"
        f"call: delta>{dthr:.4f} and Wilcoxon p<1e-3; hard also maxZ>{zthr:.4f}\n"
    )

order = ["EOMES×MIXL1", "EOMES×SOX17", "MIXL1×SOX17", "EOMES×FOXH1", "MIXL1×FOXH1", "SOX17×FOXH1"]
het = long[long.kind == "heterotypic"].copy()
colors = {"set1_catalog+JFOXH1": "#5e4fa2", "set2_jaspar": "#f4a582"}
labels = {"set1_catalog+JFOXH1": "catalog CWM (+ JASPAR FOXH1)", "set2_jaspar": "JASPAR"}
fig, ax = plt.subplots(figsize=(7.6, 3.8))
x = np.arange(len(order)); width = 0.36
for i, name in enumerate(colors):
    d = het[het.set == name].set_index("pair").reindex(order)
    ax.bar(x + (i - 0.5) * width, d.delta_over_null95, width * 0.94,
           color=colors[name], label=labels[name], edgecolor="none")
    for xi, (v, call) in enumerate(zip(d.delta_over_null95, d.call)):
        if call != "none":
            ax.text(x[xi] + (i - 0.5) * width, v + 0.02, call, ha="center", fontsize=7)
ax.axhline(1, color="#777", linestyle="--", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("multitask synergy Δ / DE-null p95")
ax.set_title("DE head of the CRESTED multitask model (full-fidelity matched null)", loc="left")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig5_multitask_delta_by_pair.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)

show = het[["set", "pair", "dS", "dJ_opt", "delta", "delta_over_null95", "opt_orient",
            "opt_gap", "maxZ", "wilcoxon_p", "call"]].sort_values(["set", "delta"], ascending=[True, False])
show.to_csv(os.path.join(TAB, "mt_pairs_de.tsv"), sep="\t", index=False)
print(open(os.path.join(TAB, "mt_null_thresholds.txt")).read())
print(show.round(4).to_string(index=False))
