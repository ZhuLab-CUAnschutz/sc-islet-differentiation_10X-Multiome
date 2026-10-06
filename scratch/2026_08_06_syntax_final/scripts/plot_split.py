#!/usr/bin/env python
"""Explicit view of the endocrine / non-endocrine block split: every synergistic pair (row) shown by
WHICH cell types it is called synergistic in (q<0.05), grouped by the block it was assigned to.

A pair enters the endocrine block if synergistic in >=1 endocrine CT, the non-endocrine block if
synergistic in >=1 non-endocrine CT -> three membership groups: endocrine-only / shared / non-endocrine-only.
Filled cell = synergistic in that CT, colored by the CT's endocrine status. This makes membership literal.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list

SF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
calls = pd.read_csv(os.path.join(SF, "tables/calls_long.tsv"), sep="\t")
ctm = pd.read_csv(os.path.join(SF, "config/cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
CTORDER = list(ctm.cell_type)
ENDO = set(ctm.loc[ctm.endocrine == "endocrine", "cell_type"])
ENDO_C, NON_C, EMPTY = "#5e4fa2", "#f46d43", "#eeeeee"

syn = (calls.pivot(index="pair", columns="ct", values="synergistic")
            .reindex(columns=CTORDER).fillna(False).astype(bool))
syn = syn[syn.any(axis=1)]                                    # 660 pairs synergistic in >=1 CT
endo_cols = [c for c in CTORDER if c in ENDO]
non_cols = [c for c in CTORDER if c not in ENDO]
in_endo = syn[endo_cols].any(axis=1); in_non = syn[non_cols].any(axis=1)
tag = pd.Series(np.where(in_endo & in_non, "shared",
        np.where(in_endo, "endocrine", "non_endocrine")), index=syn.index)

# order rows: group by block, cluster within group on the synergy pattern (jaccard) for readability
order = []; bounds = []; labels = []
for name in ["endocrine", "shared", "non_endocrine"]:
    rows = syn.index[tag == name]
    if len(rows) > 3:
        L = linkage(syn.loc[rows].values.astype(float), method="average", metric="jaccard")
        rows = [rows[i] for i in leaves_list(L)]
    order += list(rows)
    bounds.append(len(order))
    labels.append(f"{name.replace('_',' ')}-only" if name != "shared" else "shared")
labels = [f"endocrine-only ({int((tag=='endocrine').sum())})",
          f"shared ({int((tag=='shared').sum())})",
          f"non-endocrine-only ({int((tag=='non_endocrine').sum())})"]

M = syn.loc[order]
# RGB image: white where not syn; endocrine/non-endocrine color where syn
rgb = np.ones((len(M), len(CTORDER), 3))
def hex2rgb(h): return tuple(int(h[i:i+2], 16)/255 for i in (1, 3, 5))
eC, nC, emp = hex2rgb(ENDO_C), hex2rgb(NON_C), hex2rgb(EMPTY)
for j, c in enumerate(CTORDER):
    col = eC if c in ENDO else nC
    v = M[c].values
    rgb[~v, j] = emp
    rgb[v, j] = col

fig, ax = plt.subplots(figsize=(8.6, 11))
ax.imshow(rgb, aspect="auto", interpolation="nearest")
# group separators + labels
prev = 0
for b, lab in zip(bounds, labels):
    if b < len(M): ax.axhline(b - 0.5, color="#111", lw=1.6)
    ax.text(-1.7, (prev + b) / 2 - 0.5, lab, rotation=90, va="center", ha="center",
            fontsize=11, fontweight="bold")
    prev = b
# column labels + endocrine status underline
ax.set_xticks(range(len(CTORDER)))
ax.set_xticklabels(CTORDER, rotation=90, fontsize=8)
for j, c in enumerate(CTORDER):
    ax.get_xticklabels()[j].set_color(ENDO_C if c in ENDO else NON_C)
ax.set_yticks([])
ax.margins(x=0)
ax.set_title("Block membership — which cell types each pair is synergistic in (q<0.05)\n"
             "grouped by assigned block; fill color = cell type's endocrine status", fontsize=11)
ax.legend(handles=[Patch(facecolor=ENDO_C, label="synergistic · endocrine CT"),
                   Patch(facecolor=NON_C, label="synergistic · non-endocrine CT"),
                   Patch(facecolor=EMPTY, label="not synergistic")],
          loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(SF, "figures", f"block_membership.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[membership] endocrine-only={int((tag=='endocrine').sum())} shared={int((tag=='shared').sum())} "
      f"non_endocrine-only={int((tag=='non_endocrine').sum())} -> figures/block_membership.png")
