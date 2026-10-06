#!/usr/bin/env python
"""Clustermap of synergistic motif pairs (rows) x cell types (cols), from the syn_allct sweep
(each pair's optimal arrangement RE-optimized per cell type). Two views:
  A) raw Δ = ΔJ−ΔS (synergy magnitude)
  B) row-standardized z(Δ) across CTs (specificity pattern, magnitude removed)
Both-axis hierarchical clustering; column color bar by canonical cell-type color. Emits PNGs +
the clustered matrices + row block (cluster) assignments so blocks can be inspected. CPU only.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage

SYN = os.path.dirname(os.path.abspath(__file__)); FIG = f"{SYN}/figures"
os.makedirs(FIG, exist_ok=True)

# ---- matrix: pairs x CT of Δ (re-optimized per CT) ----
d = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/syn_allct/summary__*.tsv")], ignore_index=True)
meta = pd.read_csv(f"{SYN}/metadata.tsv", sep="\t")
tf = {r.short_id: (r.curator_tf if isinstance(r.curator_tf, str) and r.curator_tf.strip() else r.short_id) for r in meta.itertuples()}
d["pair"] = d.idA.map(lambda s: tf.get(s, s)) + "×" + d.idB.map(lambda s: tf.get(s, s)) + "  [" + d.idA + "·" + d.idB + "]"
M = d.pivot_table(index="pair", columns="ct", values="delta")
# canonical CT order + colors
ctm = pd.read_csv(f"{SYN}/cell_type_metadata.tsv", sep="\t").sort_values("display_order")
ctorder = [c for c in ctm.cell_type if c in M.columns]
ctcol = dict(zip(ctm.cell_type, ctm.color))
M = M[ctorder]
M = M.dropna(how="any")                       # keep pairs measured in all CTs
print(f"matrix: {M.shape[0]} pairs x {M.shape[1]} CTs")
M.to_csv(f"{FIG}/synergy_ct_matrix_delta.tsv", sep="\t")
col_colors = pd.Series({c: ctcol.get(c, "#999") for c in M.columns}, name="cell type")

def clustermap(mat, fname, title, cmap, center, zscore_note=""):
    g = sns.clustermap(mat, cmap=cmap, center=center, col_colors=col_colors,
                       col_cluster=True, row_cluster=True, method="average", metric="euclidean",
                       figsize=(11, min(60, 0.14 * mat.shape[0] + 3)), yticklabels=(mat.shape[0] <= 200),
                       xticklabels=True, cbar_kws={"label": title},
                       dendrogram_ratio=(.10, .06), colors_ratio=0.012)
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=8)
    if mat.shape[0] <= 200: g.ax_heatmap.tick_params(axis="y", labelsize=5)
    g.fig.suptitle(f"Synergistic motif pairs × cell types — {title}{zscore_note}", y=1.005, fontsize=12)
    g.savefig(fname, dpi=170, bbox_inches="tight"); plt.close(g.fig)
    return g

# A) raw Δ magnitude
vmax = float(np.nanpercentile(np.abs(M.values), 99))
gA = clustermap(M, f"{FIG}/fig_clustermap_delta.png", "Δ = ΔJ−ΔS (synergy magnitude)",
                cmap="RdBu_r", center=0)
# B) row-standardized z(Δ) (specificity pattern)
Z = M.sub(M.mean(1), axis=0).div(M.std(1).replace(0, np.nan), axis=0).fillna(0)
gB = clustermap(Z, f"{FIG}/fig_clustermap_delta_rowz.png", "z(Δ) per pair across CTs (specificity)",
                cmap="RdBu_r", center=0)

# ---- row blocks (cut the raw-Δ dendrogram) for inspection ----
L = linkage(M.values, method="average", metric="euclidean")
for k in (8, 12, 16):
    lab = fcluster(L, k, criterion="maxclust")
    blocks = pd.DataFrame({"pair": M.index, f"block_k{k}": lab})
    if k == 12:
        # summarize each block: size + which CTs it peaks in (mean Δ per CT, top 3)
        summ = []
        for b, grp in blocks.groupby("block_k12"):
            sub = M.loc[grp["pair"]]
            top = sub.mean(0).sort_values(ascending=False).head(3)
            summ.append({"block": b, "n_pairs": len(grp),
                         "top_CTs": ", ".join(f"{c}({v:.2f})" for c, v in top.items()),
                         "example_pairs": "; ".join(grp["pair"].str.split("  ").str[0].head(4))})
        pd.DataFrame(summ).to_csv(f"{FIG}/synergy_blocks_k12_summary.tsv", sep="\t", index=False)
        blocks.to_csv(f"{FIG}/synergy_blocks_k12.tsv", sep="\t", index=False)
        print("\n=== k=12 blocks (raw Δ) ===")
        print(pd.DataFrame(summ).to_string(index=False))
print(f"\nwrote clustermaps + matrices to {FIG}")
