#!/usr/bin/env python
"""
05_contrib_trajectory.py — for each FIMO motif hit, the model's count-contribution
summed over the hit window, per cell type, across the differentiation trajectory.
Connects sequence motif occurrences (FIMO) to model-attributed importance and how
it changes over 'time'. Three views: absolute heatmap (robust scale), row z-score
(temporal pattern), per-TF trajectory lines, and construct-vs-enh_dinuc scatter.

    python 05_contrib_trajectory.py outputs/sox32_all.npz construct
Run under eugene_tools.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import plot_locus as pl
import chrombpnet_locus as cl

NPZ = sys.argv[1] if len(sys.argv) > 1 else "outputs/sox32_all.npz"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "construct"
FIG, OUT = "figures", "outputs"
res = pl.LocusResults(NPZ)
cts = res.ordered_celltypes()
cmap_ct = cl.color_map()
fimo = pd.read_csv(config.FIMO_HITS_CSV)
fimo = fimo[fimo["feature"] == "enhancer"].sort_values(["motif_id", "start"])  # enhancer hits

mat = pl.fimo_contribution_matrix(res, fimo, variant=VARIANT, signed=True)
mat.to_csv(f"{OUT}/{os.path.basename(NPZ).split('.')[0]}_fimo_contrib_{VARIANT}.csv", index=False)
print(f"[matrix] {len(mat)} enhancer FIMO hits x {len(cts)} cells")

M = mat.set_index("id")[cts]; sig = mat.set_index("id")["qvalue"] < 0.05
ylab = [f"{'* ' if sig[i] else '  '}{i}" for i in M.index]

def heatmap(values, vlim, title, fname, cbar):
    fig, ax = plt.subplots(figsize=(0.42 * len(cts) + 3, 0.34 * len(M) + 2))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=8)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(ylab, fontsize=7, fontfamily="monospace")
    for j, c in enumerate(cts):
        ax.get_xticklabels()[j].set_color(cmap_ct[c])
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.025, label=cbar)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{fname}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

heatmap(M.values, np.percentile(np.abs(M.values), 99),
        f"FIMO-hit contribution across trajectory [{VARIANT}] (*=FIMO q<0.05)",
        f"sox32_fimo_contrib_heatmap_{VARIANT}", "sum contribution over hit")
Z = M.sub(M.mean(1), axis=0).div(M.std(1).replace(0, np.nan), axis=0).fillna(0)
heatmap(Z.values, 2.5, f"FIMO-hit contribution — row z-scored (temporal pattern) [{VARIANT}]",
        f"sox32_fimo_contrib_heatmap_{VARIANT}_zscore", "z across cells")
print("wrote heatmaps")

# per-TF trajectory lines
motifs = list(config.FIMO_TF_NAMES.values())
disp = {v: k for k, v in config.FIMO_TF_NAMES.items()}
fig, axes = plt.subplots(1, len(motifs), figsize=(5.2 * len(motifs), 4), sharey=True)
x = np.arange(len(cts))
for ax, m in zip(axes, motifs):
    col = config.TF_COLORS[disp[m]]
    for _, r in mat[mat.motif == m].iterrows():
        s = r["qvalue"] < 0.05
        ax.plot(x, [r[c] for c in cts], "-o", ms=3, lw=2 if s else 0.8,
                alpha=0.95 if s else 0.35, color=col, label=r["id"] if s else None)
    ax.axhline(0, color="0.6", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(cts, rotation=90, fontsize=7)
    for j, c in enumerate(cts):
        ax.get_xticklabels()[j].set_color(cmap_ct[c])
    ax.set_title(disp[m], color=col)
    if mat[mat.motif == m]["qvalue"].lt(0.05).any():
        ax.legend(fontsize=6, loc="best")
axes[0].set_ylabel("sum count-contribution over hit")
fig.suptitle(f"Per-motif-hit contribution across the trajectory [{VARIANT}] (bold=q<0.05)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/sox32_fimo_contrib_trajectory_{VARIANT}.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/sox32_fimo_contrib_trajectory_{VARIANT}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote trajectory lines")

# construct vs enh_dinuc (does removing the GC-rich flank change hit importance?)
if "enh_dinuc" in res.variants:
    m2 = pl.fimo_contribution_matrix(res, fimo, variant="enh_dinuc", signed=True)
    top = mat.assign(a=mat[cts].abs().max(1)).nlargest(10, "a")["id"].tolist()
    a = mat.set_index("id")[cts].abs().mean(1); b = m2.set_index("id")[cts].abs().mean(1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(a[top], b[top], s=30, color="#333")
    for i in top:
        ax.annotate(i, (a[i], b[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    lim = max(a[top].max(), b[top].max()) * 1.1; ax.plot([0, lim], [0, lim], "--", color="0.6", lw=0.8)
    ax.set_xlabel(f"{VARIANT} (mean |contrib|)"); ax.set_ylabel("enh_dinuc (mean |contrib|)")
    ax.set_title("Top FIMO hits: construct vs enhancer-only(dinuc flanks)")
    fig.tight_layout(); fig.savefig(f"{FIG}/sox32_fimo_contrib_construct_vs_enhdinuc.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/sox32_fimo_contrib_construct_vs_enhdinuc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote construct-vs-enh_dinuc scatter")
print("DONE")
