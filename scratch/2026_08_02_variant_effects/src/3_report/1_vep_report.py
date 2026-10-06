"""Stage 3.1: VEP report from a per-trait celltype_vep_logfc matrix.

Rebuilds the reference `celltype_vep_logfc.*` report: a PCA of cell types in
variant-effect space, colored and summarized by four facets (cell type,
developmental stage, lineage, endocrine status), a significant-variant summary,
and specificity tables. PCA is fit on the cell-type profiles (cell types are the
samples, variants are the features), so each point is a cell type.

Developmental stage is the trajectory position (DE -> GT -> PFG -> PE -> ENP ->
SC_immature -> SC_mature), not calendar day: cell types span multiple days.

Usage:
    python src/3_report/1_vep_report.py --config config/config.yaml [--trait T2D]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)
PREFIX = "celltype_vep_logfc"
FACET_KEYS = ["stage", "lineage", "endocrine_category"]  # dev_stage, lineage, endocrine


def fit_pca(logfc: pd.DataFrame, cell_types: list[str], seed: int):
    """Fit PCA on cell-type profiles. Returns (scores[PC1,PC2] indexed by ct, variance%)."""
    X = logfc[cell_types].T.values
    pca = PCA(n_components=2, random_state=seed)
    scores = pca.fit_transform(X)
    return (pd.DataFrame(scores, index=cell_types, columns=["PC1", "PC2"]),
            pca.explained_variance_ratio_ * 100)


def plot_scatter(scores, var_pct, column, colors, order, title, out_path):
    """PCA scatter of cell types colored by a categorical facet."""
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    present = [g for g in order if g in set(scores[column])]
    sns.scatterplot(data=scores, x="PC1", y="PC2", hue=column, palette=colors,
                    hue_order=present, s=95, edgecolor="black", linewidth=0.6, ax=ax)
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% var)")
    ax.legend(title=title, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    savefig(fig, out_path)


def plot_scatter_celltype(scores, var_pct, ct_colors, out_path):
    """PCA scatter with each cell type in its own color, labeled."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for ct, row in scores.iterrows():
        ax.scatter(row["PC1"], row["PC2"], color=ct_colors.get(ct, "0.5"),
                   s=95, edgecolor="black", linewidth=0.6)
        ax.annotate(ct, (row["PC1"], row["PC2"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% var)")
    savefig(fig, out_path)


def plot_boxplots(scores, column, colors, order, title, out_path):
    """Boxplots of PC1 and PC2 grouped by a facet."""
    present = [g for g in order if g in set(scores[column])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        sns.boxplot(data=scores, x=column, y=pc, order=present, hue=column,
                    palette=colors, legend=False, ax=ax)
        sns.stripplot(data=scores, x=column, y=pc, order=present, color="0.2", size=4, ax=ax)
        ax.set_xlabel(title)
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_ridgeline(scores, pc, column, order, colors, out_path):
    """Ridgeline of a PC's cell-type scores by facet.

    Facet values with fewer than 2 cell types cannot form a KDE; they render as a
    marker at the value and a warning is logged.
    """
    present = [g for g in order if g in set(scores[column])]
    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(4.8, 0.7 * n + 1), sharex=True)
    axes = np.atleast_1d(axes)
    xmin, xmax = scores[pc].min(), scores[pc].max()
    pad = 0.1 * (xmax - xmin + 1e-9)
    grid = np.linspace(xmin - pad, xmax + pad, 200)
    for ax, val in zip(axes, present):
        vals = scores.loc[scores[column] == val, pc].values
        color = colors.get(val, "0.5")
        if len(vals) >= 2 and np.ptp(vals) > 0:
            ax.fill_between(grid, gaussian_kde(vals)(grid), color=color, alpha=0.7,
                            edgecolor="black")
        else:
            log.warning("ridgeline %s %s=%s: %d point(s), no KDE", pc, column, val, len(vals))
            ax.scatter(vals, np.zeros_like(vals), color=color, edgecolor="black", zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(val, rotation=0, ha="right", va="center", fontsize=8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel(f"{pc} Score")
    fig.tight_layout()
    savefig(fig, out_path)


def significant_counts(pval: pd.DataFrame, threshold: float) -> pd.Series:
    """Number of variants with p <= threshold per cell type."""
    return (pval <= threshold).sum(axis=0)


def plot_sig_summary(counts, md, facets, threshold, out_path):
    """Multi-page significant-variant summary: bars and boxplots per facet."""
    df = counts.rename("n_sig").to_frame().join(md)
    df = df.sort_values("n_sig", ascending=False)
    xlab = f"Number of significant variants (p <= {threshold})"
    with PdfPages(out_path) as pdf:
        for f in facets:
            col, colors, title = f["column"], f["colors"], f["title"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df.index, df["n_sig"], color=df[col].map(colors), edgecolor="black")
            ax.set_ylabel(xlab); ax.set_xlabel("Cell type")
            ax.tick_params(axis="x", rotation=90)
            ax.set_title(f"Significant variants per cell type (colored by {title})")
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
            ax.legend(handles, list(colors), title=title, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            order = [g for g in f["order"] if g in set(df[col])]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            sns.boxplot(data=df, x=col, y="n_sig", order=order, hue=col,
                        palette=colors, legend=False, ax=ax)
            sns.stripplot(data=df, x=col, y="n_sig", order=order, color="0.2", size=4, ax=ax)
            ax.set_xlabel(title); ax.set_ylabel(xlab)
            ax.set_title(f"Significant variants by {title}")
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_specificity_tables(logfc, md, out_dir, prefix):
    """Top-variance, cell-type-specific, and grouping-specific variant tables."""
    logfc.var(axis=1).sort_values(ascending=False).rename("logfc_variance").to_csv(
        out_dir / f"{prefix}.top_variance_variants.tsv", sep="\t")
    row_std = logfc.std(axis=1).replace(0, np.nan)
    z = logfc.sub(logfc.mean(axis=1), axis=0).div(row_std, axis=0)
    pd.DataFrame({
        "variant_id": logfc.index, "top_cell_type": z.idxmax(axis=1).values,
        "top_z": z.max(axis=1).values, "top_logfc": logfc.max(axis=1).values,
    }).sort_values("top_z", ascending=False).to_csv(
        out_dir / f"{prefix}.celltype_specific_variants.tsv", sep="\t", index=False)
    group_means = logfc.T.groupby(md["grouping"]).mean().T
    group_means.insert(0, "top_grouping", group_means.idxmax(axis=1))
    group_means.to_csv(out_dir / f"{prefix}.cluster_specific_variants.tsv",
                       sep="\t", index_label="variant_id")


def build_trait_report(cfg, trait, md, facets, ct_colors) -> None:
    key = trait["key"]
    trait_dir = io.sandbox_path(cfg, f"results/{key}")
    logfc = pd.read_csv(trait_dir / "celltype_vep_logfc.csv", index_col="variant_id")
    pval = pd.read_csv(trait_dir / "celltype_vep_pval.csv", index_col="variant_id")
    cell_types = [c for c in md.index if c in logfc.columns]
    out = trait_dir / "vep_report_out"
    out.mkdir(parents=True, exist_ok=True)

    scores, var_pct = fit_pca(logfc, cell_types, cfg["seed"])
    scores = scores.join(md)  # adds dev_stage, lineage, endocrine, grouping, color
    p = out / PREFIX

    plot_scatter_celltype(scores, var_pct, ct_colors, f"{p}.PCA_celltypes_by_celltype.pdf")
    for fkey, f in zip(FACET_KEYS, facets):
        plot_scatter(scores, var_pct, f["column"], f["colors"], f["order"], f["title"],
                     f"{p}.PCA_celltypes_by_{fkey}.pdf")
        plot_boxplots(scores, f["column"], f["colors"], f["order"], f["title"],
                      f"{p}.PCA_celltypes_boxplots_by_{fkey}.pdf")
        for pc in ["PC1", "PC2"]:
            plot_ridgeline(scores, pc, f["column"], f["order"], f["colors"],
                           f"{p}.PCA_celltypes_ridgeplot_{pc}_by_{fkey}.pdf")

    threshold = cfg["significance"]["pval_threshold"]
    plot_sig_summary(significant_counts(pval, threshold), md, facets, threshold,
                     f"{p}.sig_variant_summary.pdf")
    write_specificity_tables(logfc, md, out, PREFIX)
    log.info("%s: %d variants x %d cell types -> %s", key, *logfc.shape, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait")
    args = ap.parse_args()

    configure_matplotlib()
    sns.set_style("white")
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    facets = [metadata.facet(cfg, k) for k in FACET_KEYS]
    ct_colors = metadata.color_map(cfg)

    traits = [t for t in cfg["traits"] if not args.trait or t["key"] == args.trait]
    for trait in traits:
        build_trait_report(cfg, trait, md, facets, ct_colors)
    io.write_provenance(cfg, "3_report_1_vep_report", {
        "traits": [t["key"] for t in traits], "facets": FACET_KEYS,
        "pval_threshold": cfg["significance"]["pval_threshold"]})
    log.info("Done.")


if __name__ == "__main__":
    main()
