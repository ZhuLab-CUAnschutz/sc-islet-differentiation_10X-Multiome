"""Stage 3.7: clustermaps of top-variable variants by effect across cell types.

For each trait, take the variants whose logfc varies most across the 22 cell types
and draw a clustered heatmap (variants x cell types, hierarchically clustered both
ways) so groups of variants with shared cell-type-effect patterns pop out. Cell-
type columns carry stage / lineage / endocrine color bars. Also aggregated views:
variants x developmental stage, x lineage, x grouping (mean logfc per group). A
pooled across-trait clustermap is written too.

CPU stage; env eugene_tools.

Usage:
    python src/3_report/7_variant_clustermaps.py --config config/config.yaml [--trait T2D --top-n 75]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib

log = setup_logging(__file__)
PREFIX = "celltype_vep_logfc"


def col_color_bars(cfg, cell_types):
    """DataFrame (index=cell_type) of Stage/Lineage/Endocrine colors for clustermap."""
    md = metadata.load_cell_type_metadata(cfg)
    _, stage_c = metadata.category(cfg, "dev_stage")
    _, lin_c = metadata.category(cfg, "lineage")
    _, endo_c = metadata.category(cfg, "endocrine")
    return pd.DataFrame({
        "Stage": md.loc[cell_types, "dev_stage"].map(stage_c),
        "Lineage": md.loc[cell_types, "lineage"].map(lin_c),
        "Endocrine": md.loc[cell_types, "endocrine"].map(endo_c),
    }, index=cell_types)


def save_clustermap(mat, out_path, col_colors=None, col_cluster=True, vmax=None):
    """Diverging clustermap of a variant x group logfc matrix centered at 0."""
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        log.warning("skip clustermap %s: shape %s", out_path.name, mat.shape)
        return
    vmax = vmax or np.nanpercentile(np.abs(mat.values), 98)
    g = sns.clustermap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                       col_colors=col_colors, col_cluster=col_cluster,
                       xticklabels=True, yticklabels=(mat.shape[0] <= 80),
                       figsize=(max(8, 0.35 * mat.shape[1] + 4), max(7, 0.12 * mat.shape[0] + 3)),
                       cbar_kws={"label": "logfc"}, dendrogram_ratio=(0.12, 0.10))
    g.ax_heatmap.tick_params(axis="y", labelsize=5)
    g.ax_heatmap.tick_params(axis="x", labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.figure)


def gini(x: np.ndarray) -> float:
    """Gini concentration of |effect| across cell types (0 = uniform/broad, 1 = one cell type)."""
    a = np.sort(np.abs(x))
    n = len(a)
    s = a.sum()
    if s == 0:
        return 0.0
    return (2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * s)


def select_variants(logfc, how, top_n, min_effect):
    """Rows for the clustermap.

    variance: top-N by across-cell-type variance (broad + specific mixed).
    specificity: variants concentrated in few cell types (top-N by Gini of |logfc|),
      gated to a real effect (max |logfc| >= min_effect) so they are visible and crisp.
    """
    if how == "variance":
        return logfc.var(axis=1).sort_values(ascending=False).head(top_n).index
    strong = logfc[logfc.abs().max(axis=1) >= min_effect]
    g = strong.apply(lambda r: gini(r.values), axis=1)
    return g.sort_values(ascending=False).head(top_n).index


def trait_clustermaps(cfg, key, top_n, cell_types, md, how, min_effect):
    td = io.sandbox_path(cfg, f"results/{key}")
    logfc = pd.read_csv(td / f"{PREFIX}.csv", index_col="variant_id")[cell_types]
    top = select_variants(logfc, how, top_n, min_effect)
    sub = logfc.loc[top]
    out = td / "vep_report_out"

    sub.to_csv(out / f"{PREFIX}.clustermap_variants.tsv", sep="\t")  # which variants are shown
    save_clustermap(sub, out / f"{PREFIX}.clustermap_by_celltype.pdf",
                    col_colors=col_color_bars(cfg, cell_types))
    for level in ["dev_stage", "lineage", "grouping"]:
        agg = sub.T.groupby(md[level]).mean().T  # variant x group (mean logfc)
        save_clustermap(agg, out / f"{PREFIX}.clustermap_by_{level}.pdf", col_cluster=True)
    log.info("%s: clustermaps for top %d %s variants", key, len(top), how)
    return logfc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--select", choices=["specificity", "variance"], default="specificity")
    ap.add_argument("--min-effect", type=float, default=0.3,
                    help="specificity: require max |logfc| >= this so rows are visible")
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    cell_types = metadata.cell_types(cfg)

    pooled = {}
    traits = [t for t in cfg["traits"] if not args.trait or t["key"] == args.trait]
    for trait in traits:
        lf = trait_clustermaps(cfg, trait["key"], args.top_n, cell_types, md,
                               args.select, args.min_effect)
        pooled[trait["key"]] = lf

    if len(pooled) > 1:
        allv = pd.concat(pooled.values())
        allv = allv[~allv.index.duplicated()]
        top = select_variants(allv, args.select, args.top_n, args.min_effect)
        cm_dir = io.sandbox_path(cfg, "results/clustermaps")
        save_clustermap(allv.loc[top], cm_dir / "pooled.clustermap_by_celltype.pdf",
                        col_colors=col_color_bars(cfg, cell_types))
        allv.loc[top].to_csv(cm_dir / "pooled.clustermap_variants.tsv", sep="\t")
        log.info("pooled clustermap for top %d %s variants across traits", len(top), args.select)

    io.write_provenance(cfg, "3_report_7_variant_clustermaps", {
        "traits": [t["key"] for t in traits], "top_n": args.top_n})
    log.info("Done.")


if __name__ == "__main__":
    main()
