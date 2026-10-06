"""Stage 4.3: summarize which variants disrupt called motif hits across cell types.

Collects the per-cell-type `variant_hit_calls.tsv` from the finemo hit caller,
maps each hit back to its variant and allele, annotates the motif with its TF
from the v1.1 catalog, and writes a long disruption table plus cross-tabs
(variant x cell type, variant x motif) and a summary figure.

The hit caller indexes variants by ``peak_id % n_variants`` (peak_id >= n means
the alternate allele), so the significant-variant BED order defines the mapping.

Usage:
    python src/4_hits/3_summarize_hits.py --config config/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)


def load_pattern_map(cfg) -> pd.DataFrame:
    """Map finemo modisco pattern index -> base catalog ST## and TF.

    finemo emits names like ``pos_patterns.pattern_{i}``. The clustered modisco h5
    has one pattern per base motif; the catalog metadata expands a few into sub-
    splits (ST05, ST36), so collapsing sub-splits gives one base row per pattern,
    in order. pattern_{i} therefore maps to the i-th base row.
    """
    md = pd.read_csv(io.repo_path(cfg, cfg["catalog_metadata"]), sep="\t")
    md["tf"] = md["short_id"].map(io.load_catalog_tf(cfg))  # curator_tf from v1.1 catalog
    md["base_id"] = md["short_id"].str.replace(r"_sub\d+$", "", regex=True)
    base = md.drop_duplicates("base_id").reset_index(drop=True)
    return pd.DataFrame({"st_id": base["base_id"], "tf": base["tf"]})


def annotate_motifs(hits: pd.DataFrame, pattern_map: pd.DataFrame) -> pd.DataFrame:
    """Add st_id and tf columns from the modisco pattern index; keep raw motif_name."""
    idx = hits["motif_name"].str.extract(r"pattern_(\d+)", expand=False).astype("Int64")
    hits = hits.copy()
    hits["st_id"] = idx.map(pattern_map["st_id"]).fillna(hits["motif_name"])
    hits["tf"] = idx.map(pattern_map["tf"]).fillna(hits["motif_name"])
    hits["motif"] = hits["st_id"].astype(str) + ":" + hits["tf"].astype(str)  # ST##:TF
    return hits


def collect_hits(cfg, cell_types, variant_ids) -> pd.DataFrame:
    """Concatenate per-cell-type variant hit calls, mapping peak_id -> variant."""
    n = len(variant_ids)
    rows = []
    for ct in cell_types:
        path = io.sandbox_path(cfg, f"results/hits/{ct}/variant_hit_calls.tsv")
        if not path.exists():
            log.warning("no hit calls for %s (%s)", ct, path)
            continue
        df = pd.read_csv(path, sep="\t")
        if df.empty:
            continue
        df["cell_type"] = ct
        df["variant_id"] = [variant_ids[int(pid) % n] for pid in df["peak_id"]]
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No variant_hit_calls.tsv found; run Stages 4.1-4.2 first")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant-bed", default="inputs/significant_variants.bed",
                    help="Sandbox-relative bed used for hit calling (peak_id order)")
    args = ap.parse_args()

    configure_matplotlib()
    sns.set_style("white")
    cfg = io.load_config(args.config)
    cell_types = metadata.cell_types(cfg)
    md = metadata.load_cell_type_metadata(cfg)

    sig_bed = pd.read_csv(io.sandbox_path(cfg, args.variant_bed), sep="\t", header=None)
    variant_ids = sig_bed[5].tolist()

    hits = collect_hits(cfg, cell_types, variant_ids)
    hits = annotate_motifs(hits, load_pattern_map(cfg))
    hits.to_csv(io.sandbox_path(cfg, "results/hits/hits_disruption.tsv"), sep="\t", index=False)
    log.info("collected %d variant-hit rows across %d cell types",
             len(hits), hits["cell_type"].nunique())

    # Cross-tabs: variant x cell type (n hits) and variant x motif (n cell types).
    vc = pd.crosstab(hits["variant_id"], hits["cell_type"])
    vc.to_csv(io.sandbox_path(cfg, "results/hits/variant_by_celltype.tsv"), sep="\t")
    vm = pd.crosstab(hits["variant_id"], hits["motif"])  # columns = ST##:TF (catalog-rooted)
    vm.to_csv(io.sandbox_path(cfg, "results/hits/variant_by_motif.tsv"), sep="\t")

    # Summary figure: disrupted-variant count per cell type, stage-ordered.
    per_ct = hits.groupby("cell_type")["variant_id"].nunique().reindex(
        [c for c in md.index if c in set(hits["cell_type"])]).fillna(0)
    _, stage_colors = metadata.category(cfg, "dev_stage")
    colors = md.loc[per_ct.index, "dev_stage"].map(stage_colors)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(per_ct.index, per_ct.values, color=colors, edgecolor="black")
    ax.set_ylabel("Variants overlapping a called hit")
    ax.set_xlabel("Cell type")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Credible-set variants disrupting called motif hits, per cell type")
    fig.tight_layout()
    savefig(fig, io.sandbox_path(cfg, "figures/hits_disruption_per_celltype.pdf"))

    io.write_provenance(cfg, "4_hits_3_summarize_hits", {
        "n_variant_hit_rows": int(len(hits)),
        "n_variants_disrupting": int(hits["variant_id"].nunique()),
        "catalog_version": cfg["catalog_version"]})
    log.info("Done.")


if __name__ == "__main__":
    main()
