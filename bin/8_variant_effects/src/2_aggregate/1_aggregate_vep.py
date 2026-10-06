"""Stage 2: aggregate per-cell-type variant scores into per-trait matrices.

Reads the union scores (one variant_scores.tsv per cell type), pivots to
variant x cell-type matrices, then splits by trait. Per trait it writes the
logfc matrix (the reference `celltype_vep_logfc.csv` schema), the matching
p-value matrix, and a per-variant cell-type ranking tsv for named and top loci.

Scoring is done once on the deduplicated union of variants; per-trait tables are
a post-hoc split, never a re-score.

Usage:
    python src/2_aggregate/1_aggregate_vep.py --config config/config.yaml \
        [--scores-dir results/chrombpnet] [--dry-run]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)

def write_variant_ranking(pval_row: pd.Series, variant_id: str, out_path: Path) -> None:
    """Write cell types ranked by -log10(p), matching the reference tsv format.

    The single data column is named by the variant id; index is cell_type.
    """
    ranked = (-np.log10(pval_row.astype(float))).sort_values(ascending=False)
    ranked.name = variant_id
    ranked.index.name = ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_frame().to_csv(out_path, sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores-dir", default="results/chrombpnet",
                    help="Sandbox-relative dir of per-cell-type scores")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    cell_types = metadata.cell_types(cfg)
    scores_dir = io.sandbox_path(cfg, args.scores_dir)
    credible = io.load_credible_sets(cfg)
    sig = cfg["significance"]

    log.info("Aggregating %d cell types from %s", len(cell_types), scores_dir)
    if args.dry_run:
        for ct in cell_types:
            p = scores_dir / ct / f"{ct}.variant_scores.tsv"
            log.info("  %s %s", "ok" if p.exists() else "MISSING", p)
        return

    logfc, pval = io.build_score_matrices(scores_dir, cell_types)
    log.info("Union matrix: %d variants x %d cell types", *logfc.shape)

    named = set(cfg["report"]["named_loci"])
    top_n = cfg["report"]["top_n_variants"]

    for trait in cfg["traits"]:
        key = trait["key"]
        ids = io.trait_variants(credible, key)["id"].unique()
        present = [v for v in ids if v in logfc.index]
        missing = len(ids) - len(present)
        if missing:
            log.warning("%s: %d/%d variants absent from scores", key, missing, len(ids))

        out_dir = io.sandbox_path(cfg, f"results/{key}")
        out_dir.mkdir(parents=True, exist_ok=True)
        lf = logfc.loc[present]
        pv = pval.loc[present]
        lf.to_csv(out_dir / "celltype_vep_logfc.csv", index_label="variant_id")
        pv.to_csv(out_dir / "celltype_vep_pval.csv", index_label="variant_id")

        # Per-variant cell-type ranking for named loci and the top-N by min p-value.
        top_ids = pv.min(axis=1).sort_values().head(top_n).index
        for vid in named.intersection(present).union(top_ids):
            safe = vid.replace(":", "_")
            fname = f"{safe}.{trait['study']}.-log10(abs_logfc_x_jsd.pval).tsv"
            write_variant_ranking(pv.loc[vid], vid, out_dir / fname)

        log.info("%s: wrote %d variants x %d cell types to %s",
                 key, lf.shape[0], lf.shape[1], out_dir)

    io.write_provenance(cfg, "2_aggregate", {
        "scores_dir": args.scores_dir,
        "n_cell_types": len(cell_types),
        "significance": sig,
        "top_n_variants": top_n,
    })
    log.info("Done.")


if __name__ == "__main__":
    main()
