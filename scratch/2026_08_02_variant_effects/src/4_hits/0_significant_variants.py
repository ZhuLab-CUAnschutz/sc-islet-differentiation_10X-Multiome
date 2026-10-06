"""Stage 4.0: select the significant-variant set for the hits analysis.

A variant is significant if its shuffled-null p-value is <= the threshold in at
least ``min_celltypes`` cell types (config `significance`). Writes a
variant-scorer-format BED (subset of the combined bed) for SHAP + hit calling,
and a table of which cell types each variant is significant in.

Usage:
    python src/4_hits/0_significant_variants.py --config config/config.yaml \
        [--scores-dir results/chrombpnet]
"""

import argparse

import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores-dir", default="results/chrombpnet")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    sig = cfg["significance"]
    cell_types = metadata.cell_types(cfg)
    scores_dir = io.sandbox_path(cfg, args.scores_dir)

    (pval,) = io.build_score_matrices(scores_dir, cell_types, cols=(sig["pval_col"],))
    hits_mask = pval <= sig["pval_threshold"]
    n_sig_ct = hits_mask.sum(axis=1)
    sig_ids = n_sig_ct[n_sig_ct >= sig["min_celltypes"]].index
    log.info("%d/%d variants significant (p<=%s in >=%d cell types)",
             len(sig_ids), pval.shape[0], sig["pval_threshold"], sig["min_celltypes"])

    # Subset the combined bed (chr, start, end, allele1, allele2, variant_id) to sig variants.
    bed = pd.read_csv(io.sandbox_path(cfg, cfg["combined_bed"]), sep="\t", header=None)
    bed_sig = bed[bed[5].isin(set(sig_ids))]
    out_bed = io.sandbox_path(cfg, "inputs/significant_variants.bed")
    bed_sig.to_csv(out_bed, sep="\t", header=False, index=False)

    # Per-variant significant cell-type membership, for the Stage 4.3 summary.
    membership = pd.DataFrame({
        "variant_id": sig_ids,
        "n_sig_celltypes": n_sig_ct.loc[sig_ids].values,
        "sig_celltypes": [";".join(hits_mask.columns[hits_mask.loc[v]].tolist()) for v in sig_ids],
    })
    out_tsv = io.sandbox_path(cfg, "results/hits/significant_membership.tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(out_tsv, sep="\t", index=False)

    io.write_provenance(cfg, "4_hits_0_significant_variants", {
        "n_significant": int(len(sig_ids)), "significance": sig,
        "scores_dir": args.scores_dir})
    log.info("wrote %s (%d variants) and %s", out_bed, len(bed_sig), out_tsv)


if __name__ == "__main__":
    main()
