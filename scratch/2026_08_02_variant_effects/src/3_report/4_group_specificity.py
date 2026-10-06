"""Stage 3.4: call variants specific to each cell-type group, per trait.

Quick-and-dirty group specificity. A cell type "hits" a variant if its shuffled-
null p-value is <= the threshold (config `significance`). For a grouping column
(grouping / lineage / dev_stage) and each group g:

  n_sig_in  = # cell types in g that hit the variant
  n_sig_out = # cell types outside g that hit the variant
  frac_in   = n_sig_in  / (cell types in g)
  frac_out  = n_sig_out / (cell types outside g)
  spec_score = frac_in - frac_out                  (1 = only g hits, <=0 = not specific)
  exclusive  = n_sig_in >= 1 and n_sig_out == 0    (strict: hits only within g)

A variant is reported for group g if n_sig_in >= 1. Rows are ranked by exclusive,
then spec_score, then |mean logfc in g|. This is a screen, not a calibrated test.

Usage:
    python src/3_report/4_group_specificity.py --config config/config.yaml \
        [--by grouping] [--trait T2D]
"""

import argparse

import numpy as np
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)


def group_specific(logfc, pval, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """Return one row per (variant, group) with n_sig_in >= 1 and specificity stats."""
    hit = pval <= threshold
    rows = []
    for g, cts in groups.groupby(groups).groups.items():
        cts_in = [c for c in cts if c in hit.columns]
        cts_out = [c for c in hit.columns if c not in cts_in]
        if not cts_in:
            continue
        n_in = hit[cts_in].sum(axis=1)
        n_out = hit[cts_out].sum(axis=1) if cts_out else pd.Series(0, index=hit.index)
        keep = n_in >= 1
        if not keep.any():
            continue
        frac_in = n_in / len(cts_in)
        frac_out = (n_out / len(cts_out)) if cts_out else pd.Series(0.0, index=hit.index)
        sub = pd.DataFrame({
            "variant_id": logfc.index[keep], "group": g,
            "n_sig_in": n_in[keep].values, "n_sig_out": n_out[keep].values,
            "frac_in": frac_in[keep].round(3).values, "frac_out": frac_out[keep].round(3).values,
            "spec_score": (frac_in - frac_out)[keep].round(3).values,
            "mean_logfc_in": logfc.loc[keep, cts_in].mean(axis=1).round(4).values,
            "max_abs_logfc_in": logfc.loc[keep, cts_in].abs().max(axis=1).round(4).values,
            "exclusive": (n_out[keep] == 0).values,
        })
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["group", "exclusive", "spec_score", "max_abs_logfc_in"],
                              ascending=[True, False, False, False])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--by", default="grouping", choices=["grouping", "lineage", "dev_stage"])
    ap.add_argument("--trait")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    threshold = cfg["significance"]["pval_threshold"]
    traits = [t for t in cfg["traits"] if not args.trait or t["key"] == args.trait]

    for trait in traits:
        key = trait["key"]
        td = io.sandbox_path(cfg, f"results/{key}")
        logfc = pd.read_csv(td / "celltype_vep_logfc.csv", index_col="variant_id")
        pval = pd.read_csv(td / "celltype_vep_pval.csv", index_col="variant_id")
        groups = md.loc[[c for c in md.index if c in logfc.columns], args.by]
        res = group_specific(logfc, pval, groups, threshold)
        out_dir = td / "group_specific"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_dir / f"{args.by}.specific_variants.tsv", sep="\t", index=False)
        counts = res.groupby("group")["exclusive"].agg(["size", "sum"]).rename(
            columns={"size": "n_variants", "sum": "n_exclusive"})
        counts.to_csv(out_dir / f"{args.by}.group_counts.tsv", sep="\t")
        log.info("%s by %s: %d group-specific rows across %d groups",
                 key, args.by, len(res), res["group"].nunique() if not res.empty else 0)

    io.write_provenance(cfg, f"3_report_4_group_specificity_{args.by}", {
        "by": args.by, "traits": [t["key"] for t in traits], "pval_threshold": threshold})
    log.info("Done.")


if __name__ == "__main__":
    main()
