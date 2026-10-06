"""Stage 4.4: overlap the significant variants with the standing v1.1 genome-wide hits.

Separate from Stage 4.3 (which re-calls finemo on each variant's own SHAP). Here we ask
the complementary question: does the variant fall inside a hit that was already called
genome-wide with the v1.1 catalog (``motifs/v1.1/hits/{ct}/hits.bed``)? Those hits name
motifs as ``Average_NNN``; we map them to ``ST##:TF`` via the per-cell-type
``motif_data.tsv`` (Average_NNN -> ``pattern_i``) and the catalog metadata (pattern order
-> base ST## -> curator TF), matching Stage 4.3's rooting.

Writes ``results/hits/standing_hit_overlaps.tsv`` (variant_id, cell_type, chr, start, end,
motif, motif_name), one row per variant x cell type x overlapping standing hit.

Usage:
    python src/4_hits/4_standing_hit_overlaps.py --config config/config.yaml
"""

import argparse

import numpy as np
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)


def base_pattern_labels(cfg) -> pd.Series:
    """pattern index i -> 'ST##:TF' (base catalog motif order, curator TF preferred)."""
    md = pd.read_csv(io.repo_path(cfg, cfg["catalog_metadata"]), sep="\t")
    tf = io.load_catalog_tf(cfg)  # short_id -> curator TF
    md["base"] = md["short_id"].str.replace(r"_sub\d+$", "", regex=True)
    base = md.drop_duplicates("base").reset_index(drop=True)
    return pd.Series([io.catalog_label(b, tf) for b in base["base"]])


def motif_label_map(cfg, ct: str, pattern_labels: pd.Series) -> dict:
    """Average_NNN -> 'ST##:TF' for one cell type's standing hits."""
    p = io.repo_path(cfg, f"{cfg['catalog_dir']}/hits/{ct}/motif_data.tsv")
    md = pd.read_csv(p, sep="\t").drop_duplicates("motif_name")
    idx = md["motif_name_orig"].str.extract(r"pattern_(\d+)")[0].astype("Int64")
    out = {}
    for name, i in zip(md["motif_name"], idx):
        out[name] = pattern_labels[int(i)] if pd.notna(i) and int(i) < len(pattern_labels) else name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant-bed", default="inputs/significant_variants.bed")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    cell_types = metadata.cell_types(cfg)
    pattern_labels = base_pattern_labels(cfg)

    bed = pd.read_csv(io.sandbox_path(cfg, args.variant_bed), sep="\t", header=None)
    variants = pd.DataFrame({"variant_id": bed[5]})
    variants[["chr", "pos"]] = variants["variant_id"].str.split(":", expand=True)[[0, 1]]
    variants["c0"] = variants["pos"].astype(int) - 1  # 0-based variant base

    rows = []
    for ct in cell_types:
        hb = io.repo_path(cfg, f"{cfg['catalog_dir']}/hits/{ct}/hits.bed")
        if not hb.exists():
            log.warning("no standing hits.bed for %s", ct)
            continue
        hits = pd.read_csv(hb, sep="\t", header=None,
                           names=["chr", "start", "end", "motif_name", "score", "strand"])
        labels = motif_label_map(cfg, ct, pattern_labels)
        by_chr = {c: g for c, g in hits.groupby("chr")}
        for _, v in variants.iterrows():
            g = by_chr.get(v["chr"])
            if g is None:
                continue
            m = g[(g["start"].values <= v["c0"]) & (g["end"].values > v["c0"])]
            for _, h in m.iterrows():
                rows.append({"variant_id": v["variant_id"], "cell_type": ct,
                             "chr": h["chr"], "start": int(h["start"]), "end": int(h["end"]),
                             "motif": labels.get(h["motif_name"], h["motif_name"]),
                             "motif_name": h["motif_name"]})
        log.info("%s: %d variant-overlapping standing hits", ct, sum(r["cell_type"] == ct for r in rows))

    out = pd.DataFrame(rows, columns=["variant_id", "cell_type", "chr", "start", "end",
                                      "motif", "motif_name"])
    out.to_csv(io.sandbox_path(cfg, "results/hits/standing_hit_overlaps.tsv"), sep="\t", index=False)
    log.info("standing-hit overlaps: %d rows, %d variants, %d cell types",
             len(out), out["variant_id"].nunique() if len(out) else 0, out["cell_type"].nunique() if len(out) else 0)
    io.write_provenance(cfg, "4_hits_4_standing_hit_overlaps",
                        {"n_rows": int(len(out)), "catalog_version": cfg["catalog_version"]})


if __name__ == "__main__":
    main()
