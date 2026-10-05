#!/usr/bin/env python
"""Step 2 - align our target genes to Decima's gene set + genomic folds.

For the zero-shot baseline (and, later, the training arms) we evaluate on genes in
Decima's held-out TEST fold so the zero-shot numbers are leakage-free. Decima ships
its gene annotation + Borzoi fold assignment in `decima-data`; we intersect that with
our pseudobulk target genes and record (gene, chrom, fold, decima_gene_id).

Emits:
  results/5_expression_models/gene_folds.tsv   (gene, chrom, fold, in_target)
  figures/step2/fold_gene_counts.png, split_chrom_map.png

⚠️ The decima loaders below are read from the Decima source; VERIFY names/signatures
   against the installed version (`python -c "import decima; help(decima)"`).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
PB_PATH = f"{DATA_DIR}/results/2_process_data/pseudobulk_expression.h5ad"
OUT_DIR = f"{DATA_DIR}/results/5_expression_models"
FIG_DIR = f"{OUT_DIR}/figures/step2"


def load_decima_gene_metadata() -> pd.DataFrame:
    """Return a DataFrame with at least [gene_symbol, chrom, fold].

    VERIFY: Decima stores gene metadata + Borzoi fold labels in its packaged data
    (HuggingFace `Genentech/decima-data`, loaded via the decima library). Depending
    on version this is exposed as e.g. `decima.data.load_gene_metadata()` or as a
    column on the anndata var of the packaged target. Replace the stub below with the
    real loader once confirmed on the cluster.
    """
    try:
        # Preferred: a helper that returns gene table with fold assignments.
        from decima.preprocess import load_gene_metadata  # type: ignore  # VERIFY

        meta = load_gene_metadata()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Could not load Decima gene metadata automatically "
            f"({e}). VERIFY the loader name against the installed decima version "
            "and point load_decima_gene_metadata() at it. Expected columns: "
            "gene_symbol/name, chrom, and a Borzoi 'fold' label."
        )
    # normalize column names
    meta = meta.rename(
        columns={"name": "gene_symbol", "gene_name": "gene_symbol", "seqnames": "chrom"}
    )
    return meta


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pb = ad.read_h5ad(PB_PATH)
    target_genes = set(map(str, pb.var_names))
    print(f"target genes: {len(target_genes)}")

    meta = load_decima_gene_metadata()
    assert {"gene_symbol", "chrom", "fold"}.issubset(meta.columns), meta.columns.tolist()
    meta["in_target"] = meta["gene_symbol"].astype(str).isin(target_genes)

    matched = int(meta["in_target"].sum())
    print(f"decima genes: {len(meta)}; matched to our target: {matched}")

    tab = meta[["gene_symbol", "chrom", "fold", "in_target"]].copy()
    tab.to_csv(f"{OUT_DIR}/gene_folds.tsv", sep="\t", index=False)
    print(f"wrote {OUT_DIR}/gene_folds.tsv")

    # figure: gene counts per fold (all decima genes vs our matched subset)
    fig, ax = plt.subplots(figsize=(7, 4))
    allc = meta["fold"].value_counts().sort_index()
    subc = meta[meta["in_target"]]["fold"].value_counts().reindex(allc.index, fill_value=0)
    x = np.arange(len(allc))
    ax.bar(x - 0.2, allc.values, width=0.4, label="all Decima genes")
    ax.bar(x + 0.2, subc.values, width=0.4, label="matched to our target")
    ax.set_xticks(x)
    ax.set_xticklabels(allc.index, rotation=45, ha="right")
    ax.set_ylabel("genes")
    ax.set_title("Genes per Decima fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fold_gene_counts.png", dpi=150)
    plt.close(fig)

    # figure: chromosome x fold map of the matched genes
    sub = meta[meta["in_target"]]
    ct = pd.crosstab(sub["chrom"], sub["fold"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_title("Matched genes: chromosome × fold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="genes")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/split_chrom_map.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
