#!/usr/bin/env python
"""Build the all-gene pseudobulk expression target for the seq->expression models.

Generalizes build_pseudobulk_tf_expression.py: instead of subsetting to TF genes
and taking the MEAN, this keeps ALL genes and takes the SUMMED counts per cell
type -- the target format Decima's Poisson+multinomial loss expects. Replicates
are pooled (one track per cell_type). Tracks n_cells per pseudobulk.

Grouping is by `cell_type` (== `rna_annotation`), the canonical 22-type set from
config/cell_type_metadata.tsv. Output is an AnnData with tracks (cell types) as
obs and genes as var, plus a CSV for quick inspection.

Usage:
    python build_pseudobulk_expression.py

See bin/5_expression_models/ for how this target is consumed.
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def resolve_counts_matrix(adata):
    """Return a raw-counts matrix (cells x genes), erroring if none is found.

    Decima's loss needs summed raw counts, NOT log-normalized values. Prefer an
    explicit counts layer; otherwise accept X only if it looks like integer
    counts. Fail loudly rather than silently summing normalized data.
    """
    # 1) explicit counts layer, by common names
    for key in ("counts", "raw_counts", "count"):
        if key in adata.layers:
            print(f"Using raw counts from adata.layers['{key}']")
            return adata.layers[key]

    # 2) adata.raw (often stores pre-normalization counts)
    if adata.raw is not None:
        X = adata.raw.X
        if _looks_like_counts(X):
            print("Using raw counts from adata.raw.X")
            return X

    # 3) fall back to X only if it is integer-valued
    if _looks_like_counts(adata.X):
        print("Using raw counts from adata.X (integer-valued)")
        return adata.X

    raise ValueError(
        "Could not find a raw-counts matrix. adata.X appears normalized and no "
        "'counts' layer / adata.raw was found. Point this script at the raw "
        "counts (Decima's Poisson+multinomial loss requires summed counts, not "
        "log-normalized values)."
    )


def _looks_like_counts(X, n_check=10000):
    """Heuristic: are the (sampled) nonzero values non-negative integers?"""
    Xs = X[:50] if X.shape[0] > 50 else X
    vals = Xs.data if sp.issparse(Xs) else np.asarray(Xs).ravel()
    if vals.size == 0:
        return False
    vals = vals[:n_check]
    if np.any(vals < 0):
        return False
    return np.allclose(vals, np.round(vals))


def main():
    # Paths (mirror build_pseudobulk_tf_expression.py)
    data_dir = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
    rna_path = os.path.join(data_dir, "results/2_process_data/rna_matrix.h5ad")
    out_h5ad = os.path.join(data_dir, "results/2_process_data/pseudobulk_expression.h5ad")
    out_csv = os.path.join(data_dir, "results/2_process_data/pseudobulk_expression.csv")

    print("Loading RNA anndata...")
    adata = ad.read_h5ad(rna_path)
    adata.obs_names_make_unique()
    print(f"Shape: {adata.shape}")

    counts = resolve_counts_matrix(adata)
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts.tocsr()

    # Canonical cell-type ordering: pull from config if importable, else sort.
    cell_types = _ordered_cell_types(adata)
    print(f"Cell types (tracks): {len(cell_types)}")

    genes = list(adata.var_names)
    pb = np.zeros((len(cell_types), len(genes)), dtype=np.float64)
    n_cells = np.zeros(len(cell_types), dtype=np.int64)

    ct_series = adata.obs["cell_type"].astype(str)
    for i, ct in enumerate(cell_types):
        mask = (ct_series == ct).values
        n_cells[i] = int(mask.sum())
        if n_cells[i] == 0:
            print(f"  WARNING: 0 cells for '{ct}'")
            continue
        # summed counts across cells in this pseudobulk
        pb[i] = np.asarray(counts[mask].sum(axis=0)).ravel()

    # Assemble AnnData: tracks (cell types) x genes
    pb_adata = ad.AnnData(
        X=pb,
        obs=pd.DataFrame({"cell_type": cell_types, "n_cells": n_cells}).set_index("cell_type"),
        var=pd.DataFrame(index=genes),
    )
    pb_adata.obs["total_counts"] = pb.sum(axis=1).astype(np.int64)

    pb_adata.write_h5ad(out_h5ad)
    pd.DataFrame(pb, index=cell_types, columns=genes).to_csv(out_csv)
    print(f"Saved: {pb_adata.shape} (tracks x genes) to {out_h5ad}")
    print(f"       CSV to {out_csv}")

    # Summary
    print("\nPer-track n_cells / total_counts:")
    for ct, nc, tc in zip(cell_types, n_cells, pb_adata.obs["total_counts"]):
        print(f"  {ct:24s} n_cells={nc:6d}  total_counts={tc:,}")


def _ordered_cell_types(adata):
    """Use config/loader.py ordering if available; otherwise sorted unique."""
    try:
        # config/loader.py exposes the canonical order; import is best-effort.
        from config.loader import load_cell_type_metadata  # type: ignore

        meta = load_cell_type_metadata()
        order = list(meta.sort_values("display_order")["cell_type"])
        present = set(adata.obs["cell_type"].dropna().unique())
        ordered = [ct for ct in order if ct in present]
        # append any present-but-unlisted types, sorted, so nothing is dropped
        ordered += sorted(present - set(ordered))
        return ordered
    except Exception as e:  # noqa: BLE001
        print(f"(config.loader unavailable: {e}; falling back to sorted unique)")
        return sorted(adata.obs["cell_type"].dropna().astype(str).unique())


if __name__ == "__main__":
    main()
