"""Data for the seq->expression training arms (Steps 4-5).

Each example = one gene:
  x : (5, 524288) one-hot DNA + gene-mask channel, built with Decima's windowing
      (gene TSS at >=163,840 bp from the upstream edge; 5th channel marks the gene body)
  y : (22,) summed pseudobulk counts across our cell types (Decima's Poisson+multinomial
      loss consumes counts, not log-CPM)

Genes are split by Decima's Borzoi folds (results/5_expression_models/gene_folds.tsv);
targets come from results/2_process_data/pseudobulk_expression.h5ad.

⚠️ The sequence/gene-mask extraction is Decima-API; VERIFY against the installed version.
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
PB_PATH = f"{DATA_DIR}/results/2_process_data/pseudobulk_expression.h5ad"
GENE_FOLDS = f"{DATA_DIR}/results/5_expression_models/gene_folds.tsv"
GENOME_FA = "/carter/users/aklie/data/ref/genomes/hg38/hg38.fa"

CONTEXT = 524_288  # Decima context
UPSTREAM = 163_840  # TSS offset from the upstream edge

TRAIN_FOLDS = {"train", "fold0", "fold1", "fold2", 0, 1, 2}  # VERIFY fold labels
VAL_FOLDS = {"val", "valid", "validation"}
TEST_FOLDS = {"test", "fold3", 3}


def _build_gene_window(gene_symbol, gene_meta):
    """Return x: torch.FloatTensor (5, CONTEXT) for one gene.

    VERIFY: Decima exposes the exact windowing + gene-mask construction it trained
    with. Prefer calling that (so our inputs match Decima's) rather than re-deriving.
    Expected: one-hot(A,C,G,T) over the 524 kb window + a 5th binary channel = 1 over
    [gene_mask_start, gene_mask_end]. Replace the stub with the real call.
    """
    from decima.preprocess import extract_gene_input  # type: ignore  # VERIFY

    x = extract_gene_input(
        gene_symbol, genome=GENOME_FA, context=CONTEXT, upstream=UPSTREAM, meta=gene_meta
    )
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class GeneExpressionDataset(Dataset):
    def __init__(self, genes, targets_df, gene_meta):
        self.genes = list(genes)
        self.targets = targets_df  # genes x 22 (summed counts), rows indexed by gene
        self.gene_meta = gene_meta

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, i):
        g = self.genes[i]
        x = _build_gene_window(g, self.gene_meta)
        y = torch.as_tensor(self.targets.loc[g].values, dtype=torch.float32)
        return x, y


def load_targets():
    """(genes x 22) summed-counts matrix + the ordered track list."""
    pb = ad.read_h5ad(PB_PATH)  # tracks x genes
    df = pd.DataFrame(np.asarray(pb.X), index=pb.obs_names, columns=pb.var_names).T
    return df, list(pb.obs_names)


def load_gene_meta():
    """Decima gene metadata (chrom, TSS, strand, fold, ...) needed for windowing.

    VERIFY: same loader as preprocess/build_windows_and_folds.py.
    """
    from decima.preprocess import load_gene_metadata  # type: ignore  # VERIFY

    return load_gene_metadata()


def make_loaders(batch_size=4, num_workers=4):
    targets, tracks = load_targets()
    gene_meta = load_gene_meta()
    folds = pd.read_csv(GENE_FOLDS, sep="\t")
    folds = folds[folds["in_target"]]

    def genes_for(fs):
        gs = folds.loc[folds["fold"].isin(fs), "gene_symbol"].astype(str)
        return [g for g in gs if g in targets.index]

    tr, va, te = genes_for(TRAIN_FOLDS), genes_for(VAL_FOLDS), genes_for(TEST_FOLDS)
    print(f"genes  train={len(tr)}  val={len(va)}  test={len(te)}")

    def dl(genes, shuffle):
        return DataLoader(
            GeneExpressionDataset(genes, targets, gene_meta),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle,
        )

    return {
        "train": dl(tr, True), "val": dl(va, False), "test": dl(te, False),
        "tracks": tracks, "test_genes": te, "targets": targets,
    }
