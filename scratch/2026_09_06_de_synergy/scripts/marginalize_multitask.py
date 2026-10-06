#!/usr/bin/env python
"""In-silico motif marginalization through a CREsted multi-task (peak regression) model.

Drops a motif into the centre of k=2-shuffled peak background sequences,
predicts the 22-output head before and after, and returns per-motif per-sequence
deltas on the target cell type's logit.

Outputs (per cell type, written to ``--output_dir/{celltype}/``):
  - ``marginalization_data.counts.npz`` with keys
      ``c_before`` (n_motifs, n_seqs), ``c_after`` (n_motifs, n_seqs), ``names``
    (matches the ChromBPNet schema consumed by
    ``bin/3_single_task_models/4_marginalization_analysis.ipynb``).

Per-CT marginalization sequences are sampled from the same per-CT narrowPeak
file used for ChromBPNet so the two pipelines are comparable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

NARROWPEAKS_SCHEMA = [
    "chrom", "start", "end", "name", "score", "strand",
    "signal_value", "p_value", "q_value", "peak",
]

ALPHABET = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(ALPHABET)}


def k_shuffle(seq: str, k: int, rng) -> str:
    """k-let preserving shuffle (k=2 = dinucleotide)."""
    seq = seq.upper()
    if k <= 1:
        chars = list(seq)
        rng.shuffle(chars)
        return "".join(chars)
    # Build de Bruijn graph over (k-1)-mers
    edges = {}
    for i in range(len(seq) - k + 1):
        u = seq[i:i + k - 1]
        v = seq[i + 1:i + k]
        edges.setdefault(u, []).append(v)
    for u in edges:
        rng.shuffle(edges[u])
    # Eulerian walk starting from first (k-1)-mer
    start = seq[:k - 1]
    walk = [start]
    stack = [start]
    while stack:
        u = stack[-1]
        if edges.get(u):
            stack.append(edges[u].pop())
        else:
            walk.append(stack.pop())
    walk.reverse()
    # Reassemble: walk[0] + last char of each subsequent (k-1)-mer
    out = walk[0]
    for u in walk[1:]:
        out += u[-1]
    return out[:len(seq)]


def one_hot(seq: str) -> "np.ndarray":
    import numpy as np
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_IDX.get(b, None)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def extract_background_sequences(genome_fasta, peaks_path, n_seqs, flank, seed):
    import numpy as np
    import pandas as pd
    import pyfaidx

    genome = pyfaidx.Fasta(genome_fasta)
    peaks = pd.read_csv(peaks_path, header=None, sep="\t", names=NARROWPEAKS_SCHEMA)
    peaks = peaks.sort_values("q_value", ascending=False).drop_duplicates(
        subset=["chrom", "start", "end"]
    )
    peaks["mid"] = (peaks["start"] + peaks["end"]) // 2
    peaks["start_w"] = peaks["mid"] - flank
    peaks["end_w"] = peaks["mid"] + flank

    rng = np.random.default_rng(seed)
    sampled = peaks.sample(n=min(n_seqs * 3, len(peaks)), random_state=seed)

    seqs = []
    for _, row in sampled.iterrows():
        if row["start_w"] < 0:
            continue
        try:
            seq = str(genome[row["chrom"]][int(row["start_w"]):int(row["end_w"])].seq).upper()
        except (KeyError, ValueError):
            continue
        if len(seq) != 2 * flank:
            continue
        n_n = sum(1 for b in seq if b not in "ACGT")
        if n_n / len(seq) > 0.05:
            continue
        # k-shuffle to get a flat background
        seqs.append(k_shuffle(seq, k=2, rng=rng))
        if len(seqs) >= n_seqs:
            break
    if len(seqs) < n_seqs:
        logger.warning("Only %d backgrounds met QC (wanted %d)", len(seqs), n_seqs)
    return seqs


def load_motifs(meme_path):
    """Parse a MEME file → list[(name, ppm[L,4])]. Returns alphabet order ACGT."""
    motifs = []
    with open(meme_path) as fh:
        line = fh.readline()
        cur_name = None
        cur_rows = []
        in_matrix = False
        while line:
            if line.startswith("MOTIF"):
                if cur_name is not None and cur_rows:
                    motifs.append((cur_name, cur_rows))
                cur_name = line.strip().split()[1]
                cur_rows = []
                in_matrix = False
            elif line.startswith("letter-probability matrix"):
                in_matrix = True
            elif in_matrix and line.strip():
                parts = line.strip().split()
                try:
                    row = [float(x) for x in parts[:4]]
                    if len(row) == 4:
                        cur_rows.append(row)
                    else:
                        in_matrix = False
                except ValueError:
                    in_matrix = False
            elif in_matrix and not line.strip():
                in_matrix = False
            line = fh.readline()
        if cur_name is not None and cur_rows:
            motifs.append((cur_name, cur_rows))
    import numpy as np
    return [(name, np.asarray(rows, dtype=np.float32)) for name, rows in motifs]


def embed_motif(oh_backgrounds, ppm, rng):
    """Sample a motif instance and overwrite centre of each background sequence."""
    import numpy as np
    n_seqs, L, _ = oh_backgrounds.shape
    mw = ppm.shape[0]
    mid = L // 2
    start = mid - mw // 2
    out = oh_backgrounds.copy()
    for i in range(n_seqs):
        sampled = np.zeros((mw, 4), dtype=np.float32)
        for j in range(mw):
            base = rng.choice(4, p=ppm[j] / ppm[j].sum())
            sampled[j, base] = 1.0
        out[i, start:start + mw, :] = sampled
    return out


def main(args):
    os.environ["KERAS_BACKEND"] = args.backend
    import numpy as np
    import keras

    out_dir = Path(args.output_dir) / args.celltype
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model)
    model = keras.models.load_model(args.model, compile=False)
    in_len = model.input_shape[1]
    n_outputs = model.output_shape[-1]
    flank = in_len // 2
    logger.info("Model input length=%d, output dim=%d", in_len, n_outputs)

    # Resolve target CT index from the obs_names of the prediction AnnData
    import anndata as ad
    adata = ad.read_h5ad(args.adata, backed="r")
    obs_names = list(adata.obs_names)
    if args.celltype not in obs_names:
        logger.error("Cell type %s not in AnnData obs_names: %s", args.celltype, obs_names)
        sys.exit(1)
    target_idx = obs_names.index(args.celltype)
    logger.info("Cell type %s -> target_idx=%d", args.celltype, target_idx)

    # Backgrounds
    logger.info("Sampling %d backgrounds from %s", args.num_seqs, args.peaks)
    bg_seqs = extract_background_sequences(
        args.genome_fasta, args.peaks, args.num_seqs, flank=flank, seed=args.seed,
    )
    oh_bg = np.stack([one_hot(s) for s in bg_seqs])
    logger.info("Background tensor: %s", oh_bg.shape)

    # Motifs
    motifs = load_motifs(args.motif_file)
    names = np.array([m[0] for m in motifs], dtype="U")
    logger.info("Loaded %d motifs from %s", len(motifs), args.motif_file)

    rng = np.random.default_rng(args.seed)

    # Predict backgrounds once
    logger.info("Predicting backgrounds")
    pred_bg = model.predict(oh_bg, batch_size=args.batch_size, verbose=0)  # (N, 22)
    c_before = np.tile(pred_bg[:, target_idx][np.newaxis, :], (len(motifs), 1))

    c_after = np.zeros_like(c_before)
    for mi, (mname, ppm) in enumerate(motifs):
        ppm = ppm + 1e-6
        ppm = ppm / ppm.sum(axis=1, keepdims=True)
        embedded = embed_motif(oh_bg, ppm, rng)
        preds = model.predict(embedded, batch_size=args.batch_size, verbose=0)
        c_after[mi] = preds[:, target_idx]
        if mi == 0 or (mi + 1) % 5 == 0:
            delta = (c_after[mi] - c_before[mi]).mean()
            logger.info("[%2d/%d] %-30s  mean(c_after-c_before)=%+8.4f",
                        mi + 1, len(motifs), mname, delta)

    out_npz = out_dir / "marginalization_data.counts.npz"
    np.savez_compressed(out_npz,
                        c_before=c_before, c_after=c_after, names=names,
                        target_idx=target_idx)
    logger.info("Wrote %s", out_npz)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Trained CREsted multi-task model (.keras)")
    p.add_argument("--motif_file", required=True, help="MEME file with the unified motif catalog")
    p.add_argument("--genome_fasta", required=True)
    p.add_argument("--peaks", required=True, help="Per-CT narrowPeak file used as background source")
    p.add_argument("--adata", required=True, help="AnnData with obs_names = cell type order (to map CT -> output index)")
    p.add_argument("--celltype", required=True)
    p.add_argument("--output_dir", required=True, help="Root marginalization output dir; per-CT subdir auto-created")
    p.add_argument("--num_seqs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backend", default="tensorflow")
    main(p.parse_args())
