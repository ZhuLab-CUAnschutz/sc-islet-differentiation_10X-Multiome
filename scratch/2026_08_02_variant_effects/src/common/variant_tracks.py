"""Shared per-variant ChromBPNet compute: prediction, attribution, seqlet annotation.

Used by the single-cell-type attribution plots (Stage 3.2) and the cell-type
contrast panels (Stage 3.5). Requires a torch/bpnetlite/tangermeme env (eugene_tools).
"""

import warnings

import numpy as np
import pandas as pd
import seqpro as sp
import torch
from bpnetlite.bpnet import BPNet, CountWrapper
from tangermeme.annotate import annotate_seqlets
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import read_meme
from tangermeme.predict import predict

from . import io

SEQ_LEN = 2114
TARGET_LEN = 1000
MUT_WINDOW = 500
PLOT_WINDOW = 300
SHAP_DTYPE = torch.float64


def check_coordinates(genome, variant_ids, min_match=0.90) -> float:
    """Guard against off-by-one: fraction of variants whose genome base at end-1
    equals allele1 must exceed min_match (SNP credible sets are ~95% ref-matching).
    Raises if it drops (e.g. a regression to 0-based/1-based centering). Returns the
    match fraction. `variant_ids` are 'chr:end:allele1:allele2'.
    """
    ok = tot = 0
    for vid in variant_ids:
        c, e, a1, a2 = vid.split(":")
        try:
            base = genome[c][int(e) - 1].seq.upper()
        except (KeyError, ValueError):
            continue
        tot += 1
        ok += base == a1
    frac = ok / tot if tot else 0.0
    if frac < min_match:
        raise ValueError(f"coordinate check failed: only {frac:.1%} of variants have "
                         f"genome[end-1]==allele1 (expected >={min_match:.0%}); likely an "
                         f"off-by-one or build/strand error in variant centering")
    return frac


def windows() -> dict:
    """Sequence and plotting window offsets used across the track plots."""
    sc, tc = SEQ_LEN // 2, TARGET_LEN // 2
    return {
        "seq_start": sc - MUT_WINDOW // 2, "seq_end": sc + MUT_WINDOW // 2,
        "plot_seq_start": sc - PLOT_WINDOW // 2, "plot_seq_end": sc + PLOT_WINDOW // 2,
        "plot_target_start": tc - PLOT_WINDOW // 2, "plot_target_end": tc + PLOT_WINDOW // 2,
    }


def log_softmax_profile(y_profile: torch.Tensor) -> torch.Tensor:
    """Log-softmax a predicted profile over its length dimension."""
    z = y_profile.shape
    y = torch.nn.functional.log_softmax(y_profile.reshape(y_profile.shape[0], -1), dim=-1)
    return y.reshape(*z)


def load_model(cfg: dict, cell_type: str):
    """Return (model, count_wrapper) on GPU for a cell type."""
    mp = io.model_path(cfg, cell_type)
    model = BPNet.from_chrombpnet(str(mp)).cuda().eval()
    return model, CountWrapper(model).cuda().eval()


def allele_sequences(genome, chrom: str, pos: int, allele1: str, allele2: str):
    """Reference and alternate 2114-bp sequences centered on the variant.

    Both allele1 and allele2 are forced into the center (matching variant-scorer),
    not the raw genome base: some credible-set variants have a genome base that
    differs from allele1 (e.g. strand-reported), and using the genome base as the
    reference would flip the predicted direction relative to the scored logfc.
    """
    half = SEQ_LEN // 2
    c0 = pos - 1  # variant_id position is the bed 1-based 'end'; pyfaidx is 0-based
    seq = genome[chrom][c0 - half:c0 + half].seq.upper()
    # Both alleles are forced at end-1 (matching variant-scorer, which does the same
    # regardless of the genome base); per-variant mismatches are strand/annotation
    # cases, so warn rather than fail. Coordinate correctness (no off-by-one) is
    # guaranteed globally by check_coordinates() below, not per variant.
    if seq[half] != allele1:
        warnings.warn(f"{chrom}:{pos}:{allele1}:{allele2}: genome base {seq[half]} != allele1")
    ref = seq[:half] + allele1 + seq[half + 1:]
    alt = seq[:half] + allele2 + seq[half + 1:]
    return ref, alt


def predict_profile_counts(model, seq: str):
    """Return (forward onehot, profile prob, exp count), averaged over both strands.

    variant-scorer scores both strands by default; averaging forward and reverse
    complement here makes the plotted counts match the scored logfc. The returned
    onehot is the forward strand (used for attributions).
    """
    oh_f = np.expand_dims(sp.ohe(seq, alphabet=sp.DNA), 0).transpose(0, 2, 1)
    oh_r = np.expand_dims(sp.ohe(sp.reverse_complement(seq, sp.DNA), alphabet=sp.DNA), 0).transpose(0, 2, 1)
    pf, cf = predict(model, torch.tensor(oh_f).float().cuda())
    pr, cr = predict(model, torch.tensor(oh_r).float().cuda())
    count = 0.5 * (float(np.squeeze(np.exp(cf.cpu().detach().numpy())))
                   + float(np.squeeze(np.exp(cr.cpu().detach().numpy()))))
    prof_f = np.squeeze(torch.exp(log_softmax_profile(pf)).cpu().detach().numpy())
    prof_r = np.squeeze(torch.exp(log_softmax_profile(pr)).cpu().detach().numpy())[::-1]
    return oh_f, 0.5 * (prof_f + prof_r), count


def attributions(count_wrapper, onehot):
    """DeepLiftShap attributions for the count head of one sequence."""
    return deep_lift_shap(count_wrapper.type(SHAP_DTYPE),
                          torch.tensor(onehot).type(SHAP_DTYPE).cuda(),
                          n_shuffles=20, batch_size=8, verbose=False, warning_threshold=1e-4)


def annotate(onehot, attr, meme_path, win, motif_names, tf_map):
    """Recursive seqlets over the mutation window; the seqlet matcher matches each
    against the v1.1 catalog (catalog.meme names are ST## ids), and we label with
    'ST##:curator_tf' from the catalog so every annotation is catalog-rooted."""
    from tangermeme.seqlet import recursive_seqlets
    seqlets = recursive_seqlets(attr[:, :, win["seq_start"]:win["seq_end"]].sum(dim=1),
                                threshold=0.05, additional_flanks=5)
    idxs = annotate_seqlets(torch.tensor(onehot)[:, :, win["seq_start"]:win["seq_end"]],
                            seqlets, meme_path)[0][:, 0]
    seqlets["start"] += win["seq_start"]
    seqlets["end"] += win["seq_start"]
    st_ids = np.atleast_1d(motif_names[np.atleast_1d(idxs)])  # catalog ST## ids
    seqlets["example_idx"] = [io.catalog_label(s, tf_map) for s in st_ids]
    return seqlets


def load_motif_tf(cfg: dict):
    """Catalog ST## motif names (from catalog.meme) and the ST## -> TF map (curator_tf)."""
    meme = str(io.repo_path(cfg, cfg["catalog_meme"]))
    names = np.array(list(read_meme(meme).keys()))  # these are the ST## ids
    return meme, names, io.load_catalog_tf(cfg)
