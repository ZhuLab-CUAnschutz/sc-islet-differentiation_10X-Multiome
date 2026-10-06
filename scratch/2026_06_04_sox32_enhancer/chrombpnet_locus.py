"""
chrombpnet_locus.py
===================

Reusable utilities for predicting an arbitrary locus / construct sequence across
the per-cell-type ChromBPNet models trained for the sc-islet-differentiation
10X-Multiome project, and for computing DeepLIFT/SHAP contribution scores.

Designed to be generalizable: point it at any FASTA (a genomic locus, an MPRA
construct, an enhancer + reporter, ...) and any subset of the 22 cell-type
models, and it will return predicted accessibility (counts + profile) and
per-base count contributions, strand-averaged where appropriate.

Environment
-----------
Run under the `eugene_tools` conda env (has torch+cu121, bpnetlite [editable],
tangermeme, seqpro). GPU strongly recommended for contributions.

    /cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python

Model layout (per cell type)
----------------------------
    <MODELS_ROOT>/<celltype>/models/chrombpnet_nobias.h5

Key API
-------
    metadata_df()                      -> cell-type metadata (order, color, grouping)
    read_fasta(path)                   -> (name, seq_upper)
    ohe(seq)                           -> (1, 4, L) one-hot
    load_model(celltype | h5_path)     -> bpnetlite BPNet on cuda/cpu
    predict(model, seq)                -> dict(counts, profile, ...)  strand-averaged
    count_contributions(model, seq)    -> (4, L) DeepLIFT/SHAP count attributions
    center_sequence(seq, total_len)    -> seq padded with N to total_len, centered
"""

import os
import glob
import numpy as np
import pandas as pd
import torch

import seqpro as sp
from bpnetlite.bpnet import BPNet, CountWrapper, ControlWrapper
from tangermeme.predict import predict as _tm_predict
from tangermeme.deep_lift_shap import deep_lift_shap

# all paths / lengths live in config.py (single source of truth)
from config import (MODELS_ROOT, CONFIG_TSV, MODEL_FILE, INPUT_LEN, OUTPUT_LEN,
                    output_window)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def metadata_df():
    """Cell-type metadata: index=cell_type, cols=grouping, color, display_order."""
    df = pd.read_csv(CONFIG_TSV, sep="\t")
    df = df.sort_values("display_order").set_index("cell_type")
    return df


def trajectory_order():
    """Cell types in developmental (display) order."""
    return metadata_df().index.tolist()


def color_map():
    """dict cell_type -> hex color."""
    return metadata_df()["color"].to_dict()


def model_path(celltype):
    """Resolve a cell-type name to its no-bias ChromBPNet .h5 path."""
    p = os.path.join(MODELS_ROOT, celltype, "models", MODEL_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No model for cell type {celltype!r}: {p}")
    return p


def peak_logcounts(celltype):
    """Predicted log-counts for this cell type's called peaks, from the ChromBPNet
    eval (`evaluation/chrombpnet_predictions.h5`). The distribution to rank a
    locus against — a per-model accessibility reference."""
    import h5py
    p = os.path.join(MODELS_ROOT, celltype, "evaluation", "chrombpnet_predictions.h5")
    with h5py.File(p, "r") as f:
        return f["predictions/logcounts"][:]


def available_celltypes():
    """All cell types that have a no-bias model (excludes *_biased / *_prediction)."""
    hits = glob.glob(os.path.join(MODELS_ROOT, "*", "models", MODEL_FILE))
    cts = sorted({os.path.relpath(h, MODELS_ROOT).split(os.sep)[0] for h in hits})
    # keep only the "clean" cell-type dirs that exist in the metadata
    meta = set(metadata_df().index)
    return [c for c in cts if c in meta]


# ----------------------------------------------------------------------------
# Sequence helpers
# ----------------------------------------------------------------------------
def read_fasta(path):
    """Return (name, sequence_upper) for a single-record FASTA."""
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
            else:
                chunks.append(line)
    return name, "".join(chunks).upper()


def ohe(seq):
    """One-hot encode a DNA string -> (1, 4, L) float32. N -> all-zero column.
    Accepts a str or a seqpro byte/char array (e.g. from `revcomp`)."""
    if isinstance(seq, str):
        seq = seq.upper()
    arr = sp.ohe(seq, alphabet=sp.DNA)                    # (L, 4)
    return np.expand_dims(arr, 0).transpose(0, 2, 1).astype("float32")  # (1, 4, L)


def revcomp(seq):
    if isinstance(seq, str):
        seq = seq.upper()
    return sp.reverse_complement(seq, sp.DNA)


def _composition_matched_pad(seq, n, seed):
    """Generate `n` bp of random DNA matched to the mononucleotide composition
    of `seq` (deterministic given `seed`). Used to pad without introducing
    all-zero (N) columns, which deep_lift_shap rejects, and without creating
    strong spurious motifs (unlike a fixed flank)."""
    if n <= 0:
        return ""
    bases = np.array(list("ACGT"))
    s = "".join(b for b in seq.upper() if b in "ACGT")
    freq = np.array([s.count(b) for b in "ACGT"], dtype=float)
    freq = freq / freq.sum() if freq.sum() else np.ones(4) / 4
    rng = np.random.RandomState(seed)
    return "".join(rng.choice(bases, size=n, p=freq))


def dinuc_shuffle(seq, seed=0):
    """Dinucleotide shuffle of a DNA string (preserves dinucleotide composition,
    destroys motifs). Returns a string of the same length. Uses tangermeme's
    Altschul-Erikson implementation."""
    import torch
    from tangermeme.ersatz import dinucleotide_shuffle as _ds
    X = torch.tensor(ohe(seq))                     # (1,4,L)
    S = _ds(X, n=1, random_state=seed)             # (1,1,4,L)
    idx = S[0, 0].numpy().argmax(0)
    return "".join("ACGT"[i] for i in idx)


def place_at_offset(core, offset, total_len=INPUT_LEN, pad="dinuc", seed=0):
    """Place `core` starting at `offset` within a window of `total_len`; fill the
    flanking ('non-centered') positions.

    pad="dinuc"  -> flanks are a DINUCLEOTIDE SHUFFLE of `core` (matched
                    dinucleotide content, no motifs; left/right use distinct seeds)
    pad="shuffle"-> mononucleotide composition-matched random
    pad=<char>   -> that character
    Returns (sequence, (core_start, core_end)). offset=(total_len-len(core))//2
    is fully centered.
    """
    L = len(core)
    assert 0 <= offset <= total_len - L, f"offset {offset} out of range for core {L}/{total_len}"
    left, right = offset, total_len - L - offset
    if pad == "dinuc":
        lpad = dinuc_shuffle(core, seed)[:left]
        rpad = dinuc_shuffle(core, seed + 1000)[:right] if right else ""
    elif pad == "shuffle":
        lpad = _composition_matched_pad(core, left, seed)
        rpad = _composition_matched_pad(core, right, seed + 1000)
    else:
        lpad, rpad = pad * left, pad * right
    return lpad + core + rpad, (left, left + L)


def keep_center_shuffle_flanks(seq, keep_width, pad="dinuc", seed=0):
    """Keep the central `keep_width` bp of `seq` intact and replace the flanking
    ('non-centered') positions. The actual sox32 enhancer is the central ~1100 bp;
    the flanks carry a minimal promoter + reporter that drive broad accessibility.
    Shrinking `keep_width` progressively shuffles that promoter/flank away to
    isolate the enhancer's own signal.

    pad="dinuc"  -> each flank is a dinucleotide shuffle of THAT flank's sequence
                    (preserves its dinucleotide content, destroys motifs)
    pad="shuffle"-> mononucleotide composition-matched random
    pad=<char>   -> that character
    Returns (sequence, (center_start, center_end)).
    """
    L = len(seq)
    keep_width = min(keep_width, L)
    cs = (L - keep_width) // 2
    ce = cs + keep_width
    lf, rf = seq[:cs], seq[ce:]
    if pad == "dinuc":
        lpad = dinuc_shuffle(lf, seed) if lf else ""
        rpad = dinuc_shuffle(rf, seed + 1000) if rf else ""
    elif pad == "shuffle":
        lpad = _composition_matched_pad(seq, len(lf), seed)
        rpad = _composition_matched_pad(seq, len(rf), seed + 1000)
    else:
        lpad, rpad = pad * len(lf), pad * len(rf)
    return lpad + seq[cs:ce] + rpad, (cs, ce)


def center_sequence(seq, total_len=INPUT_LEN, pad="shuffle", seed=0):
    """Center `seq` within a window of `total_len`. Returns (new_seq, (start, end))
    where (start, end) is the offset of the original seq within the window.

    pad : "shuffle" -> flanks = composition-matched random DNA (valid one-hot,
                       safe for deep_lift_shap; deterministic via `seed`)
          "N"       -> flanks = N (predictions only; deep_lift_shap will reject)
          any 1-char -> that character
    If seq is longer than total_len it is centrally cropped.
    """
    L = len(seq)
    if L == total_len:
        return seq, (0, L)
    if L > total_len:
        off = (L - total_len) // 2
        return seq[off:off + total_len], (-off, -off + L)
    left = (total_len - L) // 2
    right = total_len - L - left
    if pad == "shuffle":
        lpad = _composition_matched_pad(seq, left, seed)
        rpad = _composition_matched_pad(seq, right, seed + 1)
    else:
        lpad, rpad = pad * left, pad * right
    return lpad + seq + rpad, (left, left + L)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_model(celltype_or_path, device=None):
    """Load a ChromBPNet (no-bias) model via bpnetlite. Accepts a cell-type name
    or a direct .h5 path."""
    h5 = celltype_or_path
    if not str(celltype_or_path).endswith(".h5"):
        h5 = model_path(celltype_or_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BPNet.from_chrombpnet(h5).to(device).eval()
    return model


def _log_softmax_profile(y_profile):
    z = y_profile.shape
    y = y_profile.reshape(y_profile.shape[0], -1)
    y = torch.nn.functional.log_softmax(y, dim=-1)
    return y.reshape(*z)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, seq, strand_average=True):
    """Predict accessibility for a single sequence.

    Returns a dict with:
        counts        : scalar, total predicted counts (linear, not log)
        log_counts    : scalar, log of counts
        profile       : (OUTPUT_LEN,) softmax profile probabilities
        pred          : (OUTPUT_LEN,) counts * profile  (per-bp predicted signal)
    Strand-averaged (fwd + revcomp) by default, matching the variant-test nb.
    """
    device = next(model.parameters()).device

    def _one(s):
        X = torch.tensor(ohe(s)).float().to(device)
        prof, cnt = _tm_predict(model, X, device=device.type)
        cnt = float(np.exp(cnt.cpu().numpy().squeeze()))
        prof = torch.exp(_log_softmax_profile(prof)).cpu().numpy().squeeze()
        return cnt, prof

    f_cnt, f_prof = _one(seq)
    if strand_average:
        r_cnt, r_prof = _one(revcomp(seq))
        r_prof = r_prof[::-1]                 # flip rev profile back to fwd coords
        counts = (f_cnt + r_cnt) / 2.0
        profile = (f_prof + r_prof) / 2.0
    else:
        counts, profile = f_cnt, f_prof

    return {
        "counts": counts,
        "log_counts": float(np.log(counts)),
        "profile": profile,
        "pred": counts * profile,
    }


# ----------------------------------------------------------------------------
# Contributions
# ----------------------------------------------------------------------------
def count_contributions(model, seq, n_shuffles=20, batch_size=16, dtype=torch.float64,
                        random_state=0):
    """DeepLIFT/SHAP count-head contribution scores for a single sequence.

    `random_state` fixes the dinucleotide-shuffle references so attributions (and
    the downstream seqlet calls/annotations) are reproducible run-to-run.
    Returns (4, L) numpy array of per-base attributions (fwd strand).
    """
    device = next(model.parameters()).device
    wrapper = CountWrapper(ControlWrapper(model)).to(device).eval()
    X = torch.tensor(ohe(seq)).type(dtype).to(device)
    attr = deep_lift_shap(
        wrapper.type(dtype),
        X,
        n_shuffles=n_shuffles,
        batch_size=batch_size,
        device=device.type,
        random_state=random_state,
        verbose=True,
        warning_threshold=1e-4,
    )
    return attr[0].cpu().numpy().astype("float32")   # (4, L)
