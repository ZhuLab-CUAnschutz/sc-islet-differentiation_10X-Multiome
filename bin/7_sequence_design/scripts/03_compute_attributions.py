#!/usr/bin/env python
"""
compute_attributions.py
=======================
Compute CREsted multitask DE-head attribution scores (expected integrated
gradients) for every step sequence in the SOX32 enhancer greedy optimization.

Reads evolution.npz written by 01_optimize.py, iterates the 61 unique sequences
(WT + 3 objectives × 20 steps), and saves per-sequence DE attributions and
one-hot encodings in a single npz ready for tangermeme seqlet calling.

Runs on GPU under the CREsted Keras environment (KERAS_BACKEND=torch).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import crested

# ---------------------------------------------------------------------------
# Paths  (resolve relative to the script's parent = project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

EVOLUTION_NPZ = config.EVOLUTION_NPZ
MODEL_PATH = config.MODEL_PATH
OUTPUT_NPZ = config.ATTRIBUTIONS_NPZ

MODEL_CLASSES = config.MODEL_CLASSES
DE_INDEX = config.TARGET_INDEX
REFERENCE_SHA256 = config.REFERENCE_SEQUENCE_SHA256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_B = {"A": 0, "C": 1, "G": 2, "T": 3}


def ohe_LX4(seq: str) -> np.ndarray:
    """One-hot encode a sequence in (L, 4) layout for CREsted."""
    x = np.zeros((len(seq), 4), dtype="float32")
    for i, b in enumerate(seq.upper()):
        if b in _B:
            x[i, _B[b]] = 1.0
    return x


def seq_sha256(seq: str) -> str:
    import hashlib
    return hashlib.sha256(seq.upper().encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load evolution data ------------------------------------------------
    print(f"Loading evolution data from {EVOLUTION_NPZ}", flush=True)
    evo = np.load(str(EVOLUTION_NPZ), allow_pickle=True)
    objectives = evo["objectives"].astype(str)   # (3,)
    sequences = evo["sequences"].astype(str)     # (3, 21)

    # Build unique (seq_id, sequence, objective, step) records.
    # Step 0 is the WT, shared across all objectives — store once.
    wt_seq = str(sequences[0, 0])
    assert seq_sha256(wt_seq) == REFERENCE_SHA256, "WT sequence SHA256 mismatch"

    records: list[tuple[str, str, str, int]] = [("SOX32_WT", wt_seq, "wild_type", 0)]
    for oi, obj in enumerate(objectives):
        for step in range(1, 21):
            sid = f"SOX32_{obj}_step{step:02d}"
            records.append((sid, str(sequences[oi, step]), str(obj), step))
    print(f"  {len(records)} unique sequences to process", flush=True)

    # ---- Load model ---------------------------------------------------------
    print(f"Loading CREsted model from {MODEL_PATH}", flush=True)
    model = keras.models.load_model(str(MODEL_PATH), compile=False)

    # ---- Compute attributions -----------------------------------------------
    out: dict[str, object] = {
        "seq_ids": np.array([r[0] for r in records]),
        "objectives": np.array([r[2] for r in records]),
        "steps": np.array([r[3] for r in records], dtype=np.int32),
    }

    for i, (seq_id, seq, obj, step) in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {seq_id} ...", end="", flush=True)

        # Prediction (all 22 cell types)
        pred = model.predict(ohe_LX4(seq)[None], verbose=0)[0]  # (22,)
        de_pred = float(pred[DE_INDEX])

        # DE-head attribution via expected integrated gradients
        scores, ohs = crested.tl.contribution_scores(
            input=[seq],
            target_idx=[DE_INDEX],
            model=model,
            method="expected_integrated_grad",
            transpose=True,
            batch_size=64,
            seed=42,
            verbose=False,
        )
        # scores shape: (1, n_targets, 4, L) → squeeze to (4, L)
        attr = np.asarray(scores)[0, 0].astype("float32")
        ohe = np.asarray(ohs)[0].astype("float32")  # (4, L)

        out[f"attr__{seq_id}"] = attr
        out[f"ohe__{seq_id}"] = ohe
        out[f"pred__{seq_id}"] = de_pred

        print(f" DE_pred={de_pred:.1f}  attr_range=[{attr.min():.4f}, {attr.max():.4f}]",
              flush=True)

    # ---- Save ---------------------------------------------------------------
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUTPUT_NPZ), **out)
    print(f"\n[done] Wrote {OUTPUT_NPZ}  ({OUTPUT_NPZ.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
