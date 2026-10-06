#!/usr/bin/env python
"""
10_multitask_contribs.py — per-cell-type contribution scores for the construct
from the MULTI-TASK CREsted model, so we can see how the attributed motifs vary
across cell types (the multitask model is calibrated for cross-cell-type
specificity, unlike the 22 independent single-task models).

Uses crested.tl.contribution_scores (expected_integrated_grad, seed=42) for both
variants x all 22 cell-type outputs in one call, and saves to an npz in the SAME
schema as 01_predict so the existing panel/seqlet tooling (02_panels.py,
05_contrib_trajectory.py) works unchanged (logos-only: multitask has no profile).

Runs under the CREsted Keras env (NOT torch). GPU SLURM.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras
import crested

NPZ_IN = "outputs/sox32_all.npz"          # source of the variant sequences (strings)
NPZ_OUT = "outputs/sox32_multitask_contribs.npz"
MODEL = ("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
         "results/4_multi_task_model/crested/finetuned/checkpoints/23.keras")
CELLTYPES = ["DE","ENP_phase1","FB_FLT1","PFG1","PFG2","PGT1","PGT2","PGT3","PP1","PP2",
             "SC_delta_GHRL","early_ENP","early_SC_EC","early_SC_alpha","early_SC_beta",
             "exocrine","late_ENP","late_SC_EC","late_SC_alpha","late_SC_beta","liver",
             "proliferating_endocrine"]
_B = {"A": 0, "C": 1, "G": 2, "T": 3}


def ohe_LX4(seq):
    x = np.zeros((len(seq), 4), dtype="float32")
    for i, b in enumerate(seq.upper()):
        if b in _B:
            x[i, _B[b]] = 1.0
    return x


def main():
    d = np.load(NPZ_IN, allow_pickle=True)
    variants = [str(v) for v in d["variants"]]
    seqs = {v: str(d[f"seq__{v}"]) for v in variants}
    print("variants:", variants, flush=True)

    model = keras.models.load_model(MODEL, compile=False)

    out = {"name": "sox32_enh", "celltypes": np.array(CELLTYPES),
           "variants": np.array(variants), "input_len": 2114, "output_len": 1000}

    for v in variants:
        s = seqs[v]
        # predictions (per-cell-type scalar)
        pred = model.predict(ohe_LX4(s)[None], verbose=0)[0]      # (22,)
        # contributions for all 22 targets at once
        scores, ohs = crested.tl.contribution_scores(
            input=[s], target_idx=list(range(len(CELLTYPES))), model=model,
            method="expected_integrated_grad", transpose=True, batch_size=64,
            seed=42, verbose=True)
        scores = np.asarray(scores)        # (1, 22, 4, 2114)
        ohs = np.asarray(ohs)              # (1, 4, 2114)
        out[f"ohe__{v}"] = ohs[0].astype("float32")
        out[f"seq__{v}"] = s
        out[f"meta_offset__{v}"] = 0
        for j, ct in enumerate(CELLTYPES):
            out[f"attr__{v}__{ct}"] = scores[0, j].astype("float32")
            out[f"counts__{v}__{ct}"] = float(pred[j])
        print(f"  {v}: contribs {scores.shape}, top cell = {CELLTYPES[int(np.argmax(pred))]}", flush=True)

    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    np.savez_compressed(NPZ_OUT, **out)
    print(f"[done] wrote {NPZ_OUT}", flush=True)


if __name__ == "__main__":
    main()
