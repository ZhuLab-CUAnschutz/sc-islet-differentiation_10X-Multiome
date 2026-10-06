#!/usr/bin/env python
"""
09_crested_predict.py — predict the sox32 construct through the project's CREsted
models, as an alternative to the single-task ChromBPNet ensemble:
  * MULTI-TASK model  (2114 bp -> 22 cell-type accessibility outputs)
  * TOPIC model n50    (501 bp central window -> 50 topic activations)

Motivation: the single-task model predicts this enhancer near the bottom of DE's
peak distribution (weak signal => noisy contributions). A model that assigns it
more/cleaner signal would make motif analysis tractable. This compares what each
model says about cell-type / topic specificity.

Runs under the CREsted Keras env (NOT torch): /cellar/users/aklie/opt/deeptopic/.venv-keras
Reads the construct + enh_dinuc sequences from the npz (stored as strings; no torch).
GPU SLURM job.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
import numpy as np
import pandas as pd
import keras

NPZ = "outputs/sox32_all.npz"
OUT = "outputs"
MT_MODEL = ("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
            "results/4_multi_task_model/crested/finetuned/checkpoints/23.keras")
TOPIC_MODEL = ("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
               "results/4_topic_models/all/n50/crested_model/checkpoints/76.keras")
TOPIC_ANNOT = ("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
               "results/4_topic_models/all/n50/annotated/synthesis/topic_annotations.tsv")
MT_CELLTYPES = ["DE","ENP_phase1","FB_FLT1","PFG1","PFG2","PGT1","PGT2","PGT3","PP1","PP2",
                "SC_delta_GHRL","early_ENP","early_SC_EC","early_SC_alpha","early_SC_beta",
                "exocrine","late_ENP","late_SC_EC","late_SC_alpha","late_SC_beta","liver",
                "proliferating_endocrine"]
_B = {"A": 0, "C": 1, "G": 2, "T": 3}


def ohe(seq, L, center=False):
    """(1, L, 4) one-hot, ACGT. If center, take the central L bp of seq."""
    seq = seq.upper()
    if center and len(seq) > L:
        s = (len(seq) - L) // 2
        seq = seq[s:s + L]
    seq = seq[:L]
    x = np.zeros((1, L, 4), dtype="float32")
    for i, b in enumerate(seq):
        if b in _B:
            x[0, i, _B[b]] = 1.0
    return x


def main():
    d = np.load(NPZ, allow_pickle=True)
    seqs = {v: str(d[f"seq__{v}"]) for v in [str(x) for x in d["variants"]]}
    print("variants:", list(seqs), "| lengths:", {k: len(v) for k, v in seqs.items()}, flush=True)

    # ---- multitask (2114 bp -> 22 cell types) ----
    mt = keras.models.load_model(MT_MODEL, compile=False)
    rows = {}
    for v, s in seqs.items():
        rows[v] = mt.predict(ohe(s, 2114), verbose=0)[0]
    mt_df = pd.DataFrame(rows, index=MT_CELLTYPES).rename_axis("celltype").reset_index()
    mt_df.to_csv(f"{OUT}/sox32_multitask_pred.csv", index=False)
    print("\n=== MULTITASK predictions (sorted by construct) ===")
    print(mt_df.sort_values("construct", ascending=False).round(3).to_string(index=False))

    # ---- topic n50 (central 501 bp -> 50 topics) ----
    tm = keras.models.load_model(TOPIC_MODEL, compile=False)
    trows = {}
    for v, s in seqs.items():
        trows[v] = tm.predict(ohe(s, 501, center=True), verbose=0)[0]
    tdf = pd.DataFrame(trows)
    tdf.insert(0, "topic", [f"Topic{i+1}" for i in range(len(tdf))])
    try:
        ann = pd.read_csv(TOPIC_ANNOT, sep="\t")
        lab = dict(zip(ann["topic"].astype(str), ann.get("label", ann.iloc[:, 1]).astype(str)))
        tdf["label"] = [lab.get(t, "") for t in tdf["topic"]]
    except Exception as e:
        print("topic annot load failed:", e)
        tdf["label"] = ""
    tdf.to_csv(f"{OUT}/sox32_topic_pred.csv", index=False)
    print("\n=== TOPIC model: top 12 topics by construct activation ===")
    print(tdf.sort_values("construct", ascending=False).head(12).round(3).to_string(index=False))
    print("\n[done] wrote outputs/sox32_multitask_pred.csv + sox32_topic_pred.csv")


if __name__ == "__main__":
    main()
