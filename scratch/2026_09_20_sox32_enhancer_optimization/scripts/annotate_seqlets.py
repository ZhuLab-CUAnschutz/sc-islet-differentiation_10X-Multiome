#!/usr/bin/env python
"""
annotate_seqlets.py
===================
Phase 2 of the SOX32 enhancer optimization annotation pipeline.

Reads CREsted EIG attributions from step_attributions.npz, calls seqlets
with tangermeme recursive_seqlets, annotates them with TOMTOM against the
catalog+FOXH1 MEME database, and computes per-step deltas (gain/loss/change).

Follows the NKX6-1 Ledidi notebook pattern (cells 47-52 on narrows).

Outputs
-------
  beds/seqlet_annotations.bed   — all seqlets across all 61 step sequences
  tables/delta_seqlets.tsv      — gained / lost / changed motifs at each step
  tables/seqlet_summary.tsv     — per-step motif counts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tangermeme.seqlet import recursive_seqlets
from tangermeme.annotate import annotate_seqlets
from tangermeme.io import read_meme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ATTR_NPZ = ROOT / "results" / "step_attributions.npz"
MEME_DB = ROOT / "inputs" / "catalog_plus_foxh1.meme"
CATALOG_TSV = (
    ROOT.parent / "2026_08_17_tf_hits" / "motif_catalog.tsv"
)

BED_OUT = ROOT / "beds" / "seqlet_annotations.bed"
DELTA_OUT = ROOT / "tables" / "delta_seqlets.tsv"
SUMMARY_OUT = ROOT / "tables" / "seqlet_summary.tsv"

# Construct geometry (from Sep 7 config.py)
INPUT_LEN = 2114
ENHANCER_START = 507
ENHANCER_END = 1607
FEATURES = [
    ("5p_flank", 0, 507),
    ("enhancer", 507, 1607),
    ("min_promoter", 1607, 1647),
    ("eGFP_CDS", 1647, 2114),
]

OBJECTIVES_ORDERED = ("de_activity", "crested_default", "de_specificity")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def feature_of_pos(pos: int) -> str:
    for name, start, end in FEATURES:
        if start <= pos < end:
            return name
    return "unknown"


def overlap_len(s1: int, e1: int, s2: int, e2: int) -> int:
    return max(0, min(e1, e2) - max(s1, s2))


def greedy_match(
    design_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Match seqlets between design and baseline by overlap + annotation."""
    cand = (
        design_df.reset_index()
        .merge(baseline_df.reset_index(), on="annotation", suffixes=("_d", "_b"))
    )
    cand["overlap"] = cand.apply(
        lambda r: overlap_len(r["start_d"], r["end_d"], r["start_b"], r["end_b"]),
        axis=1,
    )
    cand = cand[cand["overlap"] > 0].copy()
    cand["start_gap"] = (cand["start_d"] - cand["start_b"]).abs()
    cand = cand.sort_values(["overlap", "start_gap"], ascending=[False, True])

    used_d: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, r in cand.iterrows():
        di, bi = int(r["index_d"]), int(r["index_b"])
        if di not in used_d and bi not in used_b:
            used_d.add(di)
            used_b.add(bi)
            matches.append((di, bi))
    return matches, used_d, used_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load attributions -------------------------------------------------
    print(f"Loading attributions from {ATTR_NPZ}", flush=True)
    d = np.load(str(ATTR_NPZ), allow_pickle=True)
    seq_ids = d["seq_ids"].astype(str)
    objectives = d["objectives"].astype(str)
    steps = d["steps"].astype(int)
    n = len(seq_ids)
    print(f"  {n} sequences", flush=True)

    # ---- Load motif database -----------------------------------------------
    print(f"Loading motif MEME from {MEME_DB}", flush=True)
    motifs = read_meme(str(MEME_DB))
    motif_names = np.array(list(motifs.keys()))
    print(f"  {len(motif_names)} motifs in catalog", flush=True)

    # Build label map (motif_name → human-readable label)
    motif_id_map: dict[str, str] = {}
    if CATALOG_TSV.exists():
        cat = pd.read_csv(str(CATALOG_TSV), sep="\t")
        for _, row in cat.iterrows():
            motif_id_map[str(row["motif_name"])] = str(row["label"])
    motif_id_map["FOXH1_JASPAR"] = "FOXH1_JASPAR"

    # ---- Call seqlets + annotate per sequence --------------------------------
    all_seqlets: list[pd.DataFrame] = []

    for i in range(n):
        sid = str(seq_ids[i])
        attr = d[f"attr__{sid}"].astype("float32")  # (4, L)
        ohe = d[f"ohe__{sid}"].astype("float32")     # (4, L)
        de_pred = float(d[f"pred__{sid}"])

        # Projected observed-base attributions, enhancer region only
        proj = (attr * ohe).sum(axis=0)  # (L,)
        proj_enh = proj[ENHANCER_START:ENHANCER_END]
        total_attr = float(np.abs(proj_enh).sum())

        # Call seqlets (Tutorial A4)
        proj_t = torch.tensor(proj_enh[None], dtype=torch.float32)  # (1, enh_len)
        seqlets = recursive_seqlets(proj_t, threshold=0.05, additional_flanks=5)

        if len(seqlets) == 0:
            print(f"  [{i+1}/{n}] {sid}: 0 seqlets", flush=True)
            continue

        # Annotate (Tutorial A5)
        ohe_enh = torch.tensor(
            ohe[None, :, ENHANCER_START:ENHANCER_END], dtype=torch.float32
        )
        motif_idxs = annotate_seqlets(ohe_enh, seqlets, motifs)[0][:, 0]
        ann_names = motif_names[motif_idxs.numpy()]

        # Shift coordinates back to full construct
        seqlets["start"] += ENHANCER_START
        seqlets["end"] += ENHANCER_START

        # Build per-seqlet dataframe
        df = seqlets.copy()
        df["seq_id"] = sid
        df["objective"] = str(objectives[i])
        df["step"] = int(steps[i])
        df["annotation"] = ann_names
        df["label"] = [motif_id_map.get(a, a) for a in ann_names]
        df["de_pred"] = de_pred
        df["feature"] = [feature_of_pos((s + e) // 2) for s, e in zip(df["start"], df["end"])]
        if total_attr > 0:
            df["attr_proportion"] = df["attribution"].abs() / total_attr
        else:
            df["attr_proportion"] = 0.0

        all_seqlets.append(df)
        print(
            f"  [{i+1}/{n}] {sid}: {len(df)} seqlets, "
            f"DE_pred={de_pred:.1f}, total_attr={total_attr:.1f}",
            flush=True,
        )

    # ---- Combine all seqlets -------------------------------------------------
    combined = pd.concat(all_seqlets, ignore_index=True)
    print(f"\n  Total seqlets: {len(combined)}", flush=True)

    # ---- Write BED -----------------------------------------------------------
    BED_OUT.parent.mkdir(parents=True, exist_ok=True)
    bed = pd.DataFrame(
        {
            "chrom": combined["seq_id"],
            "start": combined["start"],
            "end": combined["end"],
            "name": combined["label"],
            "score": (combined["attribution"].abs() * 10).round().astype(int),
            "strand": ".",
            "objective": combined["objective"],
            "step": combined["step"],
            "seqlet_attr": combined["attribution"].round(4),
            "attr_proportion": combined["attr_proportion"].round(4),
            "feature": combined["feature"],
            "de_pred": combined["de_pred"].round(2),
        }
    )
    bed.to_csv(str(BED_OUT), sep="\t", index=False)
    print(f"Wrote {BED_OUT}", flush=True)

    # ---- Seqlet summary per step ---------------------------------------------
    summary_rows: list[dict] = []
    for (obj, step), grp in combined.groupby(["objective", "step"]):
        counts = grp["label"].value_counts().to_dict()
        for label, count in counts.items():
            summary_rows.append(
                {"objective": obj, "step": step, "label": label, "count": count}
            )
    summary = pd.DataFrame(summary_rows)
    if len(summary) > 0:
        summary = summary.sort_values(["objective", "step", "label"])
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(SUMMARY_OUT), sep="\t", index=False)
    print(f"Wrote {SUMMARY_OUT}", flush=True)

    # ---- Delta seqlets (key deliverable) -------------------------------------
    delta_rows: list[dict] = []

    # Load trajectory_metrics for mutation positions
    traj_path = ROOT.parent / "2026_09_07_sox32_enhancer_optimization" / "tables" / "trajectory_metrics.tsv"
    traj = pd.read_csv(str(traj_path), sep="\t") if traj_path.exists() else None

    for obj in OBJECTIVES_ORDERED:
        obj_seqlets = combined[combined["objective"] == obj]
        wt_seqlets = combined[combined["step"] == 0]
        if len(wt_seqlets) == 0:
            continue

        prev_df = wt_seqlets.copy()
        for step in range(1, 21):
            curr_df = obj_seqlets[obj_seqlets["step"] == step].copy()
            if len(curr_df) == 0:
                prev_df = curr_df
                continue

            # Get mutation info for this step
            mut_pos, mut_ref, mut_alt = None, None, None
            if traj is not None:
                row = traj[(traj["objective"] == obj) & (traj["step"] == step)]
                if len(row) > 0:
                    row = row.iloc[0]
                    mut_pos = int(row["step_position_1based"]) if pd.notna(row.get("step_position_1based")) else None
                    mut_ref = str(row.get("step_ref", ""))
                    mut_alt = str(row.get("step_alt", ""))

            # Greedy match
            matches, used_d, used_b = greedy_match(
                curr_df.reset_index(drop=True),
                prev_df.reset_index(drop=True),
            )

            # Additions: seqlets in design but not matched
            for di in range(len(curr_df)):
                if di not in used_d:
                    r = curr_df.iloc[di]
                    delta_rows.append(
                        {
                            "objective": obj,
                            "step": step,
                            "mutation_pos": mut_pos,
                            "ref": mut_ref,
                            "alt": mut_alt,
                            "type": "addition",
                            "annotation": r["annotation"],
                            "label": r["label"],
                            "start": r["start"],
                            "end": r["end"],
                            "seqlet_attr": round(float(r["attribution"]), 4),
                            "delta_attr": round(float(r["attribution"]), 4),
                            "feature": r["feature"],
                        }
                    )

            # Removals: seqlets in baseline but not matched
            for bi in range(len(prev_df)):
                if bi not in used_b:
                    r = prev_df.iloc[bi]
                    delta_rows.append(
                        {
                            "objective": obj,
                            "step": step,
                            "mutation_pos": mut_pos,
                            "ref": mut_ref,
                            "alt": mut_alt,
                            "type": "removal",
                            "annotation": r["annotation"],
                            "label": r["label"],
                            "start": r["start"],
                            "end": r["end"],
                            "seqlet_attr": None,
                            "delta_attr": round(-float(r["attribution"]), 4),
                            "feature": r["feature"],
                        }
                    )

            # Matched: increase / decrease / stable
            for di, bi in matches:
                rd = curr_df.iloc[di]
                rb = prev_df.iloc[bi]
                delta = float(rd["attribution"]) - float(rb["attribution"])
                if abs(delta) < 0.01:
                    change_type = "stable"
                elif delta > 0:
                    change_type = "increase"
                else:
                    change_type = "decrease"
                delta_rows.append(
                    {
                        "objective": obj,
                        "step": step,
                        "mutation_pos": mut_pos,
                        "ref": mut_ref,
                        "alt": mut_alt,
                        "type": change_type,
                        "annotation": rd["annotation"],
                        "label": rd["label"],
                        "start": rd["start"],
                        "end": rd["end"],
                        "seqlet_attr": round(float(rd["attribution"]), 4),
                        "delta_attr": round(delta, 4),
                        "feature": rd["feature"],
                    }
                )

            prev_df = curr_df

    delta_df = pd.DataFrame(delta_rows)
    if len(delta_df) > 0:
        delta_df = delta_df.sort_values(["objective", "step", "type", "start"])
    DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(str(DELTA_OUT), sep="\t", index=False)
    print(f"Wrote {DELTA_OUT}  ({len(delta_df)} delta events)", flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
