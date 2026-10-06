#!/usr/bin/env python
"""
plot_logos.py
=============
Logo-plot visualizations of per-base DE attributions across optimization steps.

Figures
-------
1. Static logo panels — 5 selected iterations (WT, 5, 10, 15, 20) per objective,
   focused on the 5' hotspot region of the enhancer.
2. Animated GIF — all 21 steps cycling for each objective.
"""

from __future__ import annotations

from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ATTR_NPZ = ROOT / "results" / "step_attributions.npz"
BED = ROOT / "beds" / "seqlet_annotations.bed"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

ENHANCER_START = 507
ENHANCER_END = 1607

# Fixed window for logo plots: enhancer positions 30-280 (construct 537-787)
# This covers the 5' hotspot where attribution concentrates.
LOGO_START = 537   # construct coords
LOGO_END = 787     # construct coords
LOGO_LEN = LOGO_END - LOGO_START

OBJECTIVES_ORDERED = ["de_activity", "crested_default", "de_specificity"]
OBJ_LABELS = {
    "de_activity": "DE activity",
    "crested_default": "CREsted default",
    "de_specificity": "DE specificity",
}

SELECTED_STEPS = [0, 5, 10, 15, 20]

SEQLET_COLORS = {
    "ST18":             "#e41a1c",
    "ST47":             "#fb9a99",
    "ST36_sub3_LMX1A":  "#ff7f00",
    "ST36_sub2_ISL1":   "#fdbf6f",
    "FOXH1_JASPAR":     "#984ea3",
    "ST27":             "#4daf4a",
    "ST19":             "#377eb8",
    "ST55":             "#a65628",
    "ST37":             "#e78ac3",
    "ST46":             "#f781bf",
}

MOTIF_DISPLAY = {
    "ST18":             "EOMES",
    "ST47":             "T-box",
    "ST36_sub3_LMX1A":  "LMX1A",
    "ST36_sub2_ISL1":   "ISL1",
    "FOXH1_JASPAR":     "FOXH1",
    "ST27":             "GATA4/6",
    "ST19":             "SOX17",
    "ST55":             "FOX",
    "ST37":             "FOX-2",
    "ST46":             "HNF/FOXA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data():
    d = np.load(str(ATTR_NPZ), allow_pickle=True)
    seq_ids = d["seq_ids"].astype(str)
    objectives = d["objectives"].astype(str)
    steps = d["steps"].astype(int)
    bed = pd.read_csv(str(BED), sep="\t")
    return d, seq_ids, objectives, steps, bed


def get_logo_df(d, seq_id: str) -> pd.DataFrame:
    """Return a logomaker-compatible DataFrame for the logo window."""
    attr = d[f"attr__{seq_id}"].astype("float64")   # (4, L)
    ohe = d[f"ohe__{seq_id}"].astype("float64")      # (4, L)
    # Per-nucleotide attribution: attr * ohe, shape (4, L)
    contrib = attr * ohe
    # Slice to logo window, transpose to (L, 4)
    window = contrib[:, LOGO_START:LOGO_END].T
    positions = np.arange(LOGO_START - ENHANCER_START, LOGO_END - ENHANCER_START)
    return pd.DataFrame(window, index=positions, columns=["A", "C", "G", "T"])


def get_seqlets_in_window(bed: pd.DataFrame, obj: str, step: int):
    """Return seqlets overlapping the logo window for a given obj/step."""
    if step == 0:
        mask = bed["objective"] == "wild_type"
    else:
        mask = (bed["objective"] == obj) & (bed["step"] == step)
    sub = bed[mask].copy()
    # Filter to those overlapping the window
    sub = sub[(sub["end"] > LOGO_START) & (sub["start"] < LOGO_END)]
    return sub


def draw_seqlet_track(ax_track, seqlets: pd.DataFrame, xlim: tuple):
    """Draw seqlet annotations on a dedicated track axis below the logo.

    Stacks overlapping seqlets on separate rows so labels don't collide.
    """
    ax_track.set_xlim(*xlim)
    ax_track.set_axis_off()

    if len(seqlets) == 0:
        return

    # Assign rows to avoid overlap
    entries = []
    for _, row in seqlets.iterrows():
        x_start = max(row["start"], LOGO_START) - ENHANCER_START
        x_end = min(row["end"], LOGO_END) - ENHANCER_START
        color = SEQLET_COLORS.get(row["name"], "#bdbdbd")
        label = MOTIF_DISPLAY.get(row["name"], row["name"])
        entries.append((x_start, x_end, color, label))

    # Sort by start position
    entries.sort(key=lambda e: e[0])

    # Greedy row assignment — place each entry in the first row where it fits
    row_ends: list[float] = []
    row_assignments: list[int] = []
    for xs, xe, _, lbl in entries:
        placed = False
        for ri, re in enumerate(row_ends):
            if xs >= re + 3:  # small gap between labels
                row_ends[ri] = xe
                row_assignments.append(ri)
                placed = True
                break
        if not placed:
            row_assignments.append(len(row_ends))
            row_ends.append(xe)

    n_rows = max(row_assignments) + 1 if row_assignments else 1
    row_height = 1.0 / max(n_rows, 1)

    ax_track.set_ylim(0, 1)

    for (xs, xe, color, label), ri in zip(entries, row_assignments):
        y_bottom = 1.0 - (ri + 1) * row_height
        y_mid = y_bottom + row_height * 0.5
        ax_track.add_patch(plt.Rectangle(
            (xs, y_bottom + row_height * 0.15), xe - xs, row_height * 0.7,
            facecolor=color, alpha=0.7, edgecolor="none",
        ))
        mid = (xs + xe) / 2
        ax_track.text(
            mid, y_mid, label, ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold",
        )


# ---------------------------------------------------------------------------
# Figure 1: Static logo panels
# ---------------------------------------------------------------------------
def plot_static_logos(d, bed: pd.DataFrame) -> None:
    n_steps = len(SELECTED_STEPS)
    n_obj = len(OBJECTIVES_ORDERED)
    xlim = (LOGO_START - ENHANCER_START - 1, LOGO_END - ENHANCER_START + 1)

    # Each step gets 2 rows: logo (tall) + seqlet track (short)
    height_ratios = []
    for _ in range(n_steps):
        height_ratios.extend([4, 1])

    fig, all_axes = plt.subplots(
        n_steps * 2, n_obj, figsize=(20, 16),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
    )

    # Compute global y-limits per objective
    obj_ylims = {}
    for obj in OBJECTIVES_ORDERED:
        ymax = 0
        ymin = 0
        for step in SELECTED_STEPS:
            sid = "SOX32_WT" if step == 0 else f"SOX32_{obj}_step{step:02d}"
            ldf = get_logo_df(d, sid)
            ymax = max(ymax, ldf.values.max())
            ymin = min(ymin, ldf.values.min())
        obj_ylims[obj] = (ymin * 1.1, ymax * 1.1)

    for si, step in enumerate(SELECTED_STEPS):
        logo_row = si * 2
        track_row = si * 2 + 1

        for oi, obj in enumerate(OBJECTIVES_ORDERED):
            ax_logo = all_axes[logo_row, oi]
            ax_track = all_axes[track_row, oi]

            sid = "SOX32_WT" if step == 0 else f"SOX32_{obj}_step{step:02d}"
            ldf = get_logo_df(d, sid)
            de_pred = float(d[f"pred__{sid}"])

            # Logo
            logomaker.Logo(
                ldf, ax=ax_logo, shade_below=0.5, fade_below=0.5,
                color_scheme="classic",
            )
            ax_logo.set_ylim(*obj_ylims[obj])
            ax_logo.set_xlim(*xlim)
            ax_logo.set_xticklabels([])
            ax_logo.tick_params(axis="y", labelsize=6)

            # Row label
            if oi == 0:
                step_label = "WT" if step == 0 else f"Step {step}"
                ax_logo.set_ylabel(
                    f"{step_label}\nDE={de_pred:.0f}",
                    fontsize=9, fontweight="bold",
                )
            else:
                ax_logo.set_ylabel(f"DE={de_pred:.0f}", fontsize=8)

            # Column header
            if si == 0:
                ax_logo.set_title(OBJ_LABELS[obj], fontsize=12, fontweight="bold")

            # Seqlet track
            seqlets = get_seqlets_in_window(bed, obj, step)
            draw_seqlet_track(ax_track, seqlets, xlim)

            # X-axis labels only on bottom track
            if si == n_steps - 1:
                ax_track.set_xlabel("Enhancer position (bp)", fontsize=9)

    fig.suptitle(
        "SOX32 DE enhancer — attribution logos at selected optimization steps\n"
        f"(enhancer positions {LOGO_START - ENHANCER_START}–{LOGO_END - ENHANCER_START} bp)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = FIGURES / "static_logos.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Animated GIF — logo + seqlet track, one per objective
# ---------------------------------------------------------------------------
def plot_animated_logos(d, bed: pd.DataFrame) -> None:
    for obj in OBJECTIVES_ORDERED:
        # Precompute all frames
        frames = []
        ylim_max = 0
        ylim_min = 0
        for step in range(21):
            sid = "SOX32_WT" if step == 0 else f"SOX32_{obj}_step{step:02d}"
            ldf = get_logo_df(d, sid)
            de_pred = float(d[f"pred__{sid}"])
            seqlets = get_seqlets_in_window(bed, obj, step)
            frames.append((ldf, de_pred, seqlets, step))
            ylim_max = max(ylim_max, ldf.values.max())
            ylim_min = min(ylim_min, ldf.values.min())
        ylim = (ylim_min * 1.1, ylim_max * 1.1)
        xlim = (LOGO_START - ENHANCER_START - 1, LOGO_END - ENHANCER_START + 1)

        fig, (ax_logo, ax_track) = plt.subplots(
            2, 1, figsize=(14, 4.5),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.02},
        )

        def update(frame_idx):
            ax_logo.clear()
            ax_track.clear()
            ldf, de_pred, seqlets, step = frames[frame_idx]

            # Logo
            logomaker.Logo(
                ldf, ax=ax_logo, shade_below=0.5, fade_below=0.5,
                color_scheme="classic",
            )
            ax_logo.set_ylim(*ylim)
            ax_logo.set_xlim(*xlim)
            step_label = "WT" if step == 0 else f"Step {step}"
            ax_logo.set_title(
                f"{OBJ_LABELS[obj]} — {step_label}  (DE pred = {de_pred:.0f})",
                fontsize=12, fontweight="bold",
            )
            ax_logo.set_ylabel("Attribution", fontsize=9)
            ax_logo.tick_params(labelsize=7)
            ax_logo.set_xticklabels([])

            # Seqlet track
            draw_seqlet_track(ax_track, seqlets, xlim)
            ax_track.set_xlabel("Enhancer position (bp)", fontsize=9)

        anim = FuncAnimation(fig, update, frames=len(frames),
                             interval=500, repeat=True)
        out = FIGURES / f"logo_animation_{obj}.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        plt.close(fig)
        print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    d, seq_ids, objectives, steps, bed = load_data()
    print(f"Loaded {len(seq_ids)} sequences, {len(bed)} seqlet records")

    plot_static_logos(d, bed)
    plot_animated_logos(d, bed)

    print(f"\n[done] All figures in {FIGURES}")


if __name__ == "__main__":
    main()
