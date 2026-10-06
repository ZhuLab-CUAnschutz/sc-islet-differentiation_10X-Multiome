#!/usr/bin/env python
"""
plot_de_specificity.py
======================
Single-objective (DE specificity) visualizations of the SOX32 enhancer
optimization — waterfall, line traces, stacked bar, static logos, animation.
"""

from __future__ import annotations

from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
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

OBJ = "de_specificity"
OBJ_LABEL = "DE specificity"

# Logo window
LOGO_START = 537
LOGO_END = 787

SELECTED_STEPS = [0, 1, 5, 10, 15, 20]

# ---------------------------------------------------------------------------
# Per-motif colors and display names
# ---------------------------------------------------------------------------
MOTIF_COLORS = {
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

MOTIF_TO_FAMILY = {}
for m in ["ST18", "ST47"]:
    MOTIF_TO_FAMILY[m] = "T-box"
for m in ["ST36_sub3_LMX1A", "ST36_sub2_ISL1"]:
    MOTIF_TO_FAMILY[m] = "LMX1/ISL1"
MOTIF_TO_FAMILY["FOXH1_JASPAR"] = "FOXH1"
MOTIF_TO_FAMILY["ST27"] = "GATA"
MOTIF_TO_FAMILY["ST19"] = "SOX"
for m in ["ST55", "ST37"]:
    MOTIF_TO_FAMILY[m] = "FOX"
MOTIF_TO_FAMILY["ST46"] = "HNF/FOXA"

FAMILY_COLORS = {
    "T-box":     "#e41a1c",
    "LMX1/ISL1": "#ff7f00",
    "FOXH1":     "#984ea3",
    "GATA":      "#4daf4a",
    "SOX":       "#377eb8",
    "FOX":       "#a65628",
    "HNF/FOXA":  "#f781bf",
    "Other":     "#bdbdbd",
}

OTHER_COLOR = "#bdbdbd"


def motif_color(label: str) -> str:
    return MOTIF_COLORS.get(label, OTHER_COLOR)


def motif_display(label: str) -> str:
    return MOTIF_DISPLAY.get(label, label)


CURATED_MOTIFS = [
    "ST18", "ST47", "ST36_sub3_LMX1A", "ST36_sub2_ISL1",
    "FOXH1_JASPAR", "ST27", "ST19", "ST55", "ST46",
]

TOP_BAR_MOTIFS = [
    "ST18", "ST47", "ST36_sub3_LMX1A", "ST36_sub2_ISL1",
    "FOXH1_JASPAR", "ST27", "ST19", "ST55", "ST46", "ST37",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_bed() -> pd.DataFrame:
    df = pd.read_csv(str(BED), sep="\t")
    wt = df[df["objective"] == "wild_type"].copy()
    wt["objective"] = OBJ
    spec = df[df["objective"] == OBJ]
    df = pd.concat([wt, spec], ignore_index=True)
    df["color"] = df["name"].map(motif_color)
    df["family"] = df["name"].map(lambda x: MOTIF_TO_FAMILY.get(x, "Other"))
    return df


def load_attr():
    d = np.load(str(ATTR_NPZ), allow_pickle=True)
    return d


def get_logo_df(d, seq_id: str) -> pd.DataFrame:
    attr = d[f"attr__{seq_id}"].astype("float64")
    ohe = d[f"ohe__{seq_id}"].astype("float64")
    contrib = attr * ohe
    window = contrib[:, LOGO_START:LOGO_END].T
    positions = np.arange(LOGO_START - ENHANCER_START, LOGO_END - ENHANCER_START)
    return pd.DataFrame(window, index=positions, columns=["A", "C", "G", "T"])


def get_seqlets_in_window(bed: pd.DataFrame, step: int):
    if step == 0:
        mask = bed["step"] == 0
    else:
        mask = (bed["objective"] == OBJ) & (bed["step"] == step)
    sub = bed[mask].copy()
    return sub[(sub["end"] > LOGO_START) & (sub["start"] < LOGO_END)]


def draw_seqlet_track(ax_track, seqlets: pd.DataFrame, xlim: tuple):
    ax_track.set_xlim(*xlim)
    ax_track.set_axis_off()
    if len(seqlets) == 0:
        return

    entries = []
    for _, row in seqlets.iterrows():
        x_start = max(row["start"], LOGO_START) - ENHANCER_START
        x_end = min(row["end"], LOGO_END) - ENHANCER_START
        color = MOTIF_COLORS.get(row["name"], OTHER_COLOR)
        label = MOTIF_DISPLAY.get(row["name"], row["name"])
        entries.append((x_start, x_end, color, label))

    entries.sort(key=lambda e: e[0])
    row_ends: list[float] = []
    row_assignments: list[int] = []
    for xs, xe, _, lbl in entries:
        placed = False
        for ri, re in enumerate(row_ends):
            if xs >= re + 3:
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
            fontsize=7, color="white", fontweight="bold",
        )


# ---------------------------------------------------------------------------
# Figure 1: Waterfall — single panel
# ---------------------------------------------------------------------------
def plot_waterfall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sub = df.copy()

    step_max = sub.groupby("step")["seqlet_attr"].apply(
        lambda s: s.abs().quantile(0.95) if len(s) > 0 else 1.0
    ).to_dict()

    for _, row in sub.iterrows():
        x = row["start"] - ENHANCER_START
        w = row["end"] - row["start"]
        y = row["step"]
        norm = step_max.get(y, 1.0) or 1.0
        alpha = min(1.0, max(0.2, abs(row["seqlet_attr"]) / norm))
        base_color = row["color"]
        fc = to_rgba(base_color, alpha=alpha)
        ax.add_patch(plt.Rectangle(
            (x, y - 0.4), w, 0.8,
            facecolor=fc, edgecolor=base_color, linewidth=0.4,
        ))

    ax.set_xlim(0, ENHANCER_END - ENHANCER_START)
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlabel("Enhancer position (bp)", fontsize=11)
    ax.set_ylabel("Optimization step", fontsize=11)
    ax.set_title(f"SOX32 DE enhancer — {OBJ_LABEL}\nseqlet footprints across optimization",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_yticks(range(0, 21))
    ax.set_yticklabels(["WT" if s == 0 else str(s) for s in range(21)], fontsize=8)

    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
               for f in FAMILY_COLORS if f != "Other"]
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = FIGURES / "de_specificity_waterfall.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    out_pdf = FIGURES / "de_specificity_waterfall.pdf"
    fig.savefig(str(out_pdf), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 2: Line traces — broken y-axis (EOMES dominates upper range)
# ---------------------------------------------------------------------------
def plot_line_traces(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["step", "name"])["seqlet_attr"]
        .sum()
        .reset_index()
    )

    # Precompute traces
    traces = {}
    for motif in CURATED_MOTIFS:
        m = agg[agg["name"] == motif].sort_values("step")
        full_steps = pd.DataFrame({"step": range(0, 21)})
        m = full_steps.merge(m, on="step", how="left").fillna(0)
        traces[motif] = m

    # Break point: gap between FOXH1 peak (~230) and EOMES (~730)
    BREAK_LO = 275
    BREAK_HI = 375

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.06},
    )

    for motif in CURATED_MOTIFS:
        m = traces[motif]
        for ax in (ax_top, ax_bot):
            ax.plot(
                m["step"], m["seqlet_attr"], "o-",
                color=MOTIF_COLORS[motif],
                label=MOTIF_DISPLAY[motif],
                markersize=4, linewidth=2,
            )

    # Y-axis ranges
    ax_top.set_ylim(BREAK_HI, 780)
    ax_bot.set_ylim(-10, BREAK_LO)

    # Hide spines at the break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    # Break marks (diagonal lines)
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d*2, +d*2), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d*2, +d*2), **kwargs)
    kwargs["transform"] = ax_bot.transAxes
    ax_bot.plot((-d, +d), (1 - d*2, 1 + d*2), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d*2, 1 + d*2), **kwargs)

    ax_bot.set_xlabel("Optimization step", fontsize=11)
    ax_bot.set_xlim(-0.5, 20.5)
    ax_bot.set_xticks(range(0, 21))
    ax_bot.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Shared y-label
    fig.text(0.02, 0.5, "Total seqlet attribution", va="center",
             rotation="vertical", fontsize=11)

    ax_top.set_title(
        f"SOX32 DE enhancer — {OBJ_LABEL}\nmotif attribution traces",
        fontsize=13, fontweight="bold",
    )
    ax_bot.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    out = FIGURES / "de_specificity_line_traces.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Stacked bar (absolute + proportional side by side)
# ---------------------------------------------------------------------------
def plot_stacked_bars(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sub = df.copy()
    sub["motif_group"] = sub["name"].where(
        sub["name"].isin(TOP_BAR_MOTIFS), "Other"
    )
    agg = (
        sub.groupby(["step", "motif_group"])["seqlet_attr"]
        .sum()
        .unstack(fill_value=0)
    )
    cols = [m for m in TOP_BAR_MOTIFS if m in agg.columns]
    if "Other" in agg.columns:
        cols.append("Other")
    agg = agg[cols]
    colors = [MOTIF_COLORS.get(c, OTHER_COLOR) for c in cols]

    # Absolute
    agg.plot.bar(stacked=True, ax=ax1, width=0.8,
                 color=colors, edgecolor="none", legend=False)
    ax1.set_xlabel("Optimization step", fontsize=10)
    ax1.set_ylabel("Total seqlet attribution", fontsize=10)
    ax1.set_title("Absolute", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(
        ["WT" if i == 0 else str(i) for i in range(21)],
        rotation=0, fontsize=7,
    )

    # Proportional
    row_totals = agg.sum(axis=1)
    agg_prop = agg.div(row_totals, axis=0).fillna(0)
    agg_prop.plot.bar(stacked=True, ax=ax2, width=0.8,
                      color=colors, edgecolor="none", legend=False)
    ax2.set_xlabel("Optimization step", fontsize=10)
    ax2.set_ylabel("Proportion of total attribution", fontsize=10)
    ax2.set_title("Proportional", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(
        ["WT" if i == 0 else str(i) for i in range(21)],
        rotation=0, fontsize=7,
    )
    ax2.set_ylim(0, 1)

    # Legend
    handles = [mpatches.Patch(color=MOTIF_COLORS.get(m, OTHER_COLOR),
                              label=MOTIF_DISPLAY.get(m, m))
               for m in TOP_BAR_MOTIFS]
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(6, len(handles)), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"SOX32 DE enhancer — {OBJ_LABEL}\nattribution by motif",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = FIGURES / "de_specificity_stacked_bars.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Static logos at selected steps
# ---------------------------------------------------------------------------
def plot_static_logos(d, bed: pd.DataFrame) -> None:
    n_steps = len(SELECTED_STEPS)
    xlim = (LOGO_START - ENHANCER_START - 1, LOGO_END - ENHANCER_START + 1)

    height_ratios = []
    for _ in range(n_steps):
        height_ratios.extend([4, 1])

    fig, all_axes = plt.subplots(
        n_steps * 2, 1, figsize=(14, n_steps * 3.2),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
    )

    # Global y-limits
    ymax = 0
    ymin = 0
    for step in SELECTED_STEPS:
        sid = "SOX32_WT" if step == 0 else f"SOX32_{OBJ}_step{step:02d}"
        ldf = get_logo_df(d, sid)
        ymax = max(ymax, ldf.values.max())
        ymin = min(ymin, ldf.values.min())
    ylim = (ymin * 1.1, ymax * 1.1)

    for si, step in enumerate(SELECTED_STEPS):
        ax_logo = all_axes[si * 2]
        ax_track = all_axes[si * 2 + 1]

        sid = "SOX32_WT" if step == 0 else f"SOX32_{OBJ}_step{step:02d}"
        ldf = get_logo_df(d, sid)
        de_pred = float(d[f"pred__{sid}"])

        logomaker.Logo(
            ldf, ax=ax_logo, shade_below=0.5, fade_below=0.5,
            color_scheme="classic",
        )
        ax_logo.set_ylim(*ylim)
        ax_logo.set_xlim(*xlim)
        ax_logo.set_xticklabels([])
        ax_logo.tick_params(axis="y", labelsize=7)

        step_label = "WT" if step == 0 else f"Step {step}"
        ax_logo.set_ylabel(
            f"{step_label}\nDE={de_pred:.0f}",
            fontsize=10, fontweight="bold",
        )

        seqlets = get_seqlets_in_window(bed, step)
        draw_seqlet_track(ax_track, seqlets, xlim)

    # X-axis label on bottom
    all_axes[-1].set_xlabel("Enhancer position (bp)", fontsize=10)

    fig.suptitle(
        f"SOX32 DE enhancer — {OBJ_LABEL}\n"
        f"attribution logos (enhancer positions "
        f"{LOGO_START - ENHANCER_START}–{LOGO_END - ENHANCER_START} bp)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = FIGURES / "de_specificity_static_logos.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5b: WT vs Step 20 — PDF, per-row y-axis scaling
# ---------------------------------------------------------------------------
def plot_wt_vs_final_pdf(d, bed: pd.DataFrame) -> None:
    rows = [(0, "WT", (-1, 5)), (20, "Step 20", (-5, 25))]
    xlim = (LOGO_START - ENHANCER_START - 1, LOGO_END - ENHANCER_START + 1)

    height_ratios = [4, 1, 4, 1]
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )

    for i, (step, step_label, ylim) in enumerate(rows):
        ax_logo = axes[i * 2]
        ax_track = axes[i * 2 + 1]

        sid = "SOX32_WT" if step == 0 else f"SOX32_{OBJ}_step{step:02d}"
        ldf = get_logo_df(d, sid)
        de_pred = float(d[f"pred__{sid}"])

        logomaker.Logo(
            ldf, ax=ax_logo, shade_below=0.5, fade_below=0.5,
            color_scheme="classic",
        )
        ax_logo.set_ylim(*ylim)
        ax_logo.set_xlim(*xlim)
        ax_logo.set_xticklabels([])
        ax_logo.tick_params(axis="y", labelsize=7)
        ax_logo.set_ylabel(
            f"{step_label}\nDE={de_pred:.0f}",
            fontsize=10, fontweight="bold",
        )

        seqlets = get_seqlets_in_window(bed, step)
        draw_seqlet_track(ax_track, seqlets, xlim)

    axes[-1].set_xlabel("Enhancer position (bp)", fontsize=10)

    fig.suptitle(
        f"SOX32 DE enhancer — {OBJ_LABEL}\n"
        f"WT vs optimized (enhancer pos "
        f"{LOGO_START - ENHANCER_START}–{LOGO_END - ENHANCER_START} bp)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0.02, 0, 0.98, 0.97])
    out = FIGURES / "de_specificity_static_logos.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5: Animated GIF
# ---------------------------------------------------------------------------
def plot_animation(d, bed: pd.DataFrame) -> None:
    frames = []
    ylim_max = 0
    ylim_min = 0
    for step in range(21):
        sid = "SOX32_WT" if step == 0 else f"SOX32_{OBJ}_step{step:02d}"
        ldf = get_logo_df(d, sid)
        de_pred = float(d[f"pred__{sid}"])
        seqlets = get_seqlets_in_window(bed, step)
        frames.append((ldf, de_pred, seqlets, step))
        ylim_max = max(ylim_max, ldf.values.max())
        ylim_min = min(ylim_min, ldf.values.min())
    ylim = (ylim_min * 1.1, ylim_max * 1.1)
    xlim = (LOGO_START - ENHANCER_START - 1, LOGO_END - ENHANCER_START + 1)

    fig, (ax_logo, ax_track) = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.02},
    )

    def update(frame_idx):
        ax_logo.clear()
        ax_track.clear()
        ldf, de_pred, seqlets, step = frames[frame_idx]

        logomaker.Logo(
            ldf, ax=ax_logo, shade_below=0.5, fade_below=0.5,
            color_scheme="classic",
        )
        ax_logo.set_ylim(*ylim)
        ax_logo.set_xlim(*xlim)
        step_label = "WT" if step == 0 else f"Step {step}"
        ax_logo.set_title(
            f"{OBJ_LABEL} — {step_label}  (DE pred = {de_pred:.0f})",
            fontsize=13, fontweight="bold",
        )
        ax_logo.set_ylabel("Attribution", fontsize=10)
        ax_logo.tick_params(labelsize=7)
        ax_logo.set_xticklabels([])

        draw_seqlet_track(ax_track, seqlets, xlim)
        ax_track.set_xlabel("Enhancer position (bp)", fontsize=10)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=700, repeat=True)
    out = FIGURES / "de_specificity_logo_animation.gif"
    anim.save(str(out), writer=PillowWriter(fps=2))
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_bed()
    d = load_attr()
    print(f"Loaded {len(df)} seqlet records for {OBJ_LABEL}")

    plot_waterfall(df)
    plot_line_traces(df)
    plot_stacked_bars(df)
    plot_static_logos(d, df)
    plot_animation(d, df)

    print(f"\n[done] All DE specificity figures in {FIGURES}")


if __name__ == "__main__":
    main()
