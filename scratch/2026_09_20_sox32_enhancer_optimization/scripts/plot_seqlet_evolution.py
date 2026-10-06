#!/usr/bin/env python
"""
plot_seqlet_evolution.py
========================
Prototype visualizations showing how TF binding sites evolve across
the greedy optimization steps of the SOX32 DE enhancer.

Figures
-------
1. Waterfall map — seqlet footprints across the enhancer (x) vs. step (y),
   colored by motif, intensity by attribution strength. Per-step normalization
   so WT and late steps are comparably visible.
2. Line traces — curated biologically relevant motifs, distinct colors per motif.
3. Stacked bar — proportional attribution by motif (top N individual motifs,
   not lumped families).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BED = ROOT / "beds" / "seqlet_annotations.bed"
CATALOG_TSV = ROOT.parent / "2026_08_17_tf_hits" / "motif_catalog.tsv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

ENHANCER_START = 507
ENHANCER_END = 1607

OBJECTIVES_ORDERED = ["de_activity", "crested_default", "de_specificity"]
OBJ_LABELS = {
    "de_activity": "DE activity",
    "crested_default": "CREsted default",
    "de_specificity": "DE specificity",
}

# ---------------------------------------------------------------------------
# Per-motif colors — distinct color per biologically relevant motif,
# muted gray palette for the rest
# ---------------------------------------------------------------------------
MOTIF_COLORS = {
    "ST18":             "#e41a1c",   # T-box (EOMES)
    "ST47":             "#fb9a99",   # T-box lighter
    "ST36_sub3_LMX1A":  "#ff7f00",   # LMX1A
    "ST36_sub2_ISL1":   "#fdbf6f",   # ISL1
    "FOXH1_JASPAR":     "#984ea3",   # FOXH1
    "ST27":             "#4daf4a",   # GATA4/6
    "ST19":             "#377eb8",   # SOX17
    "ST55":             "#a65628",   # FOX family
    "ST37":             "#e78ac3",   # FOX family 2
    "ST46":             "#f781bf",   # HNF/FOXA
}

MOTIF_DISPLAY = {
    "ST18":             "EOMES (ST18)",
    "ST47":             "T-box (ST47)",
    "ST36_sub3_LMX1A":  "LMX1A (ST36.3)",
    "ST36_sub2_ISL1":   "ISL1 (ST36.2)",
    "FOXH1_JASPAR":     "FOXH1",
    "ST27":             "GATA4/6 (ST27)",
    "ST19":             "SOX17 (ST19)",
    "ST55":             "FOX (ST55)",
    "ST37":             "FOX (ST37)",
    "ST46":             "HNF/FOXA (ST46)",
}

OTHER_COLOR = "#bdbdbd"


def motif_color(label: str) -> str:
    return MOTIF_COLORS.get(label, OTHER_COLOR)


def motif_display(label: str) -> str:
    return MOTIF_DISPLAY.get(label, label)


# Families for waterfall legend grouping
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
    "Other":     OTHER_COLOR,
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_bed() -> pd.DataFrame:
    df = pd.read_csv(str(BED), sep="\t")
    wt = df[df["objective"] == "wild_type"].copy()
    rows = [df[df["objective"] != "wild_type"]]
    for obj in OBJECTIVES_ORDERED:
        wt_copy = wt.copy()
        wt_copy["objective"] = obj
        rows.append(wt_copy)
    df = pd.concat(rows, ignore_index=True)
    df["color"] = df["name"].map(motif_color)
    df["family"] = df["name"].map(lambda x: MOTIF_TO_FAMILY.get(x, "Other"))
    return df


# ---------------------------------------------------------------------------
# Figure 1: Waterfall map — per-step normalization
# ---------------------------------------------------------------------------
def plot_waterfall(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)

    for ax, obj in zip(axes, OBJECTIVES_ORDERED):
        sub = df[df["objective"] == obj].copy()

        # Per-step normalization so WT and late steps are comparably visible
        step_max = sub.groupby("step")["seqlet_attr"].apply(
            lambda s: s.abs().quantile(0.95) if len(s) > 0 else 1.0
        ).to_dict()

        for _, row in sub.iterrows():
            x = row["start"] - ENHANCER_START
            w = row["end"] - row["start"]
            y = row["step"]
            norm = step_max.get(y, 1.0)
            if norm == 0:
                norm = 1.0
            alpha = min(1.0, max(0.2, abs(row["seqlet_attr"]) / norm))
            base_color = row["color"]
            fc = to_rgba(base_color, alpha=alpha)
            ax.add_patch(plt.Rectangle(
                (x, y - 0.4), w, 0.8,
                facecolor=fc, edgecolor=base_color, linewidth=0.4,
            ))

        ax.set_xlim(0, ENHANCER_END - ENHANCER_START)
        ax.set_ylim(-0.5, 20.5)
        ax.set_xlabel("Enhancer position (bp)", fontsize=10)
        ax.set_title(OBJ_LABELS[obj], fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.set_yticks(range(0, 21))
        ax.set_yticklabels(
            ["WT" if s == 0 else str(s) for s in range(21)], fontsize=8
        )

    axes[0].set_ylabel("Optimization step", fontsize=11)

    # Legend by family
    handles = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f)
        for f in FAMILY_COLORS if f != "Other"
    ]
    handles.append(mpatches.Patch(color=FAMILY_COLORS["Other"], label="Other"))
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "SOX32 DE enhancer — seqlet footprints across optimization",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIGURES / "waterfall_seqlets.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Line traces — curated motif list, distinct colors
# ---------------------------------------------------------------------------
CURATED_MOTIFS = [
    "ST18",             # EOMES / T-box
    "ST47",             # T-box
    "ST36_sub3_LMX1A",  # LMX1A
    "ST36_sub2_ISL1",   # ISL1
    "FOXH1_JASPAR",     # FOXH1
    "ST27",             # GATA4/6
    "ST19",             # SOX17
    "ST55",             # FOX
    "ST46",             # HNF/FOXA
]


def plot_line_traces(df: pd.DataFrame) -> None:
    # Aggregate: total attribution per motif per (objective, step)
    agg = (
        df.groupby(["objective", "step", "name"])["seqlet_attr"]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, obj in zip(axes, OBJECTIVES_ORDERED):
        sub = agg[agg["objective"] == obj]

        for motif in CURATED_MOTIFS:
            m = sub[sub["name"] == motif].sort_values("step")
            if len(m) == 0:
                continue
            # Fill missing steps with 0
            full_steps = pd.DataFrame({"step": range(0, 21)})
            m = full_steps.merge(m, on="step", how="left").fillna(0)
            ax.plot(
                m["step"], m["seqlet_attr"], "o-",
                color=MOTIF_COLORS[motif],
                label=MOTIF_DISPLAY[motif],
                markersize=3, linewidth=1.8,
            )

        ax.set_xlabel("Optimization step", fontsize=10)
        ax.set_title(OBJ_LABELS[obj], fontsize=13, fontweight="bold")
        ax.set_xlim(-0.5, 20.5)
        ax.set_xticks(range(0, 21, 2))
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[0].set_ylabel("Total seqlet attribution", fontsize=11)

    # Single legend outside right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right",
        fontsize=8, framealpha=0.9, bbox_to_anchor=(1.01, 0.5),
    )

    fig.suptitle(
        "SOX32 DE enhancer — motif attribution traces across optimization",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    out = FIGURES / "line_traces_seqlets.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Stacked bar — top individual motifs (not lumped families)
# ---------------------------------------------------------------------------
TOP_BAR_MOTIFS = [
    "ST18", "ST47", "ST36_sub3_LMX1A", "ST36_sub2_ISL1",
    "FOXH1_JASPAR", "ST27", "ST19", "ST55", "ST46", "ST37",
]


def plot_stacked_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, obj in zip(axes, OBJECTIVES_ORDERED):
        sub = df[df["objective"] == obj].copy()
        sub["motif_group"] = sub["name"].where(
            sub["name"].isin(TOP_BAR_MOTIFS), "Other"
        )

        agg = (
            sub.groupby(["step", "motif_group"])["seqlet_attr"]
            .sum()
            .unstack(fill_value=0)
        )

        # Order: curated motifs first, then Other
        cols = [m for m in TOP_BAR_MOTIFS if m in agg.columns]
        if "Other" in agg.columns:
            cols.append("Other")
        agg = agg[cols]

        colors = [MOTIF_COLORS.get(c, OTHER_COLOR) for c in cols]
        agg.plot.bar(
            stacked=True, ax=ax, width=0.8,
            color=colors, edgecolor="none", legend=False,
        )

        ax.set_xlabel("Optimization step", fontsize=10)
        ax.set_title(OBJ_LABELS[obj], fontsize=13, fontweight="bold")
        ax.set_xticklabels(
            ["WT" if i == 0 else str(i) for i in range(21)],
            rotation=0, fontsize=7,
        )

    axes[0].set_ylabel("Total seqlet attribution", fontsize=11)

    # Legend with display names
    handles = []
    for m in TOP_BAR_MOTIFS:
        handles.append(mpatches.Patch(
            color=MOTIF_COLORS[m], label=MOTIF_DISPLAY[m],
        ))
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other"))
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "SOX32 DE enhancer — attribution by motif across optimization",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = FIGURES / "stacked_bar_motifs.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Stacked bar — proportional (fraction of total attribution)
# ---------------------------------------------------------------------------
def plot_stacked_bars_proportional(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, obj in zip(axes, OBJECTIVES_ORDERED):
        sub = df[df["objective"] == obj].copy()
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

        # Normalize to proportions
        row_totals = agg.sum(axis=1)
        agg_prop = agg.div(row_totals, axis=0).fillna(0)

        colors = [MOTIF_COLORS.get(c, OTHER_COLOR) for c in cols]
        agg_prop.plot.bar(
            stacked=True, ax=ax, width=0.8,
            color=colors, edgecolor="none", legend=False,
        )

        ax.set_xlabel("Optimization step", fontsize=10)
        ax.set_title(OBJ_LABELS[obj], fontsize=13, fontweight="bold")
        ax.set_xticklabels(
            ["WT" if i == 0 else str(i) for i in range(21)],
            rotation=0, fontsize=7,
        )
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Proportion of total attribution", fontsize=11)

    handles = []
    for m in TOP_BAR_MOTIFS:
        handles.append(mpatches.Patch(
            color=MOTIF_COLORS[m], label=MOTIF_DISPLAY[m],
        ))
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other"))
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "SOX32 DE enhancer — attribution proportion by motif",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = FIGURES / "stacked_bar_proportional.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_bed()
    print(f"Loaded {len(df)} seqlet records")

    plot_waterfall(df)
    plot_line_traces(df)
    plot_stacked_bars(df)
    plot_stacked_bars_proportional(df)

    print("\n[done] All figures in", FIGURES)


if __name__ == "__main__":
    main()
