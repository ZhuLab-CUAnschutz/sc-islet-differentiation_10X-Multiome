#!/usr/bin/env python
"""fig4 dumbbell plot from MULTITASK model data (zero GPU).

Reads the existing mt_de4/summary__all.tsv (19 pairs × 22 heads) from the de_synergy run
and renders a ΔS → ΔJ dumbbell per cell type for each pair.

Units are raw multitask counts (not log counts), so absolute values differ from single-task.
Thresholds from mt_null_thresholds.txt. Calls use delta_over_null95 already computed in the
de_synergy analyze_mt_de4.py.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

C_ADD = "#6e6e6e"

HERE = os.path.dirname(os.path.abspath(__file__))


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def fig4_mt(summary, md, idA, idB, out, null_d95=None, null_z95=None):
    sub = summary[(summary.idA == idA) & (summary.idB == idB)].copy()
    if sub.empty:
        print(f"  no data for {idA} x {idB}")
        return
    # merge cell-type metadata
    sub = sub.merge(md[["cell_type", "color", "display_order"]], left_on="ct", right_on="cell_type", how="left")
    sub = sub.sort_values("display_order", ascending=False)

    colors = dict(zip(md.cell_type, md.color))

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    y = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.plot([r.dS, r.dJ_opt], [i, i], color=colors.get(r.ct, "#888888"),
                lw=2.4, alpha=0.85, zorder=2, solid_capstyle="round")
    ax.scatter(sub.dS, y, s=52, color="white", edgecolor=C_ADD, lw=1.6, zorder=3)
    ax.scatter(sub.dJ_opt, y, s=58,
               color=[colors.get(c, "#888888") for c in sub.ct],
               edgecolor="white", lw=1.0, zorder=4)

    handles = [
        Line2D([], [], ls="", marker="o", mfc="white", mec=C_ADD, mew=1.6, ms=8,
               label="additive expectation ($\\Delta$S)"),
        Line2D([], [], ls="", marker="o", mfc="#555555", mec="white", mew=1.0, ms=8,
               label="observed ($\\Delta$J), coloured by cell type"),
    ]

    ax.set_yticks(y)
    ax.set_yticklabels([r.ct for _, r in sub.iterrows()], fontsize=9)

    # annotate synergy delta
    for i, (_, r) in enumerate(sub.iterrows()):
        delta = r.dJ_opt - r.dS
        ax.text(max(sub.dJ_opt.max() * 1.03, sub.dJ_opt.max() + 5), i,
                f"{delta:+.1f}", fontsize=8.5, va="center",
                color="#333333" if delta > 0 else "#999999")
    ax.text(max(sub.dJ_opt.max() * 1.03, sub.dJ_opt.max() + 5), len(sub) - 0.15,
            "Δ", fontsize=9, va="bottom", color="#333333", fontweight="bold")

    ax.set_xlabel("Δ (multitask raw counts)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncols=2, fontsize=9.5)

    # get TF names from the data
    tfA = sub.iloc[0].get("tfA", idA) if "tfA" in sub.columns else idA
    tfB = sub.iloc[0].get("tfB", idB) if "tfB" in sub.columns else idB

    subtitle = "multitask model (22 heads, CREsted finetuned_v2)"
    if null_d95 is not None:
        subtitle += f"  |  null Δ p95 = {null_d95:.1f}"

    ax.set_title(f"{idA} : {tfA}  x  {idB} : {tfB}\n{subtitle}",
                 loc="left", fontweight="bold", fontsize=11, pad=16)
    fig.tight_layout()
    save(fig, out, f"fig4_mt_celltypes_{idA}_{idB}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mt_summary", required=True,
                   help="path to mt_de4/summary__all.tsv from de_synergy")
    p.add_argument("--mt_null_thresholds", default=None,
                   help="path to mt_null_thresholds.txt (optional)")
    p.add_argument("--ct_metadata", required=True)
    p.add_argument("--pairs", required=True,
                   help="pair_arrangements.tsv — which pairs to plot")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    style()
    os.makedirs(args.out, exist_ok=True)
    summary = pd.read_csv(args.mt_summary, sep="\t")
    md = pd.read_csv(args.ct_metadata, sep="\t")
    pairs = pd.read_csv(args.pairs, sep="\t")

    # parse null thresholds if available
    null_d95 = None
    if args.mt_null_thresholds and os.path.exists(args.mt_null_thresholds):
        for line in open(args.mt_null_thresholds):
            if "delta p95" in line:
                null_d95 = float(line.split("p95=")[1].strip())
                break

    # generate one fig4 per pair
    for idA, idB in pairs[["idA", "idB"]].drop_duplicates().values:
        print(f"fig4_mt: {idA} x {idB}")
        fig4_mt(summary, md, idA, idB, args.out, null_d95=null_d95)


if __name__ == "__main__":
    main()
