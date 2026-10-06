#!/usr/bin/env python
"""Distance-curve figures from the de_synergy pairwise grid files (CPU, local).

Reads the per-pair .npz grids from results/de4/ and plots ΔJ vs gap for each orientation,
with ΔS as a horizontal reference line. One figure per pair.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORIENT_COLORS = {"FF": "#1f77b4", "FR": "#ff7f0e", "RF": "#2ca02c", "RR": "#d62728"}


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


def plot_pair_curve(npz_path, calls_row, out):
    z = np.load(npz_path)
    dJ = z["dJ"]
    gaps = z["gaps"]
    orients = z["orients"]
    dA, dB = float(z["dA"]), float(z["dB"])
    dS = dA + dB

    idA = calls_row["idA"]
    idB = calls_row["idB"]
    tfA = calls_row.get("tf_A", calls_row.get("tfA", idA))
    tfB = calls_row.get("tf_B", calls_row.get("tfB", idB))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for oi, orient in enumerate(orients):
        y = dJ[oi]
        mask = np.isfinite(y)
        if mask.any():
            ax.plot(gaps[mask], y[mask], lw=1.6, color=ORIENT_COLORS.get(orient, "#333333"),
                    label=orient, alpha=0.85)

    ax.axhline(dS, color="#999999", ls="--", lw=1.2, label=f"ΔS = ΔA+ΔB = {dS:.3f}")
    opt_orient = str(calls_row.get("opt_orient", ""))
    opt_gap = int(calls_row.get("opt_gap", -1))
    delta = float(calls_row.get("delta", np.nan))
    maxZ = float(calls_row.get("maxZ", np.nan))
    ax.axvline(opt_gap, color="#333333", ls=":", lw=1.0, alpha=0.5)
    ax.scatter([opt_gap], [dS + delta], marker="*", s=120, color="#333333", zorder=5)
    ax.text(opt_gap + 3, dS + delta, f"opt: {opt_orient}@{opt_gap}bp\nΔ={delta:+.3f}, Z={maxZ:.1f}",
            fontsize=9, va="center")

    ax.set_xlabel("inner-edge gap (bp)")
    ax.set_ylabel("ΔJ (log counts)")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title(f"{idA} : {tfA}  x  {idB} : {tfB}   —   DE\n"
                 f"ΔA={dA:.3f}, ΔB={dB:.3f}, ΔS={dS:.3f}",
                 loc="left", fontweight="bold", fontsize=11)
    ax.margins(x=0)
    fig.tight_layout()
    save(fig, out, f"fig_distance_{idA}_{idB}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", required=True,
                   help="directory with per-pair .npz grids (results/de4/)")
    p.add_argument("--calls", required=True,
                   help="calls_long.tsv or pair_arrangements.tsv")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    style()
    os.makedirs(args.out, exist_ok=True)

    calls = pd.read_csv(args.calls, sep="\t")

    for npz_path in sorted(glob.glob(os.path.join(args.grids, "*__DE.npz"))):
        base = os.path.basename(npz_path).replace("__DE.npz", "")
        parts = base.split("__")
        if len(parts) != 2:
            continue
        idA, idB = parts
        row = calls[(calls.idA == idA) & (calls.idB == idB)]
        if row.empty:
            print(f"  skipping {idA}x{idB} — not in calls TSV")
            continue
        r = row.iloc[0]
        print(f"distance curve: {idA} x {idB}")
        plot_pair_curve(npz_path, r, args.out)


if __name__ == "__main__":
    main()
