#!/usr/bin/env python
"""Presentation figures from the marginalization tracks. CPU only, runs on the Mac.

fig1  single-TF marginalization: background track over motif-inserted track
fig2  cell-type specificity: the same insert across all 22 models, shared backgrounds
fig3  pairwise synergy, four tracks: A alone / B alone / additive expectation / observed
fig3b the same comparison in log-enrichment space, where additivity is literal addition
fig4  synergy across cell types: additive expectation vs observed, per model
"""
import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import tracks_io as T  # noqa: E402

# validated categorical set (scripts/validate_palette.js, light surface: all six checks pass)
C_A, C_B, C_JOINT = "#1f6fb4", "#c1492e", "#7b3fa0"
C_ADD, C_BG = "#6e6e6e", "#c9c9c9"

# Half-width of the plotted window, in bp either side of the insertion site. The model output is
# 1000 bp, so 500 is the maximum. Wide enough that the marginalization peak decays back toward
# background inside the frame -- at +/-150 the signal is still ~1.4x background at the edge, which
# makes the peak look like a plateau instead of a peak.
HALF = 450
PAIR_HALF = 300  # pair panels: narrower, so the motif footprints stay visible


def set_window(half, pair_half):
    global HALF, PAIR_HALF
    HALF, PAIR_HALF = half, pair_half


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
    })


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def xs(half=None):
    half = HALF if half is None else half
    return np.arange(-half, half)


def draw_track(ax, y, color, label=None, bg=None, half=None):
    """One filled prediction track, optionally over a background reference."""
    half = HALF if half is None else half
    x = xs(half)
    if bg is not None:
        ax.fill_between(x, T.window(bg, half), 0, color=C_BG, alpha=0.55, lw=0, zorder=1,
                        label="background")
    yw = T.window(y, half)
    ax.fill_between(x, yw, 0, color=color, alpha=0.25, lw=0, zorder=2)
    ax.plot(x, yw, color=color, lw=1.6, zorder=3, label=label)
    ax.margins(x=0)


def motif_band(ax, start, width, color, label, row=0):
    """Footprint of an inserted motif, in output-window coordinates relative to center.

    `row` staggers the label vertically so two adjacent inserts (gap 0) do not overprint.
    """
    lo = start - 557 - T.CENTER  # input coord -> output index -> centered
    ax.add_patch(Rectangle((lo, 0), width, 1, transform=ax.get_xaxis_transform(),
                           color=color, alpha=0.13, lw=0, zorder=0))
    ax.text(lo + width / 2, 0.97 - 0.17 * row, label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=color, fontweight="bold")


# ---------------- fig 1 ----------------
def fig1(cond, st, tf, cat, w, ct, out):
    bg, mot = cond["bg"], cond[f"single__{st}"]
    d = T.dlog(mot, bg)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, sharey=True)
    draw_track(axes[0], T.track(bg), C_BG)
    axes[0].plot(xs(), T.window(T.track(bg), HALF), color="#555555", lw=1.6)
    axes[0].set_title(f"{st} : {tf}  —  {ct}   ({cat})", loc="left", fontweight="bold")
    axes[0].set_ylabel("background")
    axes[1].set_ylabel(f"+ {tf} motif")
    draw_track(axes[1], T.track(mot), C_A)
    motif_band(axes[1], 1057 - w // 2, w, C_A, tf)
    for ax, c in zip(axes, [bg, mot]):
        ax.text(0.985, 0.9, f"log counts {c['c'].mean():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#444444")
    axes[1].text(0.985, 0.70, f"$\\Delta$ = {d:+.2f}", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=12, color=C_A, fontweight="bold")
    axes[1].set_xlabel("position relative to insertion site (bp)")
    fig.supylabel("predicted accessibility", fontsize=11)
    fig.tight_layout()
    save(fig, out, f"fig1_single_{st}_{tf}_{ct}")


# ---------------- fig 2 ----------------
def fig2(per_ct, md, st, tf, cat, out):
    order = [c for c in md.sort_values("display_order")["cell_type"] if c in per_ct]
    fig, axes = plt.subplots(len(order), 2, figsize=(9.5, 0.42 * len(order) + 1.4),
                             sharex="col", gridspec_kw={"width_ratios": [3, 1], "hspace": 0.0,
                                                        "wspace": 0.06})
    colors = dict(zip(md.cell_type, md.color))
    lr = {ct: T.log_ratio_track(per_ct[ct][f"single__{st}"], per_ct[ct]["bg"]) for ct in order}
    dd = {ct: T.dlog(per_ct[ct][f"single__{st}"], per_ct[ct]["bg"]) for ct in order}
    hi = max(T.window(v, HALF).max() for v in lr.values())
    for i, ct in enumerate(order):
        ax, axb = axes[i, 0], axes[i, 1]
        y = T.window(lr[ct], HALF)
        ax.fill_between(xs(), y, 0, color=colors[ct], alpha=0.85, lw=0)
        ax.set_ylim(0, hi * 1.05)
        ax.set_yticks([])
        ax.set_ylabel(ct, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(i == len(order) - 1 and s == "bottom")
        axb.barh([0], [dd[ct]], color=colors[ct], height=1.4)
        axb.set_xlim(min(0, min(dd.values()) * 1.6), max(dd.values()) * 1.28)
        axb.set_ylim(-1, 1)
        axb.set_yticks([])
        axb.grid(False)
        axb.text(dd[ct], 0, f" {dd[ct]:.2f}", va="center", fontsize=8.5, color="#333333")
        for s in ("left", "right", "top"):
            axb.spines[s].set_visible(False)
        axb.spines["bottom"].set_visible(i == len(order) - 1)
        if i < len(order) - 1:
            axb.set_xticks([])
    axes[-1, 0].set_xlabel("position relative to insertion site (bp)")
    axes[-1, 1].set_xlabel("$\\Delta$ log counts")
    fig.subplots_adjust(top=0.955)   # 22 rows: the default top margin leaves a large dead band
    fig.suptitle(f"{st} : {tf} inserted into a shared background set   ({cat})",
                 x=0.02, ha="left", fontweight="bold", fontsize=12, y=0.995)
    axes[0, 1].set_title("effect size", loc="center", fontsize=9.5, color="#555555", pad=3)
    save(fig, out, f"fig2_specificity_{st}_{tf}")


# ---------------- fig 3 ----------------
def _pair_bits(cond, r):
    key = f"pair__{r.idA}__{r.idB}"
    bg = cond["bg"]
    cA, cB, cJ = cond[f"{key}__A"], cond[f"{key}__B"], cond[f"{key}__AB"]
    m = cond[key]["meta"]
    dA, dB, dJ = T.dlog(cA, bg), T.dlog(cB, bg), T.dlog(cJ, bg)
    return bg, cA, cB, cJ, m, dA, dB, dJ


def fig3(cond, r, out):
    bg, cA, cB, cJ, m, dA, dB, dJ = _pair_bits(cond, r)
    Ta, Tb = T.track(cA), T.track(cB)
    Tadd = T.additive_track(cA, cB, bg)
    Tj, Tbg = T.track(cJ), T.track(bg)

    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(4, 2, width_ratios=[3.4, 1], hspace=0.12, wspace=0.22)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    for a in axes[1:]:
        a.sharex(axes[0])
        a.sharey(axes[0])

    rows = [(Ta, C_A, f"{r.tf_A}\nalone", dA),
            (Tb, C_B, f"{r.tf_B}\nalone", dB),
            (Tadd, C_ADD, "additive\nexpectation", dA + dB),
            (Tj, C_JOINT, f"{r.tf_A} + {r.tf_B}\nobserved", dJ)]
    for ax, (y, c, lab, d) in zip(axes, rows):
        draw_track(ax, y, c, bg=Tbg, half=PAIR_HALF)
        ax.set_ylabel(lab, fontsize=10)
        ax.text(0.985, 0.9, f"$\\Delta$ = {d:+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=c, fontweight="bold")
        if lab.startswith("additive"):
            ax.plot(xs(PAIR_HALF), T.window(y, PAIR_HALF), color=c, lw=1.6, ls="--")
    motif_band(axes[0], m["sA"], m["wa"], C_A, r.tf_A)
    motif_band(axes[1], m["sB"], m["wb"], C_B, r.tf_B)
    motif_band(axes[3], m["sA"], m["wa"], C_A, r.tf_A)      # both inserts on the observed panel
    motif_band(axes[3], m["sB"], m["wb"], C_B, r.tf_B, row=1)
    for ax in axes[:3]:
        ax.tick_params(labelbottom=False)
    axes[3].set_xlabel("position relative to construct center (bp)")
    axes[0].set_title(
        f"{r.idA} : {r.tf_A}  x  {r.idB} : {r.tf_B}   —   {r.ct}\n"
        f"optimal arrangement {m['orient']}, {m['gap']} bp gap "
        f"(centre-to-centre {m['center_center']:.1f} bp);  published call: {r.call}",
        loc="left", fontweight="bold", fontsize=11)

    axb = fig.add_subplot(gs[1:3, 1])
    vals = [dA, dB, dA + dB, dJ]
    axb.bar(range(4), vals, color=[C_A, C_B, C_ADD, C_JOINT], width=0.68)
    axb.set_xticks(range(4))
    axb.set_xticklabels(["$\\Delta$A", "$\\Delta$B", "$\\Delta$S", "$\\Delta$J"], fontsize=10)
    axb.set_xlabel("$\\Delta$S = additive\n$\\Delta$J = observed", fontsize=9, color="#555555")
    axb.set_ylabel("$\\Delta$ log counts", fontsize=10)
    top = max(vals) * 1.02
    axb.annotate("", xy=(3, dJ), xytext=(3, dA + dB),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    axb.text(3.42, (dJ + dA + dB) / 2, f"synergy\n{dJ - dA - dB:+.2f}", fontsize=9.5,
             color="#333333", va="center", fontweight="bold")
    axb.set_ylim(min(0, min(vals) * 1.1), top * 1.18)
    axb.grid(axis="x", visible=False)
    save(fig, out, f"fig3_pair_{r.idA}_{r.idB}_{r.ct}")


def fig3b(cond, r, out):
    bg, cA, cB, cJ, m, dA, dB, dJ = _pair_bits(cond, r)
    LA, LB = T.log_ratio_track(cA, bg), T.log_ratio_track(cB, bg)
    LJ = T.log_ratio_track(cJ, bg)
    Ladd = LA + LB
    x = xs(PAIR_HALF)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axhline(0, color="#999999", lw=0.8)
    ax.plot(x, T.window(LA, PAIR_HALF), color=C_A, lw=1.5, label=f"{r.tf_A} alone")
    ax.plot(x, T.window(LB, PAIR_HALF), color=C_B, lw=1.5, label=f"{r.tf_B} alone")
    ax.plot(x, T.window(Ladd, PAIR_HALF), color=C_ADD, lw=1.6, ls="--", label="additive (A + B)")
    ax.plot(x, T.window(LJ, PAIR_HALF), color=C_JOINT, lw=2.0, label="observed together")
    ax.fill_between(x, T.window(Ladd, PAIR_HALF), T.window(LJ, PAIR_HALF),
                    color=C_JOINT, alpha=0.16, lw=0, label="synergy")
    ax.set_xlabel("position relative to construct center (bp)")
    ax.set_ylabel("log enrichment over background")
    ax.margins(x=0)
    ax.legend(ncols=3, fontsize=9.5, loc="upper left")
    ax.set_title(f"{r.tf_A} x {r.tf_B} — {r.ct}   (additivity is literal addition in this space;"
                 f" total synergy {dJ - dA - dB:+.2f})", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, out, f"fig3b_logspace_{r.idA}_{r.idB}_{r.ct}")


# ---------------- fig 4 ----------------
def fig4(pairs, md, idA, idB, out):
    sub = pairs[(pairs.idA == idA) & (pairs.idB == idB)].copy()
    sub = sub.sort_values("display_order", ascending=False)
    colors = dict(zip(md.cell_type, md.color))
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    y = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.plot([r.dS, r.dJ_opt], [i, i], color=colors[r.ct], lw=2.4, alpha=0.85, zorder=2,
                solid_capstyle="round")
    ax.scatter(sub.dS, y, s=52, color="white", edgecolor=C_ADD, lw=1.6, zorder=3)
    ax.scatter(sub.dJ_opt, y, s=58, color=[colors[c] for c in sub.ct], edgecolor="white", lw=1.0,
               zorder=4)
    # neutral legend handles: the filled marker takes its colour from the cell type, so a
    # per-point colour in the legend key would imply a meaning it does not have
    handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=C_ADD, mew=1.6, ms=8,
                      label="additive expectation ($\\Delta$S)"),
               Line2D([], [], ls="", marker="o", mfc="#555555", mec="white", mew=1.0, ms=8,
                      label="observed together ($\\Delta$J), coloured by cell type")]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.ct}  {'●' if r.call == 'hard' else ('○' if r.call == 'soft' else '')}"
                        for _, r in sub.iterrows()], fontsize=9)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(max(sub.dJ_opt) * 1.03, i, f"{r.delta:+.2f}", fontsize=8.5, va="center",
                color="#333333" if r.call != "none" else "#999999")
    ax.text(max(sub.dJ_opt) * 1.03, len(sub) - 0.15, "synergy $\\Delta$", fontsize=9,
            va="bottom", color="#333333", fontweight="bold")
    ax.set_xlabel("$\\Delta$ log counts vs background")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncols=2, fontsize=9.5)
    tfA, tfB = sub.tf_A.iloc[0], sub.tf_B.iloc[0]
    n_hard = int((sub.call == "hard").sum())
    n_soft = int((sub.call == "soft").sum())
    ax.set_title(f"{idA} : {tfA}  x  {idB} : {tfB}\n"
                 f"synergistic in {n_hard + n_soft} of {len(sub)} cell types "
                 f"({n_hard} hard ●, {n_soft} soft ○; BH q < 0.05)",
                 loc="left", fontweight="bold", fontsize=11, pad=16)
    fig.tight_layout()
    save(fig, out, f"fig4_celltypes_{idA}_{idB}")


def main(a):
    style()
    os.makedirs(a.out, exist_ok=True)
    pairs = pd.read_csv(a.pairs, sep="\t")
    singles = pd.read_csv(a.singles, sep="\t")
    md = pd.read_csv(a.ct_metadata, sep="\t")

    own, shared = {}, {}
    for f in sorted(glob.glob(os.path.join(a.tracks, "tracks__*.npz"))):
        cond, meta = T.load(f)
        own[meta["ct"]] = cond["own"]
        if "shared" in cond:
            shared[meta["ct"]] = cond["shared"]
    print(f"loaded {len(own)} cell types\n")

    print("fig1 + fig2 (single TF)")
    for _, s in singles.iterrows():
        best = max(own, key=lambda ct: T.dlog(own[ct][f"single__{s.short_id}"], own[ct]["bg"]))
        fig1(own[best], s.short_id, s.curator_tf, s.curator_category, int(s.width), best, a.out)
        if shared:
            fig2(shared, md, s.short_id, s.curator_tf, s.curator_category, a.out)

    print("\nfig3 + fig3b + fig4 (pairs)")
    for (idA, idB), grp in pairs.groupby(["idA", "idB"], sort=False):
        r = grp.loc[grp.delta.idxmax()] if a.headline == "max_delta" \
            else grp.loc[grp.q_delta.idxmin()]
        print(f"  {idA}x{idB}: headline cell type {r.ct}")
        fig3(own[r.ct], r, a.out)
        fig3b(own[r.ct], r, a.out)
        fig4(pairs, md, idA, idB, a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--singles", required=True)
    p.add_argument("--ct_metadata", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--headline", default="max_delta", choices=["max_delta", "min_q"])
    p.add_argument("--half", type=int, default=450,
                   help="bp either side of the insertion site for the single-TF panels (max 500)")
    p.add_argument("--pair-half", dest="pair_half", type=int, default=300,
                   help="bp either side for the pair panels; narrower keeps the footprints visible")
    args = p.parse_args()
    assert 0 < args.half <= 500 and 0 < args.pair_half <= 500, "window half-widths must be <= 500"
    set_window(args.half, args.pair_half)
    main(args)
