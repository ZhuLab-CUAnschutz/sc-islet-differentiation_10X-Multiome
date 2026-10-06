#!/usr/bin/env python
"""Presentation figures from the marginalization tracks. CPU only, runs on the Mac.

Extended from 2026_08_08_prez/scripts/plot_tracks.py to support:
  - 3-motif (triple) stacked-track figures (5 rows)
  - 4-motif (quad) stacked-track figures (6 rows)
  - common y-scale output (all figures on a shared y-limit)
  - pdf.fonttype=42 for Illustrator-editable text

fig1  single-TF marginalization: background track vs motif-inserted track
fig2  cell-type specificity: the same insert across all 22 models, shared backgrounds
fig3  pairwise synergy, four tracks: A alone / B alone / additive expectation / observed
fig3b the same comparison in log-enrichment space, where additivity is literal addition
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tracks_io as T  # noqa: E402

# validated categorical set
C_A, C_B, C_JOINT = "#1f6fb4", "#c1492e", "#7b3fa0"
C_C = "#2ca02c"  # third motif colour (green)
C_D = "#d4a017"  # fourth motif colour (gold)
C_ADD, C_BG = "#6e6e6e", "#c9c9c9"

HALF = 450
PAIR_HALF = 300


def set_window(half, pair_half):
    global HALF, PAIR_HALF
    HALF, PAIR_HALF = half, pair_half


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


def xs(half=None):
    half = HALF if half is None else half
    return np.arange(-half, half)


def draw_track(ax, y, color, label=None, bg=None, half=None):
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
    lo = start - 557 - T.CENTER
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
    fig.subplots_adjust(top=0.955)
    fig.suptitle(f"{st} : {tf} inserted into a shared background set   ({cat})",
                 x=0.02, ha="left", fontweight="bold", fontsize=12, y=0.995)
    axes[0, 1].set_title("effect size", loc="center", fontsize=9.5, color="#555555", pad=3)
    save(fig, out, f"fig2_specificity_{st}_{tf}")


# ---------------- fig 3 (pair) ----------------
def _pair_bits(cond, r):
    key = f"pair__{r.idA}__{r.idB}"
    bg = cond["bg"]
    cA, cB, cJ = cond[f"{key}__A"], cond[f"{key}__B"], cond[f"{key}__AB"]
    m = cond[key]["meta"]
    dA, dB, dJ = T.dlog(cA, bg), T.dlog(cB, bg), T.dlog(cJ, bg)
    return bg, cA, cB, cJ, m, dA, dB, dJ


def fig3(cond, r, out, common_ylim=None):
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
    if common_ylim is not None:
        axes[0].set_ylim(0, common_ylim)
    motif_band(axes[0], m["sA"], m["wa"], C_A, r.tf_A)
    motif_band(axes[1], m["sB"], m["wb"], C_B, r.tf_B)
    motif_band(axes[3], m["sA"], m["wa"], C_A, r.tf_A)
    motif_band(axes[3], m["sB"], m["wb"], C_B, r.tf_B, row=1)
    for ax in axes[:3]:
        ax.tick_params(labelbottom=False)
    axes[3].set_xlabel("position relative to construct center (bp)")
    axes[0].set_title(
        f"{r.idA} : {r.tf_A}  x  {r.idB} : {r.tf_B}   —   {r.ct}\n"
        f"optimal arrangement {m['orient']}, {m['gap']} bp gap "
        f"(centre-to-centre {m['center_center']:.1f} bp);  call: {r.call}",
        loc="left", fontweight="bold", fontsize=11)

    axb = fig.add_subplot(gs[1:3, 1])
    vals = [dA, dB, dA + dB, dJ]
    axb.bar(range(4), vals, color=[C_A, C_B, C_ADD, C_JOINT], width=0.68)
    axb.set_xticks(range(4))
    axb.set_xticklabels(["$\\Delta$A", "$\\Delta$B", "$\\Delta$S", "$\\Delta$J"], fontsize=10)
    axb.set_xlabel("$\\Delta$S = additive\n$\\Delta$J = observed", fontsize=9, color="#555555")
    axb.set_ylabel("$\\Delta$ log counts", fontsize=10)
    top = max(vals) * 1.02 if max(vals) > 0 else 0.1
    axb.annotate("", xy=(3, dJ), xytext=(3, dA + dB),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    axb.text(3.42, (dJ + dA + dB) / 2, f"synergy\n{dJ - dA - dB:+.2f}", fontsize=9.5,
             color="#333333", va="center", fontweight="bold")
    axb.set_ylim(min(0, min(vals) * 1.1), top * 1.18)
    axb.grid(axis="x", visible=False)
    save(fig, out, f"fig3_pair_{r.idA}_{r.idB}_{r.ct}")


# ---------------- fig 3 (triple) ----------------
def _triple_bits(cond, r):
    key = f"triple__{r.idA}__{r.idB}__{r.idC}"
    bg = cond["bg"]
    cA = cond[f"{key}__A"]
    cB = cond[f"{key}__B"]
    cC = cond[f"{key}__C"]
    cJ = cond[f"{key}__ABC"]
    m = cond[key]["meta"]
    dA = T.dlog(cA, bg)
    dB = T.dlog(cB, bg)
    dC = T.dlog(cC, bg)
    dJ = T.dlog(cJ, bg)
    return bg, cA, cB, cC, cJ, m, dA, dB, dC, dJ


def fig3_triple(cond, r, out, common_ylim=None):
    bg, cA, cB, cC, cJ, m, dA, dB, dC, dJ = _triple_bits(cond, r)
    Ta, Tb, Tc = T.track(cA), T.track(cB), T.track(cC)
    Tadd = T.additive_track_3way(cA, cB, cC, bg)
    Tj, Tbg = T.track(cJ), T.track(bg)

    fig = plt.figure(figsize=(10, 9.0))
    gs = fig.add_gridspec(5, 2, width_ratios=[3.4, 1], hspace=0.12, wspace=0.22)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    for a in axes[1:]:
        a.sharex(axes[0])
        a.sharey(axes[0])

    rows = [(Ta, C_A, f"{r.tfA}\nalone", dA),
            (Tb, C_B, f"{r.tfB}\nalone", dB),
            (Tc, C_C, f"{r.tfC}\nalone", dC),
            (Tadd, C_ADD, "additive\nexpectation", dA + dB + dC),
            (Tj, C_JOINT, f"{r.tfA}+{r.tfB}+{r.tfC}\nobserved", dJ)]
    for ax, (y, c, lab, d) in zip(axes, rows):
        draw_track(ax, y, c, bg=Tbg, half=PAIR_HALF)
        ax.set_ylabel(lab, fontsize=10)
        ax.text(0.985, 0.9, f"$\\Delta$ = {d:+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=c, fontweight="bold")
        if lab.startswith("additive"):
            ax.plot(xs(PAIR_HALF), T.window(y, PAIR_HALF), color=c, lw=1.6, ls="--")
    if common_ylim is not None:
        axes[0].set_ylim(0, common_ylim)

    motif_band(axes[0], m["sA"], m["wa"], C_A, r.tfA)
    motif_band(axes[1], m["sB"], m["wb"], C_B, r.tfB)
    motif_band(axes[2], m["sC"], m["wc"], C_C, r.tfC)
    motif_band(axes[4], m["sA"], m["wa"], C_A, r.tfA)
    motif_band(axes[4], m["sB"], m["wb"], C_B, r.tfB, row=1)
    motif_band(axes[4], m["sC"], m["wc"], C_C, r.tfC, row=2)
    for ax in axes[:4]:
        ax.tick_params(labelbottom=False)
    axes[4].set_xlabel("position relative to construct center (bp)")

    layout_str = m.get("opt_layout", "?")
    gap_C = m.get("opt_gap_C", "?")
    axes[0].set_title(
        f"{r.idA}:{r.tfA} x {r.idB}:{r.tfB} x {r.idC}:{r.tfC}  —  {r.ct}\n"
        f"pair {m['pair_orient']} gap {m['pair_gap']}bp;  C: {layout_str} gap {gap_C}bp",
        loc="left", fontweight="bold", fontsize=11)

    axb = fig.add_subplot(gs[1:4, 1])
    vals = [dA, dB, dC, dA + dB + dC, dJ]
    axb.bar(range(5), vals, color=[C_A, C_B, C_C, C_ADD, C_JOINT], width=0.68)
    axb.set_xticks(range(5))
    axb.set_xticklabels(["$\\Delta$A", "$\\Delta$B", "$\\Delta$C",
                         "$\\Delta$S", "$\\Delta$J"], fontsize=10)
    axb.set_xlabel("$\\Delta$S = additive\n$\\Delta$J = observed", fontsize=9, color="#555555")
    axb.set_ylabel("$\\Delta$ log counts", fontsize=10)
    top = max(vals) * 1.02 if max(vals) > 0 else 0.1
    axb.annotate("", xy=(4, dJ), xytext=(4, dA + dB + dC),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    axb.text(4.42, (dJ + dA + dB + dC) / 2, f"synergy\n{dJ - dA - dB - dC:+.2f}",
             fontsize=9.5, color="#333333", va="center", fontweight="bold")
    axb.set_ylim(min(0, min(vals) * 1.1), top * 1.18)
    axb.grid(axis="x", visible=False)
    save(fig, out, f"fig3_triple_{r.idA}_{r.idB}_{r.idC}_{r.ct}")


# ---------------- fig 3b (log-space) ----------------
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


# ================================================================
# MAIN
# ================================================================
def main(a):
    style()
    os.makedirs(a.out, exist_ok=True)
    if a.common_ylim_dir:
        os.makedirs(a.common_ylim_dir, exist_ok=True)
    pairs = pd.read_csv(a.pairs, sep="\t")
    singles = pd.read_csv(a.singles, sep="\t")
    triples = pd.read_csv(a.triples, sep="\t") if a.triples else None
    md = pd.read_csv(a.ct_metadata, sep="\t")
    id2tf = dict(zip(singles.short_id, singles.curator_tf))

    own, shared = {}, {}
    for f in sorted(glob.glob(os.path.join(a.tracks, "tracks__*.npz"))):
        cond, meta = T.load(f)
        own[meta["ct"]] = cond["own"]
        if "shared" in cond:
            shared[meta["ct"]] = cond["shared"]
    print(f"loaded {len(own)} cell types\n")

    # ── fig1 + fig2 (single TF) ──────────────────────────────────────────
    print("fig1 + fig2 (single TF)")
    for _, s in singles.iterrows():
        best = max(own, key=lambda ct: T.dlog(own[ct][f"single__{s.short_id}"], own[ct]["bg"]))
        fig1(own[best], s.short_id, s.curator_tf, s.curator_category, int(s.width), best, a.out)
        if shared:
            fig2(shared, md, s.short_id, s.curator_tf, s.curator_category, a.out)

    # ── fig3 + fig3b (pairs) ─────────────────────────────────────────────
    print("\nfig3 + fig3b (pairs)")
    # Collect max y for common-scale
    all_maxY = []
    for (idA, idB), grp in pairs.groupby(["idA", "idB"], sort=False):
        r = grp.loc[grp.delta.idxmax()]
        ct = r.ct
        if ct not in own:
            print(f"  skipping {idA}x{idB} — cell type {ct} not in tracks")
            continue
        print(f"  {idA}x{idB}: headline cell type {ct}")
        fig3(own[ct], r, a.out)
        fig3b(own[ct], r, a.out)

        # collect peak y for common scale
        key = f"pair__{idA}__{idB}"
        if key in own[ct]:
            cJ = own[ct][f"{key}__AB"]
            maxY = T.window(T.track(cJ), PAIR_HALF).max()
            all_maxY.append(maxY)

    # ── fig3 triples ───────────────────────────────────────────────────────
    if triples is not None:
        print("\nfig3 (triples)")
        for col in ["tfA", "tfB", "tfC"]:
            if col not in triples.columns:
                id_col = col.replace("tf", "id")
                triples[col] = triples[id_col].map(id2tf).fillna(triples[id_col])
        for _, r in triples.iterrows():
            ct = r.ct
            if ct not in own:
                print(f"  skipping {r.idA}x{r.idB}x{r.idC} — cell type {ct} not in tracks")
                continue
            key = f"triple__{r.idA}__{r.idB}__{r.idC}"
            if key not in own[ct]:
                print(f"  skipping {r.idA}x{r.idB}x{r.idC} — triple key not in tracks")
                continue
            print(f"  {r.idA}x{r.idB}x{r.idC} in {ct}")
            fig3_triple(own[ct], r, a.out)

    # ── common-scale pass ─────────────────────────────────────────────────
    if a.common_ylim_dir and all_maxY:
        common_ylim = max(all_maxY) * 1.05
        print(f"\ncommon-scale pass (ylim={common_ylim:.3f})")
        for (idA, idB), grp in pairs.groupby(["idA", "idB"], sort=False):
            r = grp.loc[grp.delta.idxmax()]
            ct = r.ct
            if ct not in own:
                continue
            fig3(own[ct], r, a.common_ylim_dir, common_ylim=common_ylim)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--triples", default=None,
                   help="best_triple_arrangements.tsv (optional, Phase 2)")
    p.add_argument("--singles", required=True)
    p.add_argument("--ct_metadata", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--headline", default="max_delta", choices=["max_delta", "min_q"])
    p.add_argument("--half", type=int, default=450)
    p.add_argument("--pair-half", dest="pair_half", type=int, default=300)
    p.add_argument("--common-ylim-dir", dest="common_ylim_dir", default=None,
                   help="directory for common-scale figures (optional)")
    args = p.parse_args()
    assert 0 < args.half <= 500 and 0 < args.pair_half <= 500
    set_window(args.half, args.pair_half)
    main(args)
