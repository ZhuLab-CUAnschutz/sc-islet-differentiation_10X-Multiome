#!/usr/bin/env python
"""
11_celltype_motif_summary.py — ONE composite figure for a cell type (default DE):

  A) barplot of predicted accessibility across all cell types (multitask)
  B) the cell type's contribution logos zoomed into motif regions, with a sequence
     schematic + pull-out connectors. Two modes:
       sites  : individual high-contribution FIMO sites; under each, the matched
                HOCOMOCO PWM aligned at the FIMO coordinates/strand (best alignment).
       spans  : nearby FIMO sites merged into wider composite regions; FIMO motifs
                marked inside each span.

  compare: multiple cell types (comma-list or "all") over the SAME FIMO regions,
           one shared contribution y-scale, labeled construct track + pull-out wedges.

    python 11_celltype_motif_summary.py <npz> <celltype|comma-list|all> <sites|spans|compare> \
           [hoco|compendium] [ref_celltype]
Run under eugene_tools (on SLURM — imports torch). Large/compare sets save PDF only.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from tangermeme.plot import plot_logo
from tangermeme.annotate import read_meme
import plot_locus as pl
import chrombpnet_locus as cl

NPZ = sys.argv[1] if len(sys.argv) > 1 else "outputs/sox32_multitask_contribs.npz"
CT = sys.argv[2] if len(sys.argv) > 2 else "DE"
MODE = sys.argv[3] if len(sys.argv) > 3 else "sites"
MATCHDB = sys.argv[4] if len(sys.argv) > 4 else "hoco"   # hoco | compendium (sites mode)
CT0 = "DE" if CT == "all" else CT.split(",")[0]          # primary cell type (compare = comma list / all)
REF = sys.argv[5] if len(sys.argv) > 5 else CT0          # select regions from REF; draw CT (shared scale)
COMPENDIUM_MEME = ("/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
                   "results/4_multi_task_model/crested/motifs_compendium_strain/combined_mapped.meme")
K = 6                  # sites mode: number of FIMO sites
PAD = 9                # flank around a site (sites mode)
SPAN_GAP = 35          # merge FIMO hits within this gap into one span
SPAN_PAD = 12
N_SPANS = 4

res = pl.LocusResults(NPZ)
cmap = cl.color_map()
order = res.ordered_celltypes()
disp = {v: k for k, v in config.FIMO_TF_NAMES.items()}      # FIMO id -> display TF
_HOCO = read_meme(config.HOCOMOCO_MEME)


def rc_pwm(pwm):
    return pwm[[3, 2, 1, 0]][:, ::-1]


def aligned_embed(pwm, win_len, contrib_abs):
    """Place the IC logo of `pwm` (trying both strands) at the offset within the
    window that best overlaps the contribution magnitude (cross-correlation) —
    a lightweight 'best alignment' that follows the contribution peak rather than
    the FIMO call. Returns (embedded (4,win_len), strand, offset)."""
    best = None
    for orient, p in (("+", pwm), ("-", rc_pwm(pwm))):
        ic = pl.ic_scale(np.asarray(p, dtype=float))
        if ic.shape[1] > win_len:                      # motif wider than window: central crop
            c = (ic.shape[1] - win_len) // 2
            ic = ic[:, c:c + win_len]
        icsum = ic.sum(0); w = ic.shape[1]
        for off in range(0, win_len - w + 1):
            corr = float((contrib_abs[off:off + w] * icsum).sum())
            if best is None or corr > best[0]:
                best = (corr, off, orient, ic)
    _, off, orient, ic = best
    emb = np.zeros((4, win_len), dtype="float32")
    emb[:, off:off + ic.shape[1]] = ic
    return emb, orient, off


_COMP = None
def matched_pwm(res_variant, r, win):
    """Return (pwm, label) of the DB motif matching this site.
    hoco -> the FIMO TF's HOCOMOCO PWM. compendium -> best TOMTOM match of the
    site's contribution-region sequence against the multitask motif compendium."""
    if MATCHDB == "compendium":
        global _COMP
        if _COMP is None:
            _COMP = read_meme(COMPENDIUM_MEME)
        seqlet = pd.DataFrame({"ex": [0], "start": [int(r["start"])], "end": [int(r["stop"])]})
        ann = pl.annotate(res, "construct", seqlet, motifs=COMPENDIUM_MEME)
        lab = ann.iloc[0]["label"]; pv = ann.iloc[0]["pvalue"]
        if lab is None or pv >= 0.05 or lab not in _COMP:
            return None, "(no match)"
        short = lab.replace("pos_", "").replace("cluster_final", "clust").split("__")[0]
        return np.asarray(_COMP[lab], dtype=float), f"{short} p={pv:.0e}"
    tf = disp.get(r["motif_id"], r["motif_id"])
    return np.asarray(_HOCO[config.NAMED_TF[tf]], dtype=float), config.NAMED_TF[tf].split(".")[0]


proj4 = res.attr("construct", CT0) * res.ohe("construct")           # drawn cell type (primary)
proj = proj4.sum(0)
proj4_ref = res.attr("construct", REF) * res.ohe("construct")        # region-selection reference
proj_ref = proj4_ref.sum(0)
Q_THRESH = float(os.environ.get("SOX32_QTHRESH", "0.05"))   # FIMO q cutoff for shown hits
f = pd.read_csv(config.FIMO_HITS_CSV)
f = f[(f["q-value"] < Q_THRESH) & (f["feature"] == "enhancer")].copy()
f["contrib"] = [proj[int(s) - 1:int(e)].sum() for s, e in zip(f["start"], f["stop"])]        # CT (shown)
f["contrib_ref"] = [proj_ref[int(s) - 1:int(e)].sum() for s, e in zip(f["start"], f["stop"])]  # REF (selects)


def barplot(ax, highlight=None, show_title=True):
    highlight = highlight or [CT]
    vals = [res.counts("construct", c) for c in order]
    bars = ax.bar(range(len(order)), vals, color=[cmap[c] for c in order], edgecolor="k", lw=0.4)
    for c in highlight:
        if c in order:
            bars[order.index(c)].set_edgecolor("crimson"); bars[order.index(c)].set_linewidth(2.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=8)
    for t, c in zip(ax.get_xticklabels(), order):
        t.set_color(cmap[c])
    ax.set_ylabel("multitask predicted\naccessibility")
    if show_title:
        ax.set_title(f"A)  Predicted accessibility across cell types (multitask) — {CT} highlighted",
                     fontsize=11, loc="left")


def schematic(ax, marks, lo, hi):
    ax.plot([lo, hi], [0, 0], color="0.3", lw=2)
    ax.set_xlim(lo, hi); ax.set_ylim(-1, 1); ax.axis("off")
    ax.text(hi, 0.2, " enhancer", fontsize=8, color="0.4", va="bottom")
    for mid, col in marks:
        ax.plot([mid], [0], marker="v", color=col, ms=7)


def make_sites():
    sites = f.nlargest(K, "contrib_ref").sort_values("start").reset_index(drop=True)
    lo = int(sites["start"].min()) - 40; hi = int(sites["stop"].max()) + 40
    fig = plt.figure(figsize=(14, 10.5))
    gs = GridSpec(4, len(sites), height_ratios=[2.2, 0.7, 1.7, 1.1], hspace=0.9, wspace=0.35)
    barplot(fig.add_subplot(gs[0, :]))
    ax_seq = fig.add_subplot(gs[1, :])
    vlim = np.abs(proj4_ref[:, lo:hi]).max() * 1.1     # shared scale (REF) across cell types
    marks = []
    for k, (_, r) in enumerate(sites.iterrows()):
        s, e = int(r["start"]), int(r["stop"]); mid = (s + e) / 2
        tf = disp.get(r["motif_id"], r["motif_id"]); col = config.TF_COLORS.get(tf, "crimson")
        win = (s - PAD, e + PAD); W = win[1] - win[0]
        ax_c = fig.add_subplot(gs[2, k]); ax_d = fig.add_subplot(gs[3, k])
        plot_logo(proj4[:, win[0]:win[1]], ax=ax_c, ylim=vlim)
        pwm, mlabel = matched_pwm(res, r, win)
        if pwm is not None:
            emb, orient, _ = aligned_embed(pwm, W, np.abs(proj[win[0]:win[1]]))
            plot_logo(emb, ax=ax_d)
            ax_d.set_xlabel(f"{mlabel} ({orient})", fontsize=7.5, color=col)
        else:
            ax_d.text(0.5, 0.5, mlabel, ha="center", va="center", fontsize=8, color="0.5")
        for a in (ax_c, ax_d):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(1.3)
        ax_c.set_title(f"{tf} @{s}{r['strand']}\ncontrib={r['contrib']:.1f}", fontsize=8.5, color=col)
        marks.append((mid, col))
        con = ConnectionPatch(xyA=(mid, -0.15), coordsA=ax_seq.transData, xyB=(0.5, 1.0),
                              coordsB=ax_c.transAxes, color=col, lw=0.8, alpha=0.7)
        fig.add_artist(con)
    schematic(ax_seq, marks, lo, hi)
    fig.text(0.015, 0.42, "contribution", rotation=90, fontsize=14, va="center")
    fig.text(0.015, 0.13, "matched motif", rotation=90, fontsize=12, va="center")
    fig.suptitle(f"sox32 enhancer — {CT}: accessibility + per-site contribution vs matched DB motif",
                 fontsize=13, y=0.98)
    return fig, "sites"


def merge_spans():
    g = f.sort_values("start")
    spans = []
    for _, r in g.iterrows():
        if spans and r["start"] - spans[-1]["end"] <= SPAN_GAP:
            spans[-1]["end"] = max(spans[-1]["end"], int(r["stop"]))
            spans[-1]["hits"].append(r)
        else:
            spans.append({"start": int(r["start"]), "end": int(r["stop"]), "hits": [r]})
    for sp in spans:
        sp["contrib"] = proj[sp["start"] - 1:sp["end"]].sum()           # CT (shown)
        # FIMO-based, cell-type-INDEPENDENT score: aggregate hit significance.
        # Spans are defined by the FIMO hits, so the SAME spans show for every cell type.
        sp["fimo_score"] = float(sum(-np.log10(max(h["q-value"], 1e-300)) for h in sp["hits"]))
        sp["motifs"] = {h["motif_id"] for h in sp["hits"]}
    ranked = sorted(spans, key=lambda x: -x["fimo_score"])
    sel = ranked[:N_SPANS]
    # guarantee every motif in the set is represented (e.g. a weak SMAD2 site)
    have = {m for sp in sel for m in sp["motifs"]}
    for m in {m for sp in spans for m in sp["motifs"]} - have:
        best = max((sp for sp in spans if m in sp["motifs"]), key=lambda x: x["fimo_score"])
        if best not in sel:
            sel.append(best)
    return sorted(sel, key=lambda x: x["start"])


def make_spans():
    spans = merge_spans()
    lo = min(s["start"] for s in spans) - 40; hi = max(s["end"] for s in spans) + 40
    fig = plt.figure(figsize=(14, 8.5))
    widths = [s["end"] - s["start"] + 2 * SPAN_PAD for s in spans]
    gs = GridSpec(3, len(spans), height_ratios=[2.2, 0.7, 2.6], width_ratios=widths,
                  hspace=0.9, wspace=0.25)
    barplot(fig.add_subplot(gs[0, :]))
    ax_seq = fig.add_subplot(gs[1, :])
    vlim = np.abs(proj4_ref[:, lo:hi]).max() * 1.1     # shared scale (REF) across cell types
    marks = []
    for k, sp in enumerate(spans):
        a = fig.add_subplot(gs[2, k]); w0, w1 = sp["start"] - SPAN_PAD, sp["end"] + SPAN_PAD
        plot_logo(proj4[:, w0:w1], ax=a, ylim=vlim); a.set_yticks([])
        xt = np.linspace(0, w1 - w0, 3); a.set_xticks(xt)
        a.set_xticklabels([str(int(w0 + t)) for t in xt], fontsize=7)
        ymax = a.get_ylim()[1]
        for r in sp["hits"]:                       # mark FIMO motifs inside the span
            tf = disp.get(r["motif_id"], r["motif_id"]); col = config.TF_COLORS.get(tf, "crimson")
            c = (int(r["start"]) + int(r["stop"])) / 2 - w0
            a.axvspan(int(r["start"]) - w0, int(r["stop"]) - w0, color=col, alpha=0.12, lw=0)
            a.text(c, ymax * 0.95, tf, ha="center", va="top", fontsize=7, color=col, rotation=90)
        a.set_title(f"{sp['start']}-{sp['end']}  (contrib {sp['contrib']:.1f})", fontsize=8.5)
        mid = (sp["start"] + sp["end"]) / 2; marks.append((mid, "0.3"))
        con = ConnectionPatch(xyA=(mid, -0.15), coordsA=ax_seq.transData, xyB=(0.5, 1.0),
                              coordsB=a.transAxes, color="0.4", lw=0.8, alpha=0.6)
        fig.add_artist(con)
    schematic(ax_seq, marks, lo, hi)
    fig.supylabel("contribution", fontsize=15, x=0.0)
    fig.suptitle(f"sox32 enhancer — {CT}: accessibility + contribution across motif regions (spans)",
                 fontsize=13, y=0.98)
    return fig, "spans"


_FEAT_COL = {"5'_flank": "#cfd8dc", "enhancer": "#90caf9", "min_promoter": "#ffcc80", "eGFP_CDS": "#a5d6a7"}


def construct_track(ax):
    """Linear construct with labeled feature blocks (5' flank / enhancer / promoter /
    eGFP) across 0..INPUT_LEN. Returns the y at which pull-out connectors start."""
    for lab, a, b in config.FEATURES:
        ax.add_patch(plt.Rectangle((a, 0.45), b - a, 0.4, fc=_FEAT_COL.get(lab, "0.9"),
                     ec="0.3", lw=0.7))
        if b - a >= 130:
            ax.text((a + b) / 2, 0.65, lab.replace("_", " "), ha="center", va="center", fontsize=8)
        else:
            ax.text((a + b) / 2, 0.95, lab.replace("_", " "), ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-20, config.INPUT_LEN + 20); ax.set_ylim(0, 1.15); ax.axis("off")
    return 0.45


def make_compare(cts):
    """Compare cell types over the SAME FIMO-defined spans, ONE shared contribution
    y-scale, in a single figure. Top: accessibility barplot + labeled construct track
    with two-line pull-out wedges to each region; below: one logo row per cell type."""
    spans = merge_spans()
    projs = {c: res.attr("construct", c) * res.ohe("construct") for c in cts}
    wins = [(s["start"] - SPAN_PAD, s["end"] + SPAN_PAD) for s in spans]
    vlim = max(np.abs(projs[c][:, w0:w1]).max() for c in cts for w0, w1 in wins) * 1.1
    widths = [w1 - w0 for w0, w1 in wins]
    fig = plt.figure(figsize=(15, 3.4 + 2.3 * len(cts)))
    gs = GridSpec(2 + len(cts), len(spans), height_ratios=[1.7, 1.3] + [2.3] * len(cts),
                  width_ratios=widths, hspace=0.7, wspace=0.28)
    barplot(fig.add_subplot(gs[0, :]), highlight=cts, show_title=False)
    ax_trk = fig.add_subplot(gs[1, :]); ytop = construct_track(ax_trk)
    top_axes = []
    for ri, c in enumerate(cts):
        psum = projs[c].sum(0)
        for k, sp in enumerate(spans):
            a = fig.add_subplot(gs[2 + ri, k]); w0, w1 = wins[k]
            plot_logo(projs[c][:, w0:w1], ax=a, ylim=vlim); a.set_yticks([])
            ctr = psum[sp["start"] - 1:sp["end"]].sum()
            # Σ in the title (above the logo) so it never sits behind the letters
            reg = f"{sp['start']}–{sp['end']}\n" if ri == 0 else ""
            a.set_title(f"{reg}Σ={ctr:.0f}", fontsize=8.5, color=cmap[c])
            if ri == 0:
                top_axes.append(a)
            if k == 0:
                a.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=11,
                             color=cmap[c], fontweight="bold")
            for h in sp["hits"]:
                tf = disp.get(h["motif_id"], h["motif_id"]); col = config.TF_COLORS.get(tf, "crimson")
                a.axvspan(int(h["start"]) - w0, int(h["stop"]) - w0, color=col, alpha=0.13, lw=0)
                if ri == 0:
                    a.text((int(h["start"]) + int(h["stop"])) / 2 - w0, a.get_ylim()[1] * 0.98,
                           tf, ha="center", va="top", fontsize=7, color=col, rotation=90,
                           bbox=dict(facecolor="white", alpha=0.7, pad=0.4, edgecolor="none"))
            if ri == len(cts) - 1:
                xt = np.linspace(0, w1 - w0, 3); a.set_xticks(xt)
                a.set_xticklabels([str(int(w0 + t)) for t in xt], fontsize=7)
            else:
                a.set_xticks([])
    # two-line pull-out wedges: construct track span -> top logo column corners
    for k, sp in enumerate(spans):
        ax_trk.plot([sp["start"], sp["end"]], [ytop - 0.06] * 2, color="0.45", lw=1.4)
        for sx, bx in ((sp["start"], 0.0), (sp["end"], 1.0)):
            fig.add_artist(ConnectionPatch(xyA=(sx, ytop - 0.06), coordsA=ax_trk.transData,
                           xyB=(bx, 1.0), coordsB=top_axes[k].transAxes, color="0.55", lw=0.8))
    fig.supylabel("contribution  (shared scale; Σ = summed over region)", fontsize=16, x=0.0)
    ttl = "all cell types" if CT == "all" else " vs ".join(cts)
    # fill the page: default margins leave a big blank top on tall (many-row) figures
    H = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.75 / H, bottom=0.012, left=0.07, right=0.995)
    fig.suptitle(f"sox32 enhancer — {ttl}: contribution at FIMO motif regions "
                 f"(same regions, shared scale)", fontsize=14, y=1 - 0.28 / H)
    return fig, ("compare_all" if CT == "all" else "compare_" + "_".join(cts))


CTS = res.ordered_celltypes() if CT == "all" else CT.split(",")
if CT == "all" or len(CTS) > 1:
    fig, tag = make_compare(CTS)
    out = f"figures/sox32_motif_summary_{tag}"
else:
    fig, tag = make_spans() if MODE == "spans" else make_sites()
    out = f"figures/sox32_{CT}_motif_summary_{tag}"
    if tag == "sites" and MATCHDB != "hoco":
        out += f"_{MATCHDB}"
    if REF != CT0:
        fig.text(0.5, 0.005, f"regions + y-scale fixed from {REF}; logos show {CT} contributions",
                 ha="center", fontsize=9, color="0.35")
        out += f"_ref{REF}"
out += config.SET_SUFFIX                                         # separate files per TF set
if Q_THRESH != 0.05:
    out += f"_q{Q_THRESH:g}"                                     # relaxed-threshold version (e.g. _q0.1)
fig.savefig(out + ".pdf", bbox_inches="tight")                 # vector (for Illustrator/Acrobat)
if len(CTS) <= 8:                                              # raster only for modest sizes
    fig.savefig(out + ".png", dpi=160, bbox_inches="tight")
print(f"[wrote] {out}.pdf  (mode={tag}, {len(CTS)} cell types)")
