#!/usr/bin/env python
"""Logos of the 7 inserts actually used in the DE 4-TF synergy run.
Catalog column: signed CWM (contribution, trimmed at imp_frac=0.1 = the inserted region).
JASPAR column: IC-trimmed (>=0.2 bit) PPM drawn as an information-content logo (bits).
Consensus string under each logo is what marginalize_pairs.py inserts."""
import os, sys, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, logomaker
HERE = os.path.dirname(os.path.abspath(__file__)); WD = os.path.join(HERE, ".."); sys.path.insert(0, HERE)
from build_inputs import read_jaspar, ic_bits, IC_MIN
IN = os.path.join(WD, "inputs"); FIG = os.path.join(WD, "figures"); os.makedirs(FIG, exist_ok=True)
cw = np.load(os.path.join(IN, "cwms_v1.1.npz")); ins = pd.read_csv(os.path.join(IN, "inserts.tsv"), sep="\t").set_index("id")
ROWS = [("EOMES", "ST18", "J_EOMES", "MA0800.1"), ("MIXL1", "ST36_sub1", "J_MIXL1", "MA0662.1"),
        ("SOX17", "ST66", "J_SOX17", "MA0078.1"), ("FOXH1", None, "J_FOXH1", "MA0479.1")]
RC = str.maketrans("ACGT", "TGCA")
def revcomp(seq): return seq.translate(RC)[::-1]
def rc_pfm(pfm): return pfm[::-1, ::-1]
def align_score(a, b, min_overlap=4):
    """best ungapped match count between two consensus strings over all offsets (overlap >= min_overlap)"""
    best = 0
    for off in range(-len(b) + min_overlap, len(a) - min_overlap + 1):
        ov = [(a[i], b[i - off]) for i in range(max(0, off), min(len(a), off + len(b)))]
        if len(ov) >= min_overlap: best = max(best, sum(x == y for x, y in ov))
    return best
def choose_orientation(cat_cons, j_cons):
    f, r = align_score(cat_cons, j_cons), align_score(cat_cons, revcomp(j_cons))
    return ("rc" if r > f else "fwd"), f, r

ORIENT = len(sys.argv) > 1 and sys.argv[1] == "--match_orientation"
COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
fig, axes = plt.subplots(4, 2, figsize=(10, 9.2), gridspec_kw={"hspace": 0.95, "wspace": 0.25})
for r, (tf, cid, jid, mid) in enumerate(ROWS):
    ax = axes[r, 0]
    if cid is None:
        ax.axis("off"); ax.text(0.5, 0.5, "no catalog v1.1 motif for FOXH1\n(JASPAR insert used in both sets)", ha="center", va="center", fontsize=10, color="#666")
    else:
        m = cw[cid]; W4 = m if m.shape[1] == 4 else m.T; imp = np.abs(W4).sum(1); keep = np.where(imp >= 0.1 * imp.max())[0]
        core = pd.DataFrame(W4[keep.min():keep.max() + 1], columns=list("ACGT"))
        logomaker.Logo(core, ax=ax, color_scheme=COLORS, shade_below=0.5, fade_below=0.5)
        ax.set_ylabel("contribution", fontsize=8)
        ax.set_title(f"{cid}  ·  {tf}  ·  catalog v1.1 CWM\ninsert {ins.loc[cid,'consensus']}  ({ins.loc[cid,'width']} bp)", fontsize=9.5, loc="left")
    ax = axes[r, 1]
    pfm = read_jaspar(os.path.join(IN, "jaspar", f"{mid}.jaspar")); ic = ic_bits(pfm); keep = np.where(ic >= IC_MIN)[0]
    core = pfm[keep.min():keep.max() + 1]; cons = ins.loc[jid, "consensus"]; note = ""
    if ORIENT and cid is not None:
        o, f, r = choose_orientation(ins.loc[cid, "consensus"], cons)
        print(f"{tf}: fwd match={f} rc match={r} -> {o}")
        if o == "rc": core = rc_pfm(core); cons = revcomp(cons); note = "  [revcomp to match catalog strand]"
    p = core / core.sum(1, keepdims=True)
    info = logomaker.transform_matrix(pd.DataFrame(p, columns=list("ACGT")), from_type="probability", to_type="information")
    logomaker.Logo(info, ax=ax, color_scheme=COLORS); ax.set_ylim(0, 2); ax.set_ylabel("bits", fontsize=8)
    ax.set_title(f"{jid}  ·  {tf}  ·  JASPAR {mid} (IC ≥ {IC_MIN} bit trim)\ninsert {cons}  ({ins.loc[jid,'width']} bp; untrimmed {int(ins.loc[jid,'untrimmed_width'])} bp){note}", fontsize=9.5, loc="left")
for ax in axes.ravel():
    if ax.axison:
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7); ax.set_xticks(range(len(ax.get_xticks()))) if False else None
name = "fig0b_insert_logos_oriented" if ORIENT else "fig0_insert_logos"
sub = " — JASPAR logos on the strand matching the catalog motif" if ORIENT else " (what marginalize_pairs.py places in the background)"
fig.suptitle("Inserted motif consensus sequences — DE 4-TF synergy run" + sub, fontsize=11, y=0.995)
for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
print(f"wrote {name}.png/pdf")
