#!/usr/bin/env python
"""
07_construct_map.py — annotated map of the 2114 bp construct window, from the
sequence + collaborator info (features defined in config.FEATURES). Linear insert
map (the window is the ChromBPNet input, not the full plasmid). Shows feature
blocks, point markers (TATA/Kozak/ATG), a GC% track, and FIMO TF-site ticks.
Run under eugene_tools.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
from matplotlib.patches import FancyBboxPatch
import chrombpnet_locus as cl

name, seq = cl.read_fasta(config.FASTA); L = len(seq)
feat_col = {"5'_flank": "#cfd8dc", "enhancer": "#90caf9", "min_promoter": "#ffcc80", "eGFP_CDS": "#a5d6a7"}
out_win = config.output_window()
gc = np.array([100 * sum(c in "GC" for c in seq[max(0, i-50):i+50]) / len(seq[max(0, i-50):i+50])
               for i in range(L)])
fimo = pd.read_csv(config.FIMO_HITS_CSV)
disp = {v: k for k, v in config.FIMO_TF_NAMES.items()}

fig, (axm, axg) = plt.subplots(2, 1, figsize=(15, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
axm.axhline(0, color="0.3", lw=1)
for lab, a, b in config.FEATURES:
    axm.add_patch(FancyBboxPatch((a, -0.18), b - a, 0.36, boxstyle="round,pad=0,rounding_size=8",
                  fc=feat_col.get(lab, "0.9"), ec="0.3", lw=0.8))
    # wide blocks: label inside; narrow blocks: label below with a leader line
    if b - a >= 130:
        axm.text((a + b) / 2, 0, lab.replace("_", " "), ha="center", va="center", fontsize=8.5)
    else:
        axm.annotate(lab.replace("_", " "), xy=((a + b) / 2, -0.18), xytext=((a + b) / 2, -0.34),
                     ha="center", va="top", fontsize=7.5,
                     arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
axm.axvspan(*out_win, color="0.85", alpha=0.35, zorder=0)
axm.text(np.mean(out_win), 1.04, "ChromBPNet output window (1 kb)", ha="center", fontsize=8, color="0.3")
# point features (low band): fan labels so closely-spaced TATA/Kozak/ATG don't collide
_pf_lab = {"TATA": (1430, 0.32), "Kozak": (1730, 0.40), "eGFP_ATG": (1880, 0.48)}
for lab, p in config.POINT_FEATURES:
    axm.plot([p], [0.20], marker="v", color="crimson", ms=6)
    lx, ly = _pf_lab.get(lab, (p, 0.34))
    axm.annotate(f"{lab.replace('_',' ')} @{p}", xy=(p, 0.22), xytext=(lx, ly),
                 ha="center", va="bottom", fontsize=7, color="crimson",
                 arrowprops=dict(arrowstyle="-", color="crimson", lw=0.5))
# FIMO TF ticks (upper band)
for i, (m, disp_name) in enumerate(disp.items()):
    y = 0.66 + 0.12 * i
    for _, r in fimo[fimo.motif_id == m].iterrows():
        sig = r["q-value"] < 0.05
        axm.plot([(r["start"] + r["stop"]) / 2], [y], marker="|",
                 ms=10 if sig else 6, color=config.TF_COLORS[disp_name], alpha=0.9 if sig else 0.4,
                 mew=2 if sig else 1)
    axm.text(L + 20, y, disp_name, color=config.TF_COLORS[disp_name], fontsize=8, va="center")
axm.set_ylim(-0.45, 1.1); axm.set_yticks([]); axm.set_xlim(0, L + 130)
axm.set_title(f"{name} — construct map (2114 bp ChromBPNet window) | bars=FIMO sites (bold=q<0.05)", fontsize=11)
axg.fill_between(range(L), gc, color="0.6", lw=0); axg.axhline(50, color="0.4", lw=0.5, ls=":")
for lab, a, b in config.FEATURES:
    axg.axvspan(a, b, color=feat_col.get(lab, "0.9"), alpha=0.25, lw=0)
axg.set_ylabel("GC %"); axg.set_xlabel("construct position (bp)"); axg.set_ylim(0, 100); axg.margins(x=0)
fig.tight_layout()
fig.savefig("figures/sox32_construct_map.pdf", bbox_inches="tight")
fig.savefig("figures/sox32_construct_map.png", dpi=150, bbox_inches="tight")
print("[wrote] figures/sox32_construct_map.png")
