#!/usr/bin/env python
"""
03_fimo_scan.py — direct PWM (MEME-suite FIMO v5.5.0) scan of the construct for
EOMES / T / FOXH1. Sequence-based, independent of the model. Writes a feature-
annotated hits CSV and a readable hit-map showing where each TF motif sits
relative to the construct features (enhancer / promoter / flank / eGFP).

Run under eugene_tools.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import fimo

OUTDIR = config.FIMO_DIR
t = fimo.run_fimo(config.FASTA, OUTDIR, thresh=1e-3)
t.to_csv(f"{OUTDIR}/sox32_fimo_hits.csv", index=False)
print(f"FIMO: {len(t)} hits  | by feature: "
      + ", ".join(f"{k}={v}" for k, v in t['feature'].value_counts().items()))

# hit-map: one row per TF, bars at positions (height=-log10 p, solid=q<0.05),
# construct features shaded underneath.
fimo_names = {v: k for k, v in config.FIMO_TF_NAMES.items()}   # safe -> display
motifs = list(config.FIMO_TF_NAMES.values())
fig, axes = plt.subplots(len(motifs), 1, figsize=(15, 1.5 * len(motifs) + 1), sharex=True)
feat_col = {"5'_flank": "#cfd8dc", "enhancer": "#90caf9", "min_promoter": "#ffcc80", "eGFP_CDS": "#a5d6a7"}
for ax, m in zip(np.atleast_1d(axes), motifs):
    disp = fimo_names[m]; col = config.TF_COLORS[disp]
    for lab, a, b in config.FEATURES:                       # feature backdrop
        ax.axvspan(a, b, color=feat_col.get(lab, "0.95"), alpha=0.35, lw=0)
    for _, r in t[t.motif_id == m].iterrows():
        h = -np.log10(r["p-value"]); sig = r["q-value"] < 0.05
        ax.add_patch(plt.Rectangle((r["start"], 0), max(r["stop"] - r["start"], 4), h,
                     color=col, alpha=0.9 if sig else 0.35, lw=1.2 if sig else 0, ec="k"))
        if sig:
            ax.text((r["start"] + r["stop"]) / 2, h + 0.1, f"{int(r['start'])}{r['strand']}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_ylabel(disp, rotation=0, ha="right", va="center", fontsize=10)
    ax.set_ylim(0, 6); ax.set_yticks([0, 2, 4]); ax.margins(x=0)
for lab, a, b in config.FEATURES:
    axes[0].text((a + b) / 2, 5.6, lab, ha="center", fontsize=7, color="0.3")
axes[-1].set_xlabel("construct position (bp) — bar height=-log10(p); solid=q<0.05; shading=feature")
fig.suptitle("FIMO (MEME-suite v5.5.0) EOMES / T / FOXH1 PWM hits vs construct features", fontsize=12)
fig.tight_layout()
fig.savefig(f"figures/sox32_fimo_hitmap{config.SET_SUFFIX}.pdf", bbox_inches="tight")
fig.savefig(f"figures/sox32_fimo_hitmap{config.SET_SUFFIX}.png", dpi=150, bbox_inches="tight")
print(f"[wrote] figures/sox32_fimo_hitmap{config.SET_SUFFIX}.png")
