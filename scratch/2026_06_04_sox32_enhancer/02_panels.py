#!/usr/bin/env python
"""
02_panels.py — per-cell-type figures from a prediction .npz.

  * predicted-accessibility bar plot (both variants)
  * full-window contribution panel (no seqlet annotation; fast overview)
  * zoom panels with native plot_logo seqlet annotation (Vierstra labels)
  * seqlet boundary table annotated vs catalog / named3 / vierstra, with the
    construct feature each seqlet falls in (enhancer / promoter / flank / eGFP)

    python 02_panels.py outputs/sox32_all.npz construct
Run under eugene_tools (head-node safe; plotting only).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import plot_locus as pl

NPZ = sys.argv[1] if len(sys.argv) > 1 else "outputs/sox32_all.npz"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "construct"
ZOOMS = [("EOMES_Tbox", (600, 800)), ("FOXH1_region", (1280, 1470))]
FIG, OUT = "figures", "outputs"
tag = os.path.splitext(os.path.basename(NPZ))[0]
res = pl.LocusResults(NPZ)
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# 1. accessibility bar (all variants)
fig, axes = plt.subplots(1, len(res.variants), figsize=(6.5 * len(res.variants), 4), squeeze=False)
for j, v in enumerate(res.variants):
    pl.predicted_accessibility_barplot(res, variant=v, ax=axes[0, j])
fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_accessibility_bar.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/{tag}_accessibility_bar.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote accessibility bar")

# 2. full-window overview (no annotation — slow on 1 kb logos)
fig, _ = pl.cross_celltype_panel(res, variant=VARIANT, annotate_logos=False)
fig.savefig(f"{FIG}/{tag}_{VARIANT}_panel_fullwindow.pdf", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote full-window panel")

# 3. zoom panels (native Vierstra seqlet annotation)
for nm, region in ZOOMS:
    fig, _ = pl.cross_celltype_panel(res, variant=VARIANT, region=region)
    base = f"{FIG}/{tag}_{VARIANT}_zoom_{nm}_{region[0]}_{region[1]}"
    fig.savefig(base + ".pdf", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote zoom {nm} {region}")

# 4. seqlet boundary table over the enhancer, 3 DBs + construct feature
tbl = pl.seqlet_table(res, variant=VARIANT, region=config.ENHANCER)
tbl.to_csv(f"{OUT}/{tag}_{VARIANT}_seqlets.csv", index=False)
print(f"wrote {OUT}/{tag}_{VARIANT}_seqlets.csv ({len(tbl)} seqlets)")
print("DONE")
