#!/usr/bin/env python
"""
08_relative_accessibility.py — how accessible is the sox32 construct *relative to
each cell type's own called peaks*? The construct's predicted log-counts is placed
in the distribution of that cell type's peak predicted log-counts (from the
ChromBPNet eval), giving a percentile. This is a natural per-model normalization
(addresses the un-normalized cross-cell-type counts caveat).

Outputs:
  (a) DE peak log-counts distribution with the construct + enh_dinuc marked + percentile
  (b) percentile of the construct across the whole trajectory (normalized accessibility)
Run under eugene_tools.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
plt = config.setup_mpl()
import chrombpnet_locus as cl
import plot_locus as pl

NPZ = sys.argv[1] if len(sys.argv) > 1 else "outputs/sox32_all.npz"
FIG, OUT = "figures", "outputs"
res = pl.LocusResults(NPZ)
cmap = cl.color_map()
traj = res.ordered_celltypes()

def pct(logc, dist):
    return 100.0 * (dist < logc).mean()

# ---- (a) DE distribution with construct markers ----
de_lc = cl.peak_logcounts("DE")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(de_lc, bins=80, color="0.8", edgecolor="none")
ax.axvline(np.median(de_lc), color="0.4", ls=":", lw=1, label=f"DE peak median ({np.median(de_lc):.2f})")
for v, col in [("construct", "#d6604d"), ("enh_dinuc", "#4393c3")]:
    lc = np.log(res.counts(v, "DE"))
    ax.axvline(lc, color=col, lw=2, label=f"sox32 {v}: {lc:.2f}  ({pct(lc, de_lc):.1f}th pct)")
ax.set_xlabel("predicted log-counts"); ax.set_ylabel("# DE peaks (n=%d)" % len(de_lc))
ax.set_title("sox32 enhancer vs the DE peak accessibility distribution")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/sox32_DE_relative_accessibility.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/sox32_DE_relative_accessibility.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[wrote] DE relative-accessibility histogram")

# ---- (b) percentile across the trajectory (normalized) ----
rows = []
for c in traj:
    try:
        dist = cl.peak_logcounts(c)
    except Exception as e:
        print(f"  skip {c}: {e}"); continue
    rows.append({"celltype": c,
                 "construct_pct": pct(np.log(res.counts("construct", c)), dist),
                 "enh_dinuc_pct": pct(np.log(res.counts("enh_dinuc", c)), dist),
                 "peak_median_logc": float(np.median(dist))})
pdf = pd.DataFrame(rows)
pdf.to_csv(f"{OUT}/sox32_relative_accessibility_percentile.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.5))
xs = range(len(pdf))
ax.bar(xs, pdf["construct_pct"], color=[cmap[c] for c in pdf.celltype], edgecolor="k", lw=0.4)
ax.plot(xs, pdf["enh_dinuc_pct"], "o-", color="0.2", ms=4, lw=1, label="enh_dinuc")
ax.axhline(50, color="0.5", ls=":", lw=0.8)
ax.set_xticks(list(xs)); ax.set_xticklabels(pdf.celltype, rotation=90)
ax.set_ylabel("percentile within that cell type's peaks")
ax.set_title("sox32 construct: accessibility percentile vs each cell type's own peaks "
             "(bars=construct, dots=enh_dinuc)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/sox32_relative_accessibility_trajectory.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/sox32_relative_accessibility_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[wrote] trajectory percentile figure")
print(pdf.round(1).to_string(index=False))
