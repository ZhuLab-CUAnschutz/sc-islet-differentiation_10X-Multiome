#!/usr/bin/env python
"""Multitask (CRESTED) synergy calls + clustermap + all-pairs-per-CT tables.

The MT model scores all 22 cell types in ONE forward pass, so the MT run gives all 2628 pairs × 22 CT
(results/mt) AND the lineage-mismatched null in all 22 CT (results/mt_null) — i.e. a per-cell-type null
for free. MT ΔJ is on the CRESTED-logit scale (Δ ~ -170..+1650), NOT comparable to the single-task
log-count Δ, so calls use MT's OWN per-CT null. Runs locally (CPU).

Outputs:
  mt/mt_per_ct_null_thresholds.tsv      per-CT MT null Δ/Z 95th pct + moments
  mt/mt_calls_long.tsv                   per (pair,CT): delta, dstar, maxZ, opt arrangement, call
  mt/mt_pair_summary.tsv                 per pair: breadth, tag, primary CT, Tau
  per_ct_tables_mt/<CT>__all_pairs.tsv   all 2628 pairs in each CT (MT), sorted by Δ  (the 'all-pairs' version)
  per_ct_tables_mt/<CT>__synergy.tsv     MT-synergistic pairs in each CT
  figures/mt_clustermap_dstar_rowz.png   MT specificity clustermap (compare to single-task blocks)
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

SYN = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(SYN, "figures")
MTD = os.path.join(SYN, "mt"); os.makedirs(MTD, exist_ok=True)
TBL = os.path.join(SYN, "per_ct_tables_mt"); os.makedirs(TBL, exist_ok=True)

ctm = pd.read_csv(f"{SYN}/cell_type_metadata.tsv", sep="\t").sort_values("display_order")
CTORDER = list(ctm.cell_type); LIN = dict(zip(ctm.cell_type, ctm.grouping))

# motif -> curator TF (all motifs, from metadata.tsv)
meta = pd.read_csv(f"{SYN}/metadata.tsv", sep="\t")
tf = {r.short_id: (r.curator_tf if isinstance(r.curator_tf, str) and r.curator_tf.strip() else r.short_id)
      for r in meta.itertuples()}

def load(d):
    return pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/{d}/*.tsv") if os.path.getsize(f) > 0],
                     ignore_index=True)
mt = load("mt"); mtn = load("mt_null")
cts = [c for c in CTORDER if c in set(mt.ct)]
print(f"MT {len(mt)} rows, {mt[['idA','idB']].drop_duplicates().shape[0]} pairs × {len(cts)} CT")

# ---- per-CT MT null thresholds ----
NT = {}
for ct, g in mtn.groupby("ct"):
    d = g.delta.dropna(); z = g.maxZ.dropna()
    NT[ct] = dict(dmean=float(d.mean()), dsd=float(d.std() + 1e-9),
                  d95=float(np.nanpercentile(d, 95)), z95=float(np.nanpercentile(z, 95)), n=int(len(g)))
nt = pd.DataFrame(NT).T.reindex([c for c in cts if c in NT])
nt.to_csv(f"{MTD}/mt_per_ct_null_thresholds.tsv", sep="\t")
print("per-CT MT null Δ95 range:", round(nt.d95.min(), 1), "..", round(nt.d95.max(), 1),
      "| Z95 range:", round(nt.z95.min(), 2), "..", round(nt.z95.max(), 2))

mt["pair"] = mt.idA + "__" + mt.idB
D = mt.pivot_table(index="pair", columns="ct", values="delta").reindex(columns=cts)
Zm = mt.pivot_table(index="pair", columns="ct", values="maxZ").reindex(columns=cts)
d95 = pd.Series({c: NT[c]["d95"] for c in cts}); z95 = pd.Series({c: NT[c]["z95"] for c in cts})
dmean = pd.Series({c: NT[c]["dmean"] for c in cts}); dsd = pd.Series({c: NT[c]["dsd"] for c in cts})
CALL = (D > d95) & (Zm > z95)
Dstar = (D - dmean) / dsd
breadth = CALL.sum(1)
def tagf(b): return "none" if b == 0 else "specific" if b <= 3 else "restricted" if b <= 11 else "broad"

# ---- per (pair,CT) long ----
ann = mt.drop_duplicates("pair").set_index("pair")[["idA", "idB", "wa", "wb"]]
rows = []
for r in mt.itertuples():
    p = r.pair; c = r.ct
    rows.append(dict(idA=r.idA, idB=r.idB, tfA=tf.get(r.idA, r.idA), tfB=tf.get(r.idB, r.idB), ct=c,
                     delta=round(float(r.delta), 2), dstar=round(float((r.delta - NT[c]["dmean"]) / NT[c]["dsd"]), 3),
                     dJ_opt=round(float(r.dJ_opt), 2), dS=round(float(r.dS), 2),
                     opt_orient=r.opt_orient, opt_gap=int(r.opt_gap), opt_center_center=round(float(r.opt_center_center), 1),
                     maxZ=round(float(r.maxZ), 2), wilcoxon_p=float(r.wilcoxon_p),
                     synergistic_in_ct=bool((r.delta > NT[c]["d95"]) and (r.maxZ > NT[c]["z95"]))))
L = pd.DataFrame(rows)
L.to_csv(f"{MTD}/mt_calls_long.tsv", sep="\t", index=False)

# ---- per-pair summary ----
def tau_index(row):
    x = np.clip(row.values.astype(float), 0, None)
    if not np.isfinite(x).any() or x.max() <= 0: return np.nan
    x = x / x.max(); return float(np.nansum(1 - x) / (np.isfinite(x).sum() - 1))
tau = Dstar.apply(tau_index, axis=1)
prim = D.idxmax(1)
summ = pd.DataFrame({"pair": D.index, "idA": ann.idA, "idB": ann.idB,
                     "tfA": [tf.get(a, a) for a in ann.idA], "tfB": [tf.get(b, b) for b in ann.idB],
                     "breadth": breadth.values, "tag": breadth.map(tagf).values,
                     "tau": tau.round(3).values, "primary_ct": prim.values,
                     "primary_delta": D.max(1).round(2).values})
summ.sort_values(["breadth", "primary_delta"], ascending=[True, False]).to_csv(f"{MTD}/mt_pair_summary.tsv", sep="\t", index=False)
n_syn_pairs = int((breadth > 0).sum())
print(f"MT synergistic pairs (≥1 CT, MT per-CT null): {n_syn_pairs}/{len(D)}  tag: {breadth.map(tagf).value_counts().to_dict()}")
print("MT synergistic pairs per CT:", {c: int(CALL[c].sum()) for c in cts})

# ---- all-pairs & synergy per-CT tables (MT) ----
cols = ["idA", "idB", "tfA", "tfB", "delta", "dstar", "dJ_opt", "dS", "opt_orient", "opt_gap",
        "opt_center_center", "maxZ", "wilcoxon_p", "synergistic_in_ct"]
for c in cts:
    sub = L[L.ct == c].sort_values("delta", ascending=False)
    sub[cols].to_csv(f"{TBL}/{c}__all_pairs.tsv", sep="\t", index=False)
    sub[sub.synergistic_in_ct][cols].to_csv(f"{TBL}/{c}__synergy.tsv", sep="\t", index=False)
print(f"wrote {len(cts)*2} MT per-CT tables (all_pairs + synergy) to per_ct_tables_mt/")

# ---- MT specificity clustermap (synergistic pairs, row-z of Δ*) ----
syn = breadth[breadth > 0].index
Rz = Dstar.loc[syn].sub(Dstar.loc[syn].mean(1), axis=0).div(Dstar.loc[syn].std(1) + 1e-9, axis=0).fillna(0)
Lk = linkage(Rz.values, method="ward")
sil = {k: silhouette_score(Rz.values, fcluster(Lk, k, "maxclust"))
       for k in range(2, 13) if len(set(fcluster(Lk, k, "maxclust"))) > 1}
kbest = max(sil, key=sil.get) if sil else 2
clusters = pd.Series(fcluster(Lk, kbest, "maxclust"), index=syn, name="cluster")
clusters.to_csv(f"{MTD}/mt_synergy_clusters_k{kbest}.tsv", sep="\t")
print(f"MT clustermap: {len(syn)} pairs, natural k={kbest} (silhouette {sil.get(kbest, float('nan')):.3f})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
cg = sns.clustermap(Rz[cts], row_linkage=Lk, col_cluster=False, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                    figsize=(11, 13), xticklabels=True, yticklabels=False,
                    cbar_kws={"label": "row-z of Δ* (MT specificity)"})
cg.ax_heatmap.set_xlabel("cell type (developmental order)"); cg.ax_heatmap.set_ylabel(f"{len(syn)} MT-synergistic pairs")
cg.fig.suptitle(f"Multitask synergy specificity (standardized Δ*, row-z) — natural k={kbest}", y=1.01)
cg.savefig(f"{FIG}/mt_clustermap_dstar_rowz.png", dpi=135, bbox_inches="tight"); plt.close()

json.dump({"n_pairs": len(D), "n_synergistic": n_syn_pairs, "tags": breadth.map(tagf).value_counts().to_dict(),
           "natural_k": int(kbest), "silhouette": round(float(sil.get(kbest, np.nan)), 3),
           "per_ct_synergistic": {c: int(CALL[c].sum()) for c in cts}},
          open(f"{MTD}/mt_summary.json", "w"), indent=2)
print("wrote figures/mt_clustermap_dstar_rowz.png + mt/ tables")
