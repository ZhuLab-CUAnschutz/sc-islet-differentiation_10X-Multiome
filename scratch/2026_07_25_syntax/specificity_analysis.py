#!/usr/bin/env python
"""Define cell-type-specific synergy THREE ways and test for coherence between them.

Call layer (per-CT null-calibrated): a pair is 'synergistic in cell type X' if, in X's own model,
Δ exceeds X's null 95th pct AND maxZ exceeds X's null-Z 95th pct. Per-CT null = the 42 lineage-
mismatched pairs run through each of the 22 single-task models (results/null_allct).

Specificity layer (three definitions of 'specific to X'):
  M1  standardized-Δ clustering: Δ* = (Δ − null_mean_ct)/null_sd_ct (magnitude-comparable across
      models); row-z across CTs; Ward clustering -> blocks; the global view.
  M2  categorical breadth: # cell types a pair is synergistic in -> specific / restricted / broad.
  M3  differential contrast: Tau specificity index on Δ*, one-vs-rest score per CT, and a per-lineage
      Mann-Whitney contrast (Δ* in-lineage vs out) -> the lineage each pair is specific to + p.

Coherence: do the three agree on WHICH pairs are specific and to WHICH cell types/lineages? Output a
consensus 'cell-type-specific' set (pairs flagged by all three, coherent lineage). Runs locally (CPU).
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

SYN = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(SYN, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(SYN, "specificity"); os.makedirs(OUT, exist_ok=True)

def load_summaries(d):
    fs = [f for f in glob.glob(f"{SYN}/RES/{d}/summary__*.tsv") if os.path.getsize(f) > 0]
    if not fs: return None
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)

# ---- cell types (developmental order + lineage grouping) ----
ctm = pd.read_csv(os.path.join(SYN, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
CTORDER = list(ctm.cell_type); LIN = dict(zip(ctm.cell_type, ctm.grouping))

# ---- per-CT null (results/null_allct): thresholds per cell type ----
nul = load_summaries("null_allct")
if nul is None:
    raise SystemExit("results/null_allct not present yet — run the per-CT null first (job null_allct).")
NT = {}   # ct -> thresholds/moments
for ct, g in nul.groupby("ct"):
    d = g.delta.dropna(); z = g.maxZ.dropna()
    NT[ct] = dict(dmean=float(d.mean()), dsd=float(d.std() + 1e-9), d95=float(np.nanpercentile(d, 95)),
                  z95=float(np.nanpercentile(z, 95)), n=int(len(g)))
nt = pd.DataFrame(NT).T.reindex([c for c in CTORDER if c in NT])
nt.to_csv(f"{OUT}/per_ct_null_thresholds.tsv", sep="\t")
print(f"per-CT null: {len(NT)} cell types\n", nt.round(3).to_string())

# ---- syn_allct: per-(pair,CT) Δ, maxZ (582 × 22) ----
sa = load_summaries("syn_allct")
sa["pair"] = sa.idA + "__" + sa.idB
cts_present = [c for c in CTORDER if c in set(sa.ct)]
D = sa.pivot_table(index="pair", columns="ct", values="delta").reindex(columns=cts_present)
Z = sa.pivot_table(index="pair", columns="ct", values="maxZ").reindex(columns=cts_present)
pairs = D.index.tolist()
# annotation
ann = sa.drop_duplicates("pair").set_index("pair")[["idA", "idB"]]
lng = pd.read_csv(f"{FIG}/synergistic_pairs_allCT_long.tsv", sep="\t")
lng["pair"] = lng.idA + "__" + lng.idB
meta = lng.drop_duplicates("pair").set_index("pair")[["tfA", "tfB", "family_A", "family_B"]]
ann = ann.join(meta)

# ---- standardized Δ* = (Δ − null_mean)/null_sd per cell type ----
dmean = pd.Series({c: NT[c]["dmean"] for c in cts_present})
dsd = pd.Series({c: NT[c]["dsd"] for c in cts_present})
Dstar = (D - dmean) / dsd
# row-z of Δ* across cell types (specificity, not magnitude)
rowz = Dstar.sub(Dstar.mean(1), axis=0).div(Dstar.std(1) + 1e-9, axis=0)

# ==== M2: per-CT synergy calls + breadth tag ====
d95 = pd.Series({c: NT[c]["d95"] for c in cts_present})
z95 = pd.Series({c: NT[c]["z95"] for c in cts_present})
CALL = (D > d95) & (Z > z95)
breadth = CALL.sum(1)
def tag(b):
    return "none" if b == 0 else "specific" if b <= 3 else "restricted" if b <= 11 else "broad"
tags = breadth.map(tag)

# ==== M1: cluster pairs on row-z(Δ*); natural k by silhouette ====
syn_pairs = breadth[breadth > 0].index          # cluster only pairs synergistic in >=1 CT (per-CT null)
Rz = rowz.loc[syn_pairs].fillna(0.0)
Lk = linkage(Rz.values, method="ward")
sil = {}
for k in range(2, 13):
    lab = fcluster(Lk, k, criterion="maxclust")
    if len(set(lab)) > 1:
        sil[k] = float(silhouette_score(Rz.values, lab))
kbest = max(sil, key=sil.get) if sil else 2
clusters = pd.Series(fcluster(Lk, kbest, criterion="maxclust"), index=syn_pairs, name="cluster")
print(f"\nM1 clustering: {len(syn_pairs)} pairs, natural k={kbest} (silhouette {sil.get(kbest,float('nan')):.3f})")

# ==== M3: Tau specificity index, one-vs-rest, lineage contrast ====
def tau_index(row):
    x = np.clip(row.values.astype(float), 0, None)
    if np.all(~np.isfinite(x)) or x.max() <= 0: return np.nan
    x = x / x.max()
    return float(np.nansum(1 - x) / (np.isfinite(x).sum() - 1))
tau = Dstar.apply(tau_index, axis=1)            # 0 = broad, 1 = one-CT-specific
# one-vs-rest per (pair,ct): Δ* minus median of the pair's other CTs
def ovr_matrix(M):
    out = M.copy() * np.nan
    for p in M.index:
        r = M.loc[p]
        for c in M.columns:
            out.loc[p, c] = r[c] - np.nanmedian(r.drop(c))
    return out
OVR = ovr_matrix(Dstar)
top_ct_m1 = rowz.idxmax(1)                       # CT of peak specificity (M1)
top_ct_m3 = OVR.idxmax(1)                         # CT of peak one-vs-rest (M3)
# lineage contrast: best lineage per pair by Mann-Whitney of Δ* in vs out
lineages = sorted(set(LIN[c] for c in cts_present))
ct_lin = {c: LIN[c] for c in cts_present}
best_lin, best_linp = {}, {}
for p in pairs:
    r = Dstar.loc[p]
    bestp, bl = 1.0, None
    for L in lineages:
        inl = [c for c in cts_present if ct_lin[c] == L]
        out = [c for c in cts_present if ct_lin[c] != L]
        if len(inl) < 2 or len(out) < 2: continue
        vi, vo = r[inl].dropna(), r[out].dropna()
        if len(vi) < 2 or len(vo) < 2: continue
        try:
            u, pv = mannwhitneyu(vi, vo, alternative="greater")
        except ValueError:
            continue
        if pv < bestp and vi.median() > vo.median(): bestp, bl = pv, L
    best_lin[p], best_linp[p] = bl, bestp

# ==== coherence across methods ====
def specific_set_m1(p): return set(rowz.columns[(rowz.loc[p] > 1.0).values])
def specific_set_m2(p): return set(CALL.columns[CALL.loc[p].values])
def specific_set_m3(p): return set(OVR.columns[(OVR.loc[p] > 1.0).values])
def jacc(a, b): return len(a & b) / len(a | b) if (a | b) else np.nan
rows = []
for p in pairs:
    s1, s2, s3 = specific_set_m1(p), specific_set_m2(p), specific_set_m3(p)
    rows.append(dict(pair=p, idA=ann.loc[p, "idA"], idB=ann.loc[p, "idB"],
                     tfA=ann.loc[p, "tfA"], tfB=ann.loc[p, "tfB"],
                     breadth=int(breadth[p]), tag=tags[p], tau=round(float(tau[p]), 3),
                     cluster=int(clusters[p]) if p in clusters.index else 0,
                     top_ct_rowz=top_ct_m1[p], top_ct_ovr=top_ct_m3[p],
                     best_lineage=best_lin[p], lineage_p=best_linp[p],
                     jac_m1m2=round(jacc(s1, s2), 3), jac_m1m3=round(jacc(s1, s3), 3),
                     jac_m2m3=round(jacc(s2, s3), 3),
                     top_ct_agree=int(top_ct_m1[p] == top_ct_m3[p]),
                     consensus_specific=int(tags[p] == "specific" and tau[p] >= 0.6
                                            and top_ct_m1[p] == top_ct_m3[p])))
scores = pd.DataFrame(rows).sort_values(["consensus_specific", "tau"], ascending=[False, False])
scores.to_csv(f"{OUT}/specificity_scores.tsv", sep="\t", index=False)

# per (pair,ct) long: delta, dstar, rowz, ovr, call
long_rows = []
for p in pairs:
    for c in cts_present:
        long_rows.append(dict(pair=p, idA=ann.loc[p, "idA"], idB=ann.loc[p, "idB"], ct=c, lineage=ct_lin[c],
                              delta=round(float(D.loc[p, c]), 3) if pd.notna(D.loc[p, c]) else None,
                              dstar=round(float(Dstar.loc[p, c]), 3) if pd.notna(Dstar.loc[p, c]) else None,
                              rowz=round(float(rowz.loc[p, c]), 3) if pd.notna(rowz.loc[p, c]) else None,
                              ovr=round(float(OVR.loc[p, c]), 3) if pd.notna(OVR.loc[p, c]) else None,
                              call=bool(CALL.loc[p, c])))
pd.DataFrame(long_rows).to_csv(f"{OUT}/specificity_by_ct_long.tsv", sep="\t", index=False)

# coherence summary
coh = dict(
    n_pairs=len(pairs),
    n_synergistic_any=int((breadth > 0).sum()),
    tag_counts=tags.value_counts().to_dict(),
    natural_k=int(kbest), silhouette=round(float(sil.get(kbest, np.nan)), 3),
    tau_vs_breadth_spearman=round(float(pd.Series(tau).corr(breadth.astype(float), method="spearman")), 3),
    median_jaccard_m1m2=round(float(scores.jac_m1m2.median()), 3),
    median_jaccard_m1m3=round(float(scores.jac_m1m3.median()), 3),
    median_jaccard_m2m3=round(float(scores.jac_m2m3.median()), 3),
    frac_topct_agree=round(float(scores.top_ct_agree.mean()), 3),
    n_consensus_specific=int(scores.consensus_specific.sum()),
)
json.dump(coh, open(f"{OUT}/coherence_summary.json", "w"), indent=2)
print("\nCOHERENCE:", json.dumps(coh, indent=2))

# cluster × lineage enrichment (do M1 blocks map to lineages that M3 flags?)
if len(syn_pairs):
    cl = clusters.to_frame().join(scores.set_index("pair")[["best_lineage", "tag"]])
    ct_lin_tab = pd.crosstab(cl.cluster, cl.best_lineage)
    ct_lin_tab.to_csv(f"{OUT}/cluster_by_lineage.tsv", sep="\t")
    print("\ncluster × best-lineage:\n", ct_lin_tab.to_string())

# ---- figures: Δ* clustermap (dev-order cols) + tau-vs-breadth + method-agreement ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
# clustermap of row-z(Δ*), rows clustered, cols developmental
cg = sns.clustermap(Rz[cts_present], row_linkage=Lk, col_cluster=False, cmap="RdBu_r", center=0,
                    vmin=-2.5, vmax=2.5, figsize=(11, 13), xticklabels=True, yticklabels=False,
                    cbar_kws={"label": "row-z of Δ* (specificity)"})
cg.ax_heatmap.set_xlabel("cell type (developmental order)"); cg.ax_heatmap.set_ylabel(f"{len(syn_pairs)} synergistic pairs")
cg.fig.suptitle("M1 — cell-type-specificity of synergy (standardized Δ*, row-z)", y=1.01)
cg.savefig(f"{FIG}/spec_clustermap_dstar_rowz.png", dpi=135, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(breadth, tau, s=10, c=breadth, cmap="viridis_r", alpha=.6)
ax[0].set_xlabel("breadth = # cell types synergistic in (M2)"); ax[0].set_ylabel("Tau specificity index (M3)")
ax[0].set_title(f"M2 vs M3  (Spearman {coh['tau_vs_breadth_spearman']})")
jm = scores[["jac_m1m2", "jac_m1m3", "jac_m2m3"]].median()
ax[1].bar(["M1–M2", "M1–M3", "M2–M3"], jm.values, color=["#3b2d7e", "#1e6091", "#e67e22"])
ax[1].set_ylabel("median Jaccard of specific-CT sets"); ax[1].set_ylim(0, 1); ax[1].set_title("method agreement (specific-CT sets)")
plt.tight_layout(); plt.savefig(f"{FIG}/spec_method_coherence.png", dpi=135, bbox_inches="tight"); plt.close()

print(f"\nwrote specificity/ tables + figures/spec_clustermap_dstar_rowz.png + figures/spec_method_coherence.png")
print(f"consensus cell-type-specific pairs: {int(scores.consensus_specific.sum())}")
print(scores[scores.consensus_specific == 1].head(15)[["idA","idB","tfA","tfB","tag","tau","top_ct_rowz","best_lineage"]].to_string(index=False))
