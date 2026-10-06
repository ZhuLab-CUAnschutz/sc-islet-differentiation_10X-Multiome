#!/usr/bin/env python
"""Define which cell states / groups of cell states 'have' a synergistic pair.

Substrate: the binary per-(pair,CT) synergy calls (Delta>d95 & maxZ>z95) already in
specificity/specificity_by_ct_long.tsv. Three views:

  L1  single-state private pairs      : per cell state -> pairs on ONLY there (breadth==1)
  L2  supervised group rollup         : per developmental lineage -> pairs on across that lineage
  L3  discovered groups (unsupervised): cluster cell STATES by Jaccard of the pairs they co-share,
      then (a) majority-vote pair modules per discovered group, (b) formal spectral bicluster,
      and cross-tab discovered state-groups vs known lineages as a coherence check.
"""
import os, json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

SYN = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(SYN, "specificity")
FIG = os.path.join(SYN, "figures")
OUT = os.path.join(SYN, "specificity_groups"); os.makedirs(OUT, exist_ok=True)

# ---- load calls + build pair x state CALL matrix ----
lng = pd.read_csv(f"{SPEC}/specificity_by_ct_long.tsv", sep="\t")
CT_LIN = lng.drop_duplicates("ct").set_index("ct")["lineage"].to_dict()   # ct -> lineage
CALL = lng.pivot_table(index="pair", columns="ct", values="call", aggfunc="first").fillna(False).astype(bool)
# keep developmental column order as it appears in the long file
CTORDER = list(dict.fromkeys(lng.ct))
CALL = CALL.reindex(columns=[c for c in CTORDER if c in CALL.columns])

scores = pd.read_csv(f"{SPEC}/specificity_scores.tsv", sep="\t").set_index("pair")
syn = CALL.index[CALL.sum(1) > 0]                    # 233 synergistic pairs
C = CALL.loc[syn]
print(f"CALL matrix: {C.shape[0]} synergistic pairs x {C.shape[1]} cell states; "
      f"{int(C.values.sum())} total calls")

def tf(p, which): return scores.loc[p, which] if p in scores.index and which in scores.columns else ""

# ============================================================
# L1 — single-state private pairs (breadth == 1)
# ============================================================
breadth = C.sum(1)
private = breadth[breadth == 1].index
l1_rows = []
for ct in C.columns:
    on = C.index[C[ct].values]                       # every pair on in this state
    priv = [p for p in on if p in set(private)]      # pairs on ONLY here
    l1_rows.append(dict(cell_state=ct, lineage=CT_LIN.get(ct, ""),
                        n_pairs_on=len(on), n_private=len(priv),
                        private_pairs=";".join(f"{tf(p,'tfA')}:{tf(p,'tfB')}" for p in priv),
                        all_pairs=";".join(f"{tf(p,'tfA')}:{tf(p,'tfB')}" for p in on)))
l1 = pd.DataFrame(l1_rows)
l1.to_csv(f"{OUT}/L1_per_state_pairs.tsv", sep="\t", index=False)
print(f"\nL1 per-state: {len(private)} pairs private to a single state")
print(l1[["cell_state", "lineage", "n_pairs_on", "n_private"]].to_string(index=False))

# ============================================================
# L2 — supervised: known-lineage rollup (CALL-based)
# a pair is 'on across lineage L' if called in a MAJORITY of L's states (>=50%, >=2 states)
# ============================================================
lin_states = {}
for ct in C.columns:
    lin_states.setdefault(CT_LIN.get(ct, "NA"), []).append(ct)
l2_rows, lin_pair_sets = [], {}
for L, sts in lin_states.items():
    if len(sts) < 1: continue
    frac = C[sts].mean(1)                             # fraction of L's states each pair is on in
    thr = 0.5 if len(sts) >= 2 else 1.0
    on = frac[(frac >= thr) & (C[sts].sum(1) >= min(2, len(sts)))].index
    lin_pair_sets[L] = set(on)
    l2_rows.append(dict(lineage=L, n_states=len(sts), states=";".join(sts), n_pairs=len(on),
                        pairs=";".join(f"{tf(p,'tfA')}:{tf(p,'tfB')}" for p in on)))
l2 = pd.DataFrame(l2_rows).sort_values("n_pairs", ascending=False)
l2.to_csv(f"{OUT}/L2_per_lineage_pairs.tsv", sep="\t", index=False)
print(f"\nL2 per-lineage (majority-on):")
print(l2[["lineage", "n_states", "n_pairs"]].to_string(index=False))

# ============================================================
# L3 — discovered groups of cell states (unsupervised)
# ============================================================
# (a) state x state similarity = Jaccard over the pairs each pair-of-states co-shares
S = C.columns.tolist(); n = len(S)
J = np.eye(n)
for i in range(n):
    ai = set(C.index[C[S[i]].values])
    for j in range(i + 1, n):
        aj = set(C.index[C[S[j]].values])
        u = len(ai | aj)
        J[i, j] = J[j, i] = (len(ai & aj) / u) if u else 0.0
Jdf = pd.DataFrame(J, index=S, columns=S)
Jdf.to_csv(f"{OUT}/L3_state_jaccard.tsv", sep="\t")

# cluster states on 1 - Jaccard; pick k by silhouette
Dist = 1 - J
np.fill_diagonal(Dist, 0.0)
Lk = linkage(squareform(Dist, checks=False), method="average")
sil = {}
for k in range(2, min(9, n)):
    lab = fcluster(Lk, k, criterion="maxclust")
    if 1 < len(set(lab)) < n:
        try: sil[k] = float(silhouette_score(Dist, lab, metric="precomputed"))
        except Exception: pass
kbest = max(sil, key=sil.get) if sil else 3
state_grp = pd.Series(fcluster(Lk, kbest, criterion="maxclust"), index=S, name="group")
print(f"\nL3 discovered state groups: k={kbest} (silhouette {sil.get(kbest, float('nan')):.3f})")

# (b) majority-vote pair modules per discovered state group
l3_rows, grp_pair_sets = [], {}
for g, sub in state_grp.groupby(state_grp):
    sts = sub.index.tolist()
    frac = C[sts].mean(1)
    thr = 0.5 if len(sts) >= 2 else 1.0
    on = frac[(frac >= thr) & (C[sts].sum(1) >= min(2, len(sts)))].index
    grp_pair_sets[g] = set(on)
    lins = pd.Series([CT_LIN.get(s, "") for s in sts]).value_counts()
    l3_rows.append(dict(group=int(g), n_states=len(sts), states=";".join(sts),
                        dominant_lineages=";".join(f"{k}({v})" for k, v in lins.items()),
                        n_module_pairs=len(on),
                        module_pairs=";".join(f"{tf(p,'tfA')}:{tf(p,'tfB')}" for p in on)))
l3 = pd.DataFrame(l3_rows).sort_values("group")
l3.to_csv(f"{OUT}/L3_state_groups.tsv", sep="\t", index=False)
print(l3[["group", "n_states", "dominant_lineages", "n_module_pairs"]].to_string(index=False))

# (c) discovered state-group  x  known lineage crosstab (coherence)
cross = pd.crosstab(pd.Series([CT_LIN.get(s, "") for s in S], index=S, name="lineage"),
                    state_grp)
cross.to_csv(f"{OUT}/L3_group_vs_lineage.tsv", sep="\t")
print("\nL3 discovered-group x known-lineage:\n", cross.to_string())

# (d) formal spectral bicluster (secondary) — guarded
bic = None
try:
    from sklearn.cluster import SpectralCoclustering
    M = C.loc[C.sum(1) > 0, C.sum(0) > 0].astype(float)
    nb = min(kbest, M.shape[1] - 1)
    sc = SpectralCoclustering(n_clusters=nb, random_state=0).fit(M.values)
    bic = pd.DataFrame(dict(cell_state=M.columns, bicluster=sc.column_labels_)).sort_values("bicluster")
    bic.to_csv(f"{OUT}/L3_spectral_biclusters_states.tsv", sep="\t", index=False)
    pd.DataFrame(dict(pair=M.index, bicluster=sc.row_labels_)).to_csv(
        f"{OUT}/L3_spectral_biclusters_pairs.tsv", sep="\t", index=False)
    print(f"\nspectral coclustering: {nb} biclusters (states):\n", bic.to_string(index=False))
except Exception as e:
    print(f"\nspectral coclustering skipped: {e}")

# ---- coherence summary ----
def jac(a, b): return len(a & b) / len(a | b) if (a | b) else float("nan")
# best-matching lineage for each discovered group
matches = {}
for g, ps in grp_pair_sets.items():
    matches[int(g)] = {L: round(jac(ps, s), 3) for L, s in lin_pair_sets.items()}
summ = dict(n_pairs=int(C.shape[0]), n_states=int(C.shape[1]),
            n_private=int(len(private)), k_state_groups=int(kbest),
            silhouette=round(float(sil.get(kbest, float("nan"))), 3),
            group_vs_lineage_jaccard=matches)
json.dump(summ, open(f"{OUT}/coherence.json", "w"), indent=2)

# ---- figure: state x state Jaccard clustermap ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
cg = sns.clustermap(Jdf, row_linkage=Lk, col_linkage=Lk, cmap="magma",
                    figsize=(9.5, 9.5), xticklabels=True, yticklabels=True,
                    cbar_kws={"label": "Jaccard of co-shared synergistic pairs"})
cg.ax_heatmap.set_xlabel("cell state"); cg.ax_heatmap.set_ylabel("cell state")
cg.fig.suptitle(f"Cell states grouped by shared synergy programs (k={kbest})", y=1.01)
cg.savefig(f"{FIG}/state_shared_synergy_jaccard.png", dpi=135, bbox_inches="tight"); plt.close()

print(f"\nwrote specificity_groups/ tables + figures/state_shared_synergy_jaccard.png")
