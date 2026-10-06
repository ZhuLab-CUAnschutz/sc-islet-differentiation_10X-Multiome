#!/usr/bin/env python
"""Derived source-data tables for Fig-4 variant effects: enrichment, clustering, representative loci.

Reads variant_scores.h5 (variant x 22 groups x {chrombpnet_logfc, chrombpnet_ies_pval, peakreg_delta})
and inputs/credible_sets.tsv (trait membership + PIP), and writes, per trait:
  enrichment.tsv          per-group significant-variant counts + stage/endocrine annotation
  group_clusters.tsv      hierarchical clustering of the 22 groups on the design-oracle matrix
  representative_loci.tsv  cell-type-specific + cluster-specific variants (stage/lineage-specific effects)
plus heatmap PNGs in --figdir. Enrichment uses the ChromBPNet IES p-value; clustering &
representative loci use the peak-regression 'design oracle' (METHODS §8.4).
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# name-derived developmental ordering (built-in semantics, no curated mapping file)
STAGE_ORDER = ["DE", "PGT", "PFG", "PP", "ENP", "early_SC", "SC", "late_SC", "other"]
IES_SIG = -np.log10(0.01)      # ChromBPNet significance on -log10(IES p)
TOPK = 3000                    # top-variance variants for clustering features
K_CLUSTERS = 3
CT_Z_THRESH, CT_MARGIN_THRESH, CLUSTER_SPEC_THRESH = 2.0, 1.0, 0.5


def stage_of(g):
    if g.startswith("early_SC") or g.startswith("late_SC"):
        return g.split("_")[0] + "_SC"
    for tok in ["DE", "PGT", "PFG", "PP", "ENP", "SC"]:
        if tok in g:
            return tok
    return "other"


def endocrine_of(g):
    return "Endocrine" if any(t in g for t in ["ENP", "SC_alpha", "SC_beta", "SC_delta", "SC_EC", "endocrine"]) else "Non-endocrine"


def row_z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--credible_sets", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--figdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    annot = pd.read_csv(args.credible_sets, sep="\t")
    with h5py.File(args.h5, "r") as h:
        groups = [x.decode() for x in h["groups"][:]]
        all_ids = [x.decode() for x in h["variant_id"][:]]
        layers = {k: pd.DataFrame(h[k][:], index=all_ids, columns=groups) for k in h.keys()
                  if k not in ("variant_id", "groups")}

    ies = -np.log10(layers["chrombpnet_ies_pval"].clip(lower=1e-300)) if "chrombpnet_ies_pval" in layers else None
    design = layers.get("peakreg_delta", layers.get("chrombpnet_logfc"))  # design oracle for clustering
    stage = pd.Series({g: stage_of(g) for g in groups})
    endo = pd.Series({g: endocrine_of(g) for g in groups})

    enrichment_all, clusters_all, reploci_all = [], [], []

    for trait in annot["trait"].unique():
        ids = annot.loc[annot["trait"] == trait, "id"]
        ids = [i for i in ids if i in all_ids]
        if not ids:
            continue

        # --- enrichment: per-group significant-variant counts (ChromBPNet IES) ---
        if ies is not None:
            sub = ies.reindex(ids)
            counts = (sub >= IES_SIG).sum(axis=0)
            e = pd.DataFrame({"trait": trait, "group": groups,
                              "n_sig": counts.reindex(groups).values,
                              "n_tested": sub.notna().sum(axis=0).reindex(groups).values,
                              "stage": stage.reindex(groups).values,
                              "endocrine": endo.reindex(groups).values})
            e["frac_sig"] = e["n_sig"] / e["n_tested"].replace(0, np.nan)
            enrichment_all.append(e)

        # --- clustering of the 22 groups on the design-oracle matrix ---
        D = design.reindex(ids).dropna(how="all")
        Z = row_z(D).dropna(how="all")
        topk = Z.var(axis=1).sort_values(ascending=False).head(TOPK).index
        feats = Z.loc[topk].fillna(0)
        link = linkage(pdist(feats.T, metric="correlation"), method="average")
        cl = fcluster(link, K_CLUSTERS, criterion="maxclust")
        clusters_all.append(pd.DataFrame({"trait": trait, "group": feats.columns, "cluster": cl,
                                          "stage": stage.reindex(feats.columns).values}))

        # --- representative loci ---
        Zc = Z.loc[topk]
        # cell-type-specific
        for ct in groups:
            if ct not in Zc.columns:
                continue
            others = Zc.drop(columns=ct)
            margin = Zc[ct] - others.max(axis=1)
            hit = (Zc[ct] >= CT_Z_THRESH) & (margin >= CT_MARGIN_THRESH)
            for vid in Zc.index[hit.fillna(False)]:
                reploci_all.append({"trait": trait, "type": "cell_type_specific", "variant": vid,
                                    "group": ct, "z": round(float(Zc.loc[vid, ct]), 3),
                                    "margin": round(float(margin.loc[vid]), 3)})
        # cluster-specific: mean-Z within a cluster minus the best OTHER cluster's mean-Z
        g2c = dict(zip(feats.columns, cl))
        cluster_mean = {cid: Zc[[g for g in feats.columns if g2c[g] == cid]].mean(axis=1)
                        for cid in sorted(set(cl))}
        for cid in sorted(set(cl)):
            others = [c for c in cluster_mean if c != cid]
            if not others:
                continue
            in_c = [g for g in feats.columns if g2c[g] == cid]
            mean_in = cluster_mean[cid]
            max_out = pd.concat([cluster_mean[c] for c in others], axis=1).max(axis=1)
            spec = mean_in - max_out
            for vid in Zc.index[(spec > CLUSTER_SPEC_THRESH).fillna(False)]:
                reploci_all.append({"trait": trait, "type": "cluster_specific", "variant": vid,
                                    "group": f"cluster{cid}:" + ",".join(in_c),
                                    "z": round(float(mean_in.loc[vid]), 3),
                                    "margin": round(float(spec.loc[vid]), 3)})

        # --- heatmap ---
        if len(topk) > 1:
            cc = pd.DataFrame({"stage": stage.reindex(feats.columns).map(
                {s: c for s, c in zip(STAGE_ORDER, sns.color_palette("viridis", len(STAGE_ORDER)))}).values,
                "endocrine": endo.reindex(feats.columns).map({"Endocrine": "#d62728", "Non-endocrine": "#7f7f7f"}).values},
                index=feats.columns)
            try:
                g = sns.clustermap(feats.T, cmap="vlag", center=0, row_colors=cc,
                                   xticklabels=False, figsize=(12, 8), method="average", metric="correlation")
                g.fig.suptitle(f"{trait} — design-oracle variant effects (top {len(topk)} var)")
                g.savefig(os.path.join(args.figdir, f"{trait}.design_clustermap.png"), dpi=150)
                plt.close(g.fig)
            except Exception as ex:
                print(f"  [warn] clustermap {trait}: {ex}")

    if enrichment_all:
        pd.concat(enrichment_all).to_csv(os.path.join(args.outdir, "enrichment.tsv"), sep="\t", index=False)
    if clusters_all:
        pd.concat(clusters_all).to_csv(os.path.join(args.outdir, "group_clusters.tsv"), sep="\t", index=False)
    if reploci_all:
        pd.DataFrame(reploci_all).to_csv(os.path.join(args.outdir, "representative_loci.tsv"), sep="\t", index=False)
    print(f"wrote enrichment / group_clusters / representative_loci to {args.outdir}")


if __name__ == "__main__":
    main()
