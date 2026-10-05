#!/usr/bin/env python
"""Step 3 - zero-shot Decima baseline (the POC floor). CPU-only.

decima 0.7.2 ships PRECOMPUTED predictions: DecimaResult.load().anndata has
layers['preds'] = predicted expression for all 8,856 atlas tracks x 18,457 genes, plus
var['dataset'] in {train,val,test}. So the floor needs no GPU and no inference:
  1. map our 22 cell types -> atlas tracks (state_mapping.tsv), average their preds,
  2. restrict to Decima's held-out TEST genes (leakage-free) that we also measured,
  3. compare to our pseudobulk, write figures + scorecard rows.

KEY CAVEAT (see state_mapping.tsv): the atlas collapses islet endocrine cells into one
'enteroendocrine cell' label, so our beta/alpha/EC map to identical tracks -> zero-shot
across-track specificity among them is ~0 by construction. That is the motivation for FT.
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/carter/users/aklie/data/decima_cache")
os.environ.setdefault("HF_HUB_CACHE", "/carter/users/aklie/data/decima_cache/hub")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, ".."))
from _utils import (  # noqa: E402
    across_gene_correlation, across_track_correlation, align, cpm_log1p,
    fig_across_gene_scatter, fig_across_track_violin, summarize, update_scorecard,
)

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
OUT_DIR = f"{DATA_DIR}/results/5_expression_models"
FIG_DIR = f"{OUT_DIR}/figures/step3"
PB_PATH = f"{DATA_DIR}/results/2_process_data/pseudobulk_expression.h5ad"
SCORECARD = f"{OUT_DIR}/scorecard.csv"
MAPPING = os.path.join(HERE, "state_mapping.tsv")
DESIGN = ["late_SC_beta", "late_SC_alpha", "late_SC_EC"]
HEALTHY = {"healthy", "normal", "na", "nan", "none", ""}


def build_pred_matrix(A, mapping):
    """Average matched atlas-track preds per our cell type -> (our types x genes)."""
    preds = pd.DataFrame(np.asarray(A.layers["preds"]), index=A.obs_names, columns=A.var_names)
    obs = A.obs
    disease_l = obs["disease"].astype(str).str.lower() if "disease" in obs else None
    rows, provenance = {}, []
    for _, m in mapping.iterrows():
        sel = obs["cell_type"].astype(str) == str(m["decima_cell_type"])
        for key, col in [("tissue_query", "tissue"), ("organ_query", "organ")]:
            q = m.get(key)
            if isinstance(q, str) and q.strip() and col in obs:
                terms = [t.strip().lower() for t in q.split("|")]
                sel &= obs[col].astype(str).str.lower().apply(lambda x: any(t in x for t in terms))
        if disease_l is not None and (sel & disease_l.isin(HEALTHY)).any():
            sel &= disease_l.isin(HEALTHY)  # prefer healthy/NA tracks when available
        idx = obs.index[sel]
        n = len(idx)
        provenance.append({"our_cell_type": m["our_cell_type"], "n_tracks": n,
                           "example_tracks": ",".join(map(str, idx[:3]))})
        rows[m["our_cell_type"]] = preds.loc[idx].mean(axis=0) if n else pd.Series(np.nan, index=preds.columns)
        print(f"  {m['our_cell_type']:24s} <- {n} tracks")
    return pd.DataFrame(rows).T, pd.DataFrame(provenance)


def fig_similarity_heatmap(pred, meas, path):
    p, m = align(pred, meas)
    tr = list(m.index)
    S = np.full((len(tr), len(tr)), np.nan)
    for i, ti in enumerate(tr):
        for j, tj in enumerate(tr):
            x, y = m.loc[ti].values, p.loc[tj].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 3:
                S[i, j] = np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(S, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels(tr, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tr))); ax.set_yticklabels(tr, fontsize=6)
    ax.set_xlabel("predicted track"); ax.set_ylabel("measured track")
    ax.set_title("Measured vs zero-shot-predicted track similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # our measured pseudobulk -> log1p CPM
    pb = ad.read_h5ad(PB_PATH)
    meas_counts = pd.DataFrame(np.asarray(pb.X), index=pb.obs_names, columns=pb.var_names)

    # Decima metadata anndata (cached): preds layer + test split
    A = DecimaResult_load()
    test_genes = list(A.var_names[A.var["dataset"].astype(str) == "test"])
    print(f"Decima test genes: {len(test_genes)}")

    mapping = pd.read_csv(MAPPING, sep="\t", comment="#")
    pred_all, provenance = build_pred_matrix(A, mapping)
    provenance.to_csv(f"{OUT_DIR}/zeroshot_track_provenance.csv", index=False)

    # common genes: Decima test ∩ our pseudobulk ∩ atlas
    common = [g for g in test_genes if g in meas_counts.columns and g in pred_all.columns]
    print(f"common test genes (in our target too): {len(common)}")
    pred = pred_all[common]
    meas = cpm_log1p(meas_counts)[common]

    ag = across_gene_correlation(pred, meas)
    at = across_track_correlation(pred, meas)
    ag.to_csv(f"{OUT_DIR}/zeroshot_across_gene.csv")
    at.to_csv(f"{OUT_DIR}/zeroshot_across_track.csv")

    fig_across_gene_scatter(pred, meas, DESIGN, f"{FIG_DIR}/across_gene_design_targets.png",
                            label="Decima zero-shot")
    fig_across_track_violin(at, f"{FIG_DIR}/across_track_violin.png", title="Zero-shot specificity")
    fig_similarity_heatmap(pred, meas, f"{FIG_DIR}/track_similarity_heatmap.png")

    update_scorecard(SCORECARD, "zeroshot_decima", summarize(pred, meas))

    # mean-expression baseline (exposes whether specificity is learned at all)
    gene_mean = meas.mean(axis=0)
    mean_pred = pd.DataFrame(np.tile(gene_mean.values, (meas.shape[0], 1)),
                             index=meas.index, columns=meas.columns)
    update_scorecard(SCORECARD, "mean_expression", summarize(mean_pred, meas))

    print("\n=== per-track across-gene r (zero-shot) ===")
    print(ag.sort_values(ascending=False).to_string())
    print(f"\nScorecard -> {SCORECARD.replace('.csv', '.png')}")
    print(pd.read_csv(SCORECARD).to_string(index=False))


def DecimaResult_load():
    from decima import DecimaResult
    return DecimaResult.load().anndata


if __name__ == "__main__":
    main()
