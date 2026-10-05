#!/usr/bin/env python
"""Evaluate a trained arm on Decima's held-out TEST genes and append to the scorecard.

  python evaluate_model.py --ckpt <path> --arm decima --regime probe

Produces (into figures/step4 or step5 by regime): across-gene design-target scatters,
across-track specificity violin, marker-recovery bars; and updates the shared scorecard
row `<arm>_<regime>` alongside zero-shot Decima + mean-expression baselines.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, ".."))
sys.path.append(os.path.join(HERE, "..", "train"))
from _utils import (  # noqa: E402
    across_gene_correlation,
    across_track_correlation,
    cpm_log1p,
    fig_across_gene_scatter,
    fig_across_track_violin,
    marker_auroc,
    summarize,
    update_scorecard,
)
from data import make_loaders  # noqa: E402
from model import SeqExpr  # noqa: E402

DATA_DIR = "/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome"
OUT_DIR = f"{DATA_DIR}/results/5_expression_models"
SCORECARD = f"{OUT_DIR}/scorecard.csv"
DESIGN = ["late_SC_beta", "late_SC_alpha", "late_SC_EC"]


@torch.no_grad()
def predict_test(model, loaders):
    """Predict (22 x test_genes) for the test split."""
    model.eval().cuda()
    tracks = loaders["tracks"]
    genes = loaders["test_genes"]
    preds = []
    for x, _ in loaders["test"]:
        preds.append(model(x.cuda()).cpu().numpy())
    P = np.concatenate(preds, axis=0)  # (n_genes, 22)
    return pd.DataFrame(P.T, index=tracks, columns=genes)  # tracks x genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["decima", "borzoi"], required=True)
    ap.add_argument("--regime", choices=["probe", "lora", "full"], required=True)
    args = ap.parse_args()

    step = "step5" if args.regime in ("lora", "full") else "step4"
    fig_dir = f"{OUT_DIR}/figures/{step}"
    os.makedirs(fig_dir, exist_ok=True)
    tag = f"{args.arm}_{args.regime}"

    loaders = make_loaders(batch_size=8, num_workers=4)
    model = SeqExpr.load_from_checkpoint(args.ckpt, arm=args.arm, regime=args.regime)
    pred = predict_test(model, loaders)  # log-scale expression, tracks x genes

    # measured: log1p CPM on the same test genes/tracks
    meas = cpm_log1p(loaders["targets"].T).loc[pred.index, pred.columns]
    pred_l = np.log1p(pred)

    ag = across_gene_correlation(pred_l, meas)
    at = across_track_correlation(pred_l, meas)
    _, marker_tab = marker_auroc(pred_l)
    ag.to_csv(f"{OUT_DIR}/{tag}_across_gene.csv")
    at.to_csv(f"{OUT_DIR}/{tag}_across_track.csv")
    marker_tab.to_csv(f"{OUT_DIR}/{tag}_markers.csv", index=False)

    fig_across_gene_scatter(pred_l, meas, DESIGN,
                            f"{fig_dir}/{tag}_across_gene_design_targets.png", label=tag)
    fig_across_track_violin(at, f"{fig_dir}/{tag}_across_track_violin.png",
                            title=f"{tag} specificity")

    update_scorecard(SCORECARD, tag, summarize(pred_l, meas))
    print(f"\nUpdated scorecard -> {SCORECARD.replace('.csv', '.png')}")
    print(pd.read_csv(SCORECARD).to_string(index=False))


if __name__ == "__main__":
    main()
