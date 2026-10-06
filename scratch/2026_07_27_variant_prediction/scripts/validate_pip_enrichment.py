#!/usr/bin/env python
"""Model-validation panel: do high model scores enrich for high-PIP (likely-causal) variants?

Port of ChromBetaNet's credset_prediction.py, adapted to our aggregated per-trait matrices.
For each (trait, group), treats -log10(IES p) as the model score and asks whether it ranks
fine-mapped high-PIP variants above the rest: auPRC vs a PIP>=thresh label, plus Fisher
odds-ratio / lift at top-percentile cutoffs. Written for the ChromBPNet IES matrices; the
MAGIC traits carry only pseudo-PIP (LD-proxy), so read their numbers with that caveat.

Input : {outdir}/{trait}.chrombpnet.ies_pval.tsv  (PIP col + one IES-pval col per group)
Output: {outdir}/pip_enrichment.tsv               (trait, group, PIP_thresh, auPRC, random_auPRC,
                                                    top_pct, odds_ratio, pvalue, lift)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score

GROUPS = ["DE", "ENP_phase1", "FB_FLT1", "PFG1", "PFG2", "PGT1", "PGT2", "PGT3",
          "PP1", "PP2", "SC_delta_GHRL", "early_ENP", "early_SC_EC", "early_SC_alpha",
          "early_SC_beta", "exocrine", "late_ENP", "late_SC_EC", "late_SC_alpha",
          "late_SC_beta", "liver", "proliferating_endocrine"]
EPS = 1e-300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_dir", required=True, help="dir with {trait}.chrombpnet.ies_pval.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pip_thresholds", nargs="+", type=float, default=[0.9, 0.5, 0.1])
    ap.add_argument("--percentile_cutoffs", nargs="+", type=float, default=[0.001, 0.005, 0.01])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.matrix_dir, "*.chrombpnet.ies_pval.tsv"))):
        trait = os.path.basename(path).split(".")[0]
        df = pd.read_csv(path, sep="\t")
        if "PIP" not in df.columns:
            print(f"  [skip] {trait}: no PIP column")
            continue
        pip = pd.to_numeric(df["PIP"], errors="coerce").fillna(0.0)
        groups = [g for g in GROUPS if g in df.columns]

        for g in groups:
            score = -np.log10(pd.to_numeric(df[g], errors="coerce").clip(lower=EPS))
            ok = score.notna()
            s, p = score[ok].to_numpy(), pip[ok].to_numpy()
            if len(s) == 0:
                continue
            for pth in args.pip_thresholds:
                label = (p >= pth).astype(int)
                if label.sum() == 0 or label.sum() == len(label):
                    continue
                auprc = average_precision_score(label, s)
                random_auprc = label.mean()
                order = np.argsort(-s)
                s_sorted, lab_sorted = s[order], label[order]
                for pct in args.percentile_cutoffs:
                    top_n = max(int(len(s) * pct), 1)
                    A = lab_sorted[:top_n].sum(); B = top_n - A
                    C = lab_sorted[top_n:].sum(); D = (len(s) - top_n) - C
                    orr, pval = fisher_exact([[A, B], [C, D]], alternative="greater")
                    precision = A / top_n
                    lift = precision / random_auprc if random_auprc > 0 else np.nan
                    rows.append(dict(trait=trait, group=g, PIP_thresh=pth, auPRC=round(auprc, 4),
                                     random_auPRC=round(float(random_auprc), 5), top_pct=pct,
                                     odds_ratio=round(float(orr), 3), pvalue=float(pval),
                                     lift=round(float(lift), 3)))
        print(f"  {trait}: {len(groups)} groups scored")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "pip_enrichment.tsv"), sep="\t", index=False)
    print(f"wrote pip_enrichment.tsv ({len(out)} rows)")


if __name__ == "__main__":
    main()
