#!/usr/bin/env python
"""Aggregate per-group variant scores into per-trait wide matrices + a compact tensor.

Consumes:
  - ChromBPNet: results/chrombpnet/{group}/{group}.variant_scores.tsv  (one per group)
  - Peak-reg:   results/peakreg/peakreg.variant_scores.tsv             (wide: id + 22 delta cols)
  - inputs/credible_sets.tsv  (long variant x trait annotation, canonical id = chr:end:a1:a2)

Produces (in --outdir):
  {trait}.chrombpnet.logfc.tsv          variants(annotated) x 22 groups   (log2FC)
  {trait}.chrombpnet.ies_pval.tsv       variants(annotated) x 22 groups   (abs_logfc_x_jsd.pval)
  {trait}.peakreg.delta.tsv             variants(annotated) x 22 groups   (ALT-REF)
  variant_scores.h5                     tensor: variant x 22 groups x {chrombpnet_logfc, chrombpnet_ies_pval, peakreg_delta}
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

GROUPS = ["DE", "ENP_phase1", "FB_FLT1", "PFG1", "PFG2", "PGT1", "PGT2", "PGT3",
          "PP1", "PP2", "SC_delta_GHRL", "early_ENP", "early_SC_EC", "early_SC_alpha",
          "early_SC_beta", "exocrine", "late_ENP", "late_SC_EC", "late_SC_alpha",
          "late_SC_beta", "liver", "proliferating_endocrine"]


def variant_ids(df):
    """Canonical id chr:end:a1:a2, matching build_credible_sets.py. Scorer output may
    label the 1-based position column 'end' or 'pos' — accept either."""
    pos = "end" if "end" in df.columns else "pos"
    return (df["chr"].astype(str) + ":" + df[pos].astype(str)
            + ":" + df["allele1"].astype(str) + ":" + df["allele2"].astype(str))


def load_chrombpnet(cbp_dir):
    """Return {stat: DataFrame(index=id, columns=groups)} for logfc and IES p-value."""
    logfc, ies = {}, {}
    for g in GROUPS:
        fn = os.path.join(cbp_dir, g, f"{g}.variant_scores.tsv")
        if not os.path.exists(fn):
            print(f"  [warn] missing {fn}")
            continue
        d = pd.read_csv(fn, sep="\t")
        d.index = variant_ids(d)
        logfc[g] = d["logfc"]
        ies[g] = d["abs_logfc_x_jsd.pval"]
    return pd.DataFrame(logfc), pd.DataFrame(ies)


def load_peakreg(tsv):
    d = pd.read_csv(tsv, sep="\t")
    d.index = d["id"] if "id" in d.columns else variant_ids(d)
    present = [g for g in GROUPS if g in d.columns]
    return d[present]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrombpnet_dir", required=True)
    ap.add_argument("--peakreg_tsv")
    ap.add_argument("--credible_sets", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    annot = pd.read_csv(args.credible_sets, sep="\t")   # long: id, ..., trait, PIP, ...
    traits = list(annot["trait"].unique())

    cbp_logfc, cbp_ies = load_chrombpnet(args.chrombpnet_dir)
    print(f"ChromBPNet: {cbp_logfc.shape[1]} groups x {cbp_logfc.shape[0]} variants")
    pr_delta = load_peakreg(args.peakreg_tsv) if args.peakreg_tsv and os.path.exists(args.peakreg_tsv) else None
    if pr_delta is not None:
        print(f"Peak-reg:   {pr_delta.shape[1]} groups x {pr_delta.shape[0]} variants")

    # reindex all matrices onto the master ordered group list + master variant list
    master_ids = cbp_logfc.index
    cbp_logfc = cbp_logfc.reindex(columns=GROUPS)
    cbp_ies = cbp_ies.reindex(columns=GROUPS)
    if pr_delta is not None:
        pr_delta = pr_delta.reindex(index=master_ids, columns=GROUPS)

    # --- per-trait annotated wide matrices ---
    for trait in traits:
        ids = annot.loc[annot["trait"] == trait, "id"]
        meta = annot[annot["trait"] == trait].set_index("id")[["chr", "end", "allele1", "allele2", "PIP", "credset", "beta", "pval", "rsID"]]
        for name, mat in [("chrombpnet.logfc", cbp_logfc),
                          ("chrombpnet.ies_pval", cbp_ies),
                          ("peakreg.delta", pr_delta)]:
            if mat is None:
                continue
            sub = mat.reindex(index=ids)
            out = meta.join(sub).reset_index().rename(columns={"index": "id"})
            path = os.path.join(args.outdir, f"{trait}.{name}.tsv")
            out.to_csv(path, sep="\t", index=False)
        print(f"  wrote {trait} matrices ({len(ids)} variants)")

    # --- compact tensor: variant x group x layer ---
    layers = {"chrombpnet_logfc": cbp_logfc, "chrombpnet_ies_pval": cbp_ies}
    if pr_delta is not None:
        layers["peakreg_delta"] = pr_delta
    with h5py.File(os.path.join(args.outdir, "variant_scores.h5"), "w") as h:
        h.create_dataset("variant_id", data=np.array(master_ids, dtype="S"))
        h.create_dataset("groups", data=np.array(GROUPS, dtype="S"))
        for name, mat in layers.items():
            h.create_dataset(name, data=mat.reindex(index=master_ids, columns=GROUPS).to_numpy(np.float32))
    print(f"wrote variant_scores.h5  ({len(master_ids)} variants x {len(GROUPS)} groups x {len(layers)} layers)")


if __name__ == "__main__":
    main()
