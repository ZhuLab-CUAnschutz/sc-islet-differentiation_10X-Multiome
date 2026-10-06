#!/usr/bin/env python
"""Build the frozen credible-set table + combined scoring bed for Fig-4 variant effects.

Reads the six disease/glycemic credible sets, harmonizes their (inconsistent) schemas
by *column name* — T2D lacks `variant_id` and reorders `rsID`, so a positional read
would silently corrupt it — and keys every variant on the canonical id
`chr:end:allele1:allele2` (the same convention ChromBetaNet's credset_prediction.py uses).

Outputs (to --outdir):
  credible_sets.tsv                         long table, one row per (variant, trait)
  combined_credset.minimal.snps.chrombpnet.bed   unique variants, col6 = canonical id (scoring input)
"""
import argparse
import os
import pandas as pd

VARIANT_DIR = "/carter/users/aklie/data/ref/variants"

# trait -> stem (minimal bed = stem + '.minimal.snps.chrombpnet.bed', full = stem + '.full.snps.txt')
TRAITS = {
    "T1D":   "T1D/T1D_Chiou_2021_cred_set.wGWAS_info",
    "T2D":   "T2D/T2D_DIAMANTE_multiancestry.cred99.hg38.wGWAS_info",
    "FG":    "MAGIC/FG_MAGIC_trans_ancestry_pseudo_credset.LDproxyRsq0.8.wGWAS_info",
    "FI":    "MAGIC/FI_MAGIC_trans_ancestry_pseudo_credset.LDproxyRsq0.8.wGWAS_info",
    "HbA1c": "MAGIC/HbA1c_MAGIC_trans_ancestry_pseudo_credset.LDproxyRsq0.8.wGWAS_info",
    "2hGlu": "MAGIC/2hGlu_MAGIC_trans_ancestry_pseudo_credset.LDproxyRsq0.8.wGWAS_info",
}

# columns we carry into the frozen table (read by NAME, present in all six full tables)
ANNOT_COLS = ["credset", "PIP", "beta", "pval", "rsID"]
BED_COLS = ["chr", "start", "end", "allele1", "allele2", "id"]


def canonical_id(df):
    return (df["chr"].astype(str) + ":" + df["end"].astype(str)
            + ":" + df["allele1"].astype(str) + ":" + df["allele2"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    annot_rows = []   # long: one per (variant, trait)
    bed_frames = []   # per-trait scoring beds

    for trait, stem in TRAITS.items():
        full_path = f"{VARIANT_DIR}/{stem}.full.snps.txt"
        bed_path = f"{VARIANT_DIR}/{stem}.minimal.snps.chrombpnet.bed"

        # --- annotation (header-keyed; T2D's missing variant_id / reordered rsID handled by name) ---
        full = pd.read_csv(full_path, sep="\t")
        full["id"] = canonical_id(full)
        keep = ["id", "chr", "end", "allele1", "allele2"] + [c for c in ANNOT_COLS if c in full.columns]
        sub = full[keep].copy()
        for c in ANNOT_COLS:
            if c not in sub.columns:
                sub[c] = pd.NA
        sub["trait"] = trait
        annot_rows.append(sub)

        # --- scoring bed (authoritative, already ref-matched); rewrite col6 = canonical id ---
        bed = pd.read_csv(bed_path, sep="\t", header=None,
                          names=["chr", "start", "end", "allele1", "allele2", "name"])
        bed["id"] = canonical_id(bed)
        bed_frames.append(bed[BED_COLS])
        print(f"{trait:6s} full={len(full):>6d}  bed={len(bed):>6d}")

    # frozen long annotation table
    annot = pd.concat(annot_rows, ignore_index=True)
    annot = annot[["id", "chr", "end", "allele1", "allele2", "trait"] + ANNOT_COLS]
    annot_path = os.path.join(args.outdir, "credible_sets.tsv")
    annot.to_csv(annot_path, sep="\t", index=False)

    # combined UNIQUE scoring bed (a variant shared across traits is scored once)
    combined = pd.concat(bed_frames, ignore_index=True).drop_duplicates(subset="id")
    combined = combined.sort_values(["chr", "start"])
    bed_out = os.path.join(args.outdir, "combined_credset.minimal.snps.chrombpnet.bed")
    combined.to_csv(bed_out, sep="\t", header=False, index=False)

    print(f"\ncredible_sets.tsv : {len(annot):>6d} (variant,trait) rows across {annot['trait'].nunique()} traits")
    print(f"combined bed      : {len(combined):>6d} unique variants "
          f"(from {sum(len(b) for b in bed_frames)} pre-dedup)")
    print(f"per-trait rows    : " + ", ".join(f"{t}={n}" for t, n in annot['trait'].value_counts().items()))
    print(f"PIP non-null      : " + ", ".join(
        f"{t}={annot[annot.trait==t]['PIP'].notna().sum()}" for t in TRAITS))
    print(f"wrote:\n  {annot_path}\n  {bed_out}")


if __name__ == "__main__":
    main()
