# Per-cell-type synergy tables

Single-task ChromBPNet pairwise marginalization, split by cell type. Three intended versions per
cell type (22 cell types):

| suffix | contents | status |
|---|---|---|
| `<CT>__all_pairs.tsv` | every keep-set pair (2628) scored in that CT | **PENDING** — needs all-pairs × all-CT data (coming from the multitask run; single-task scored each pair only at its one assigned primary CT) |
| `<CT>__synergy.tsv` | pairs called synergistic **in that CT** (Δ>0.504 & maxZ>5.38 vs lineage-mismatched null), sorted by Δ | done |
| `<CT>__synergy_deeplift_pass.tsv` | same, minus the 7 pairs DeepLIFT flagged as attribution-artifact (`deeplift_flag=="artifact?"`) | done |

Source: `figures/synergistic_pairs_allCT_long.tsv` (582 synergistic pairs × 22 CT, arrangement
re-optimized per CT).

## Columns
`idA,idB` motif ids · `tfA,tfB` curator TF · `delta` = ΔJ_opt − ΔS (optimal-arrangement joint minus
log-additive null) · `dJ_opt,dS` · `opt_orient,opt_gap,opt_center_center` optimal arrangement ·
`maxZ` arrangement z-score · `wilcoxon_p` joint-vs-independent signed-rank · `synergistic_in_ct,hard_in_ct` ·
`primary_ct,primary_delta` the pair's strongest CT · `deeplift_frac_in,deeplift_flag` ·
`wa,wb` motif widths · `category_*,family_*,best_match_*` curator annotation.

`delta` is computed exactly as in Liu et al. Nature 2026 (optimal arrangement over orientation × 0–200 bp
spacing vs the log-additive model). We run 1 fold (they average 5) and all 2628 pairs (they pre-filtered
138), which is why thresholds are recalibrated against a lineage-mismatched null.

Note: older `synergy__<CT>.tsv` files (prior session) are superseded by `<CT>__synergy.tsv`.
