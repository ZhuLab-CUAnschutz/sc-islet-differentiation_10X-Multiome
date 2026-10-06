# Methods

Variant-effect prediction for T1D/T2D and glycemic credible-set variants across the 22 single-task ChromBPNet
cell types, with cell-type-specific summaries and a variant-by-called-hits disruption analysis. Every parameter
below is read from `config/config.yaml`; every stage writes a `provenance/` JSON (parameters, tool versions,
variant-scorer git sha, seed, ref build, catalog version) and a log under `logs/`.

## Inputs

- **Credible sets** (`inputs/credible_sets.tsv`): 6 traits, one row per variant x trait, with PIP. Traits: T1D
  (Chiou 2021), T2D (DIAMANTE multi-ancestry), and MAGIC FG, FI, HbA1c, 2hGlu. Canonical variant id is
  `chr:end:allele1:allele2` on hg38. This set matches the previous run exactly (T2D identical by coordinate id;
  T1D identical by rsID, 43,074 shared).
- **Scoring list** (`inputs/combined_credset.minimal.snps.chrombpnet.bed`): the 55,544 unique variants across all
  traits, scored once (variants shared across traits are not re-scored).
- **Models**: 22 single-task ChromBPNet bias-corrected models,
  `results/3_single_task_models/models/{cell_type}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5`.
- **Called-hits catalog**: motif catalog v1.1 (`results/3_single_task_models/motifs/v1.1`): clustered modisco h5,
  70 pos patterns, `metadata.tsv` with TF matches, `catalog.meme` (74 named motifs incl. sub-splits).

## Stage 1: ChromBPNet variant scoring

`src/1_score/1_score_chrombpnet.sh` runs the kundajelab **variant-scorer** (`variant_scoring.py`, env
`chrombpnet`, one SLURM array task per cell type on a carter-gpu a30). Each model scores all 55,544 variants.
Per variant per cell type it predicts accessibility for the reference and alternate allele and reports:

- `logfc`: log fold-change of predicted counts (alt vs ref); the primary effect size.
- `jsd`: Jensen-Shannon divergence between the two predicted profiles (shape change).
- `abs_logfc_x_jsd`: magnitude x shape, and `abs_logfc_x_jsd.pval`: its empirical p-value against a
  **dinucleotide-shuffled null** (`--num_shuf 10`, 10 shuffled backgrounds per SNP).

Output: `results/chrombpnet/{cell_type}/{cell_type}.variant_scores.tsv` (55,544 rows).

## Stage 2: aggregation and per-trait split

`src/2_aggregate/1_aggregate_vep.py` loads all 22 score tables (joined on the canonical id), builds the union
variant x 22-cell-type `logfc` and `abs_logfc_x_jsd.pval` matrices, then subsets each to a trait's variants:

- `results/{trait}/celltype_vep_logfc.csv`, `results/{trait}/celltype_vep_pval.csv`.
- Per-variant cell-type ranking tsv `{variant}.{trait}.-log10(abs_logfc_x_jsd.pval).tsv` for named and top loci.

## Cell-type facets (developmental stage and lineage)

`config/cell_type_metadata.tsv` holds, per cell type: `grouping`, `color`, `display_order`, `day`, `endocrine`,
`dev_stage`, `lineage`. Plots color and order by four facets:

- **Cell type**: each point its own color (`color` column).
- **Endocrine status**: endocrine vs non_endocrine.
- **Developmental stage** (`dev_stage`): trajectory position, not calendar day, because a cell type spans multiple
  days. Order and colors in `config/dev_stage_metadata.tsv`. Assignment follows the differentiation schematic
  hPSC -> DE -> GT -> PFG -> PE -> ENP -> immature beta -> mature beta:
  DE=DE; posterior_gut_tube=GT; posterior_foregut=PFG; pancreatic_progenitor=PE;
  endocrine_progenitor + proliferating_endocrine=ENP; early_SC_*=SC_immature; late_SC_* + SC_delta_GHRL=SC_mature;
  exocrine/liver/FB_FLT1=non_pancreatic.
- **Lineage** (`lineage`): terminal fate. progenitor (DE,GT,PFG,PE) / endocrine_progenitor (ENP, proliferating) /
  beta / alpha / EC / delta / non_pancreatic. Order and colors in `config/lineage_metadata.tsv`.

These `dev_stage` and `lineage` assignments are **curated** (editable in one TSV), not derived from the data. The
`day` column is retained for reference but is not plotted.

## Stage 3.1: VEP report

`src/3_report/1_vep_report.py`, per trait:

- **PCA** fit on the cell-type profiles (samples = cell types, features = the trait's variant `logfc` vector), so
  each point is a cell type. Scatter colored by each facet, plus a cell-type-labeled scatter.
- **Boxplots** and **ridgeplots** (KDE) of PC1/PC2 by facet. Facet values with fewer than two cell types cannot
  form a KDE and render as a point (logged).
- **Significant-variant summary** (`sig_variant_summary.pdf`): per-cell-type counts of variants with
  `abs_logfc_x_jsd.pval <= 0.01`, as bar charts and count boxplots colored by each facet.
- **Specificity tables**: `top_variance_variants` (variance of logfc across cell types), `celltype_specific_variants`
  (z-score of the top cell type within a variant's across-cell-type distribution), `cluster_specific_variants`
  (mean logfc per grouping).

## Stage 3.2: per-variant prediction and attribution plots

`src/3_report/2_plot_variant.py` (env eugene_tools, GPU), ported from the 2025_11_03 notebook. For selected
variants (top-N per trait by minimum p-value, plus named loci), in the strongest cell type: load the model
(bpnetlite `BPNet.from_chrombpnet`), predict reference and alternate accessibility (tangermeme `predict`), compute
DeepLiftShap attributions for both alleles, call seqlets (`recursive_seqlets`) and annotate them against the v1.1
catalog (`annotate_seqlets`). Outputs the two-allele track with attribution logos and the delta version.

All TF annotations across the pipeline are rooted in the v1.1 catalog: every seqlet, called hit, and ranked
example is labeled `ST##:TF`, where `ST##` is the catalog short id and `TF` is the catalog `curator_tf` (falling
back to `best_match_tf` when a curator label is absent). The seqlet matcher supplies the match; the catalog
supplies the name. This keeps figure labels, hit tables, and browser/contrast rankings consistent with the
catalog rather than with matcher-internal motif names.

## Stage 3.3: motif wall and model QC

`src/3_report/3_motif_and_model_qc.py`: a grid of the 74 v1.1 catalog motif logos (information content, titled by
TF match), and per-cell-type ChromBPNet performance (counts Pearson/Spearman, profile median JSD) from the
single-task metrics table.

## Stage 3.4: group-specificity screen

`src/3_report/4_group_specificity.py` calls variants "specific" to each group of a chosen grouping column
(`grouping`, `lineage`, or `dev_stage`), per trait. A cell type hits a variant if `abs_logfc_x_jsd.pval <= 0.01`.
For group g:

- `n_sig_in` / `n_sig_out`: hitting cell types inside / outside g.
- `frac_in` = n_sig_in / (cell types in g); `frac_out` = n_sig_out / (cell types outside g).
- `spec_score` = frac_in - frac_out (1 = only g hits; <=0 = not specific).
- `exclusive` = n_sig_in >= 1 and n_sig_out == 0.

A variant is reported for g if `n_sig_in >= 1`; rows are ranked by exclusive, then spec_score, then max |logfc| in
g. This is a **screen, not a calibrated test**: it inherits the per-SNP shuffled-null p-value and the fixed 0.01
threshold, and does not model correlation between cell types.

## Stage 3.6: ranking the contrast examples

`src/3_report/6_rank_examples.py` ranks variants as contrast-panel examples, best first. Per variant, its top
cell type is where |logfc| is largest. A candidate is kept if it is significant there (p <= 0.01) AND sits in an
accessible region there (predicted counts above the per-trait median, so the effect lands on a real peak rather
than low signal). Candidates are ordered by |logfc| (primary) then JSD (secondary, a proxy for a clear profile-
shape change / gained or lost motif). Each row is annotated with whether the variant overlaps a called motif hit
(from Stage 4.3) and in which cell types / TFs. Writes `results/variants/example_ranking.tsv` (full) and drives
the Stage 3.5 render order; contrast filenames are `rank{NN}_...` and the report gallery shows them best-first,
after the hits section. The contrast panel title notes effect size, peak height, and hit disruption.

## Stage 3.7: clustermaps of top-variable variants

`src/3_report/7_variant_clustermaps.py` clusters the most cell-type-specific variants (default `--select
specificity`: top-N by the Gini concentration of |logfc| across the 22 cell types, gated to max |logfc| >=
`--min-effect` 0.3 so rows are visible; `--select variance` for the old top-variance behavior). The exact
variants shown are written to `clustermap_variants.tsv`. It draws hierarchically clustered heatmaps (variants x cell
types, both axes clustered) with stage / lineage / endocrine color bars on the columns, plus aggregated views
where cell types are collapsed to mean logfc per developmental stage, per lineage, and per grouping. A pooled
clustermap over the union of variants across traits is also written. Values are logfc on a diverging scale
centered at 0 (robust 98th-percentile limits). Outputs live beside the per-trait reports
(`vep_report_out/*.clustermap_by_*.pdf`) and in `results/clustermaps/`.

## Stage 3.8 + browser: variant browser deliverable

For tracking individual variants, `deliverable/browser/variant_browser.html` is a sortable/filterable table
of a curated set; clicking a variant shows its info, a logfc-across-all-22-cell-types effect bar, and its
all-cell-type figure (predicted accessibility ref/alt + attribution logos for every cell type).

- Selection (`src/3_report/8_select_browser_variants.py`, curated union, pooled across traits, default 50):
  excludes CTCF/CTCFL; prioritizes disruptions/additions of lineage-specific TF binding sites (a curated
  islet/developmental TF list: FOXA1/2, HNF1A/B, HNF4G, HNF6, GATA2, PAX6, NKX6-1, ISL1, RFX2/4/6, NEUROD1,
  MAFB, OTX2, SOX9, LMX1A, ...), then high-PIP causal, most cell-type-specific, and largest effects; caps per
  disrupted TF for motif variety; drops `broad` scope (ubiquitous-factor sites). `direction` = loss (logfc<0,
  disrupted motif) or gain (logfc>0, created motif). Writes `results/variants/browser_variants.tsv`.
- All-cell-type figures (`5_variant_celltype_contrast.py --all-celltypes`): every cell type in trajectory
  order; attribution panels zoom to +/-60 bp around the variant and label only the variant-overlapping seqlet;
  accessibility auto-scaled per cell type. GPU.
- Browser build (`src/5_report/build_variant_browser.py`): rasterizes each figure into `browser/allct/` and
  writes the HTML (effect bars inlined; large figures referenced from `allct/`, so keep the folder together).

## Stage 4: variant-by-called-hits disruption

Which credible-set variants sit in a called motif hit, and in which cell types. Significant variants only.

- **4.0** `0_significant_variants.py`: variants with `pval <= 0.01` in at least one cell type (2,549 variants) ->
  `inputs/significant_variants.bed`.
- **4.1** `1_variant_shap.sh` (env chrombpnet, GPU array): variant-scorer `variant_shap.py` computes counts-head
  DeepLiftShap for the significant variants per cell type.
- **4.2** `2_call_variant_hits.sh`: variant-scorer `hitcaller_variant.py` runs finemo `extract-regions-chrombpnet-h5`
  then `call-hits` against the v1.1 clustered modisco h5, keeping hits that overlap the central variant. Run with
  the `finemo` build from env `finemo_gpu` (its finemo matches its polars; the eugene_tools editable checkout does
  not), under eugene_tools python for pandas.
- **4.3** `3_summarize_hits.py`: maps each hit back to its variant (`peak_id % n_variants`; peak_id >= n is the
  alternate allele) and to a TF. finemo emits modisco pattern names (`pos_patterns.pattern_i`); the clustered h5
  has one pattern per base catalog motif, so `pattern_i` maps positionally to the i-th base `ST##` (sub-splits
  collapsed), then to its TF via `metadata.tsv`. Outputs the long `hits_disruption.tsv`, cross-tabs
  (variant x cell type, variant x TF), and a per-cell-type disruption figure. The pattern -> ST## map is a
  documented positional assumption; the raw `motif_name` is retained.

## Cell-type-specificity of effects (ad hoc)

For a single variant, specificity is summarized as the z-score of its strongest cell type's logfc within the
variant's across-cell-type distribution (max ~sqrt(n_ct-1) ~ 4.6 for 22 cell types; z near that maximum means the
effect is essentially in one cell type), combined with the effect magnitude and the number of cell types it is
significant in.

## Variant coordinates in figures (correctness check)

Figure code builds sequences with `common/variant_tracks.allele_sequences`: the variant_id / BED `end` is
1-based and pyfaidx is 0-based, so the variant is centered at `end-1`, and both allele1 and allele2 are forced
into the center (matching variant-scorer, which reads the BED 0-based `start`). This is validated: with `end-1`
centering, bpnetlite reproduces variant-scorer's counts exactly. A guard `check_coordinates()` runs before each
figure batch and hard-fails if fewer than 90% of variants have `genome[end-1] == allele1` (about 95.6% match;
the rest are strand/annotation-flipped and warn only). Scoring, matrices, hits, clustermaps, and rankings come
from variant-scorer and are independent of the figure coordinate handling.

## Environments and reproducibility

- Scoring and SHAP: conda `chrombpnet` (TensorFlow). variant-scorer additionally needs `pybedtools` (installed).
- Analysis and plotting: conda `eugene_tools` (pandas, scikit-learn, seaborn, logomaker, tangermeme, bpnetlite).
- finemo hit calling: `finemo` binary from `finemo_gpu`.
- Seeds are threaded from `config.yaml` (`seed: 1234`). Scripts are idempotent (a stage overwrites only its own
  outputs) and re-runnable from committed code + config + the frozen inputs.

## Caveats

- The shuffled-null p-value uses `--num_shuf 10` (scales with the variant set), not a fixed 1M-shuffle background.
- The 0.01 significance threshold is a fixed default, not calibrated per trait or cell type.
- `dev_stage` and `lineage` are curated biological assignments; edit `config/*_metadata.tsv` to change them.
- The finemo pattern -> ST## -> TF mapping is a positional assumption between the modisco h5 and the catalog.
- Group-specificity is a screen; treat exclusivity and spec_score as ranking aids, not significance.
