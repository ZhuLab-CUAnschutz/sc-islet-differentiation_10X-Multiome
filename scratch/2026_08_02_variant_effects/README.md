# Variant-effect prediction on the 22 developmental cell types

Score T1D/T2D and glycemic (MAGIC) credible-set variants for predicted enhancer-activity change across the 22
single-task ChromBPNet models, and report cell-type-specific effects. Reproduces the `2025_11_05` report
(`scratch/2025_11_03`) on the new cell types, and adds a variant-by-called-hits disruption analysis (v1.1 catalog).

Paths and parameters are declared once in `config/config.yaml`. Raw inputs in `inputs/` are frozen. GPU scoring is
separate from CPU analysis, so the report re-runs without re-scoring. Each stage is idempotent and writes a
`provenance/` JSON; logs mirror `src/` under `logs/`.

## Run order

Score on carter-gpu (a30); analysis in conda env `eugene_tools`; per-variant plots and SHAP need a torch/bpnetlite env.

1. `src/1_score/1_score_chrombpnet.sh` (SLURM array, 22 models x combined bed).
2. `src/2_aggregate/1_aggregate_vep.py` (per-trait VEP matrices + ranking tsvs).
3. `src/3_report/{1_vep_report,2_plot_variant,3_motif_and_model_qc,4_group_specificity,5_variant_celltype_contrast}.py`.
4. `src/4_hits/{1_variant_shap.sh,2_call_variant_hits.sh,3_summarize_hits.py}` (variant-centric
   finemo re-call on each variant's SHAP, ref+alt), then `src/4_hits/4_standing_hit_overlaps.py`
   (overlap the variants with the standing genome-wide v1.1 catalog hits `motifs/v1.1/hits/{ct}`).
5. Variant browser: `src/3_report/{6_rank_examples,8_select_browser_variants}.py` (select), then
   `src/3_report/9_variant_pages.sh` (SLURM: per-variant page figures, GPU compute cached to
   `intermediates.npz`), then `src/5_report/build_variant_pages.py` (index + one page per variant).
   All TF labels are catalog-rooted `ST##:TF`; the headline motif is the one called in the strongest
   cell type(s); panels are ordered by |logfc|. Each attribution logo has three annotation rows
   (v1.1 catalog / finemo / seqlet), showing only hits that overlap the variant. Re-render after a
   style change is CPU-only: `src/3_report/9_variant_pages_render.sh` (carter-compute, uses the cache).
6. `src/5_report/build.py` (deliverable folder + REPORT.md + report.html; preserves `browser/`).

## Layout

`config/` parameters and metadata; `inputs/` frozen raw; `src/common/` shared helpers; `src/N_*/` numbered
stages; `results/` derived; `figures/`; `deliverable/` curated dump; `logs/`; `provenance/`.
