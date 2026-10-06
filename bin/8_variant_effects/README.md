# 8_variant_effects: variant-effect prediction across the 22 cell types

Score T1D/T2D and glycemic (MAGIC) credible-set variants for predicted enhancer-activity change
across the 22 single-task ChromBPNet models, report cell-type-specific effects, and ask which
called motif hits each variant disrupts.

Paths and parameters are declared once: Python stages read `config/config.yaml`, shell stages read
`paths.sh`. GPU scoring is separate from CPU analysis, so the report re-runs without re-scoring.
Each stage can be re-run safely and writes a `provenance/` JSON; logs mirror `src/` under `logs/`.

## Run order

Submit from this directory. Scoring and SHAP need the `chrombpnet` env (TensorFlow) on
`carter-gpu`; analysis runs in `eugene_tools`. Outputs go to the work dir set by `sandbox:` in
`config.yaml` (default `results/8_variant_effects/`), so the repo holds only code and config.

| # | Stage | Where | Entry point |
|---|---|---|---|
| 1 | Score variants | GPU array | `sbatch --array=1-22%4 --export=ALL,LIST=inputs/combined_credset.minimal.snps.chrombpnet.bed,OUTROOT=results/chrombpnet src/1_score/1_score_chrombpnet.sh` |
| 2 | Aggregate per trait | CPU | `src/2_aggregate/1_aggregate_vep.py` |
| 3 | Report figures | CPU | `src/3_report/{1_vep_report,2_plot_variant,3_motif_and_model_qc,4_group_specificity,5_variant_celltype_contrast}.py` |
| 4 | Variant-centric hits | GPU → CPU | `src/4_hits/{1_variant_shap.sh,2_call_variant_hits.sh,3_summarize_hits.py}`, then `4_standing_hit_overlaps.py` |
| 5 | Variant browser | mixed | `src/3_report/{6_rank_examples,8_select_browser_variants}.py` → `src/3_report/9_variant_pages.sh` → `src/5_report/build_variant_pages.py` |
| 6 | Deliverable | CPU | `src/5_report/build.py` |

A smoke run is the same stage 1 command with `LIST=inputs/smoke_100.bed` and a `--array=1-2%2`.

Stage 4 re-calls motifs on each variant's own SHAP (ref and alt), then stage 4.4 overlaps the
variants with the standing genome-wide v1.1 catalog hits. In the browser pages, TF labels are
catalog-rooted `ST##:TF`, the headline motif is the one called in the strongest cell type(s), and
panels are ordered by `|logfc|`. Re-rendering after a style change is CPU-only
(`src/3_report/9_variant_pages_render.sh`), since the GPU work is cached in `intermediates.npz`.

## Inputs

- Credible sets and the combined SNP bed, under the work dir's `inputs/` (frozen).
- The 22 single-task models at `model_root/<cell_type>/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5`,
  plus hg38 and its chrom sizes.
- v1.1 catalog: `metadata.tsv`, `catalog.meme`, logos, and
  `cluster/clustered_motifs.modisco.h5` for the hit caller.
- Cell-type metadata and the lineage / dev-stage / endocrine facets from the repo `config/`.
- The Kundaje-lab [variant-scorer](https://github.com/kundajelab/variant-scorer) checkout
  (`SCORER_DIR`), which supplies `variant_scoring.py`, `variant_shap.py` and `hitcaller_variant.py`.

Traits and the significance call are set in `config.yaml`. The default call is
`abs_logfc_x_jsd.pval <= 0.01` in at least one cell type; the value actually used by a run is
recorded in its provenance JSON.

## Gotchas

- **Cell-group order is `LC_ALL=C`.** The scoring and SHAP arrays map `--array` index to cell group
  by position. The group list is read from the shared metadata and sorted with `LC_ALL=C`, which
  reproduces the original order; a locale-aware sort silently reorders it and would score the wrong
  model for a given index.
- **The hit caller needs a matching finemo.** Stage 4.2 runs under `HITCALL_ENV` but puts
  `FINEMO_BIN` first on `PATH`, because that build matches its polars version while the analysis
  env's older editable checkout does not.
- **Scoring is the expensive half.** Stages 2 onward re-run from the stored
  `variant_scores.tsv`, so iterate on the report without re-scoring.
- **Significance is a starting default, not a result.** `pval_threshold` in `config.yaml` is a
  knob; check the provenance JSON for what a given run used.

## Layout

`config/` parameters; `paths.sh` cluster paths for the shell stages; `src/common/` shared helpers
(config loading, path resolution, score IO, provenance, plotting, metadata); `src/N_*/` numbered
stages. The work dir holds `inputs/`, `results/`, `figures/`, `deliverable/`, `logs/` and
`provenance/`.

`METHODS.md` is the methods write-up for this pipeline.

## Provenance

Promoted from `scratch/2026_08_02_variant_effects`. Changes while promoting: cluster paths moved
out of the four shell stages into `paths.sh`; the hand-maintained 22-cell-group arrays replaced by
a read of the shared metadata; cell-type metadata and facets now resolve from the repo `config/`
via a new `io.config_dir()` (the sandbox kept its own copies, which differed from the repo only in
one column header); and a stale duplicate `src/3_report/metadata.py` removed, since every stage
imports `common.metadata`.
