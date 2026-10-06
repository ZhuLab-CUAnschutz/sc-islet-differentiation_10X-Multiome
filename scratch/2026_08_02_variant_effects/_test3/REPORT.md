# Variant-effect prediction on the 22 developmental cell types

Predicted enhancer-activity change for T1D/T2D and glycemic credible-set variants across the 22 single-task ChromBPNet cell types, with cell-type-specific summaries and a variant-by-called-hits disruption analysis. Reproduces the 2025_11_05 report on the new cell types.

## What was run

- Traits: T1D, T2D, FG, FI, HbA1c, 2hGlu. Scored once on the deduplicated variant union, split per trait for analysis.
- Significance: abs_logfc_x_jsd.pval <= 0.01 (shuffled null from variant-scorer).
- Catalog for hit calling: v1.1.
- Provenance for each stage is in `provenance/`; logs in `logs/`.

## Outputs

QC (all cell types): model_performance_overview.pdf, sc-islet-differentiation_motif_logos.pdf.

Per trait with a VEP report: T2D. Each `{trait}/vep_report_out/` holds the PCA of cell types (by stage and endocrine status), ridgeplots, boxplots, the significant-variant summary, and specificity tables.

Per-variant prediction and attribution PDFs: 0 in `variants/`.

Hits disruption: none yet. `hits_disruption.tsv` is the long variant x cell type x motif table; the cross-tabs give variant x cell type and variant x TF.

## Reproduce

```
# from the sandbox dir; scoring on carter-gpu, analysis in env eugene_tools
sbatch --array=1-22%8 --export=ALL,LIST=inputs/combined_credset.minimal.snps.chrombpnet.bed,OUTROOT=results/chrombpnet src/1_score/1_score_chrombpnet.sh
python src/2_aggregate/1_aggregate_vep.py --config config/config.yaml
python src/3_report/1_vep_report.py --config config/config.yaml
sbatch --export=ALL src/3_report/2_plot_variant.sh
python src/3_report/3_motif_and_model_qc.py --config config/config.yaml
python src/4_hits/0_significant_variants.py --config config/config.yaml
sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed src/4_hits/1_variant_shap.sh
sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed src/4_hits/2_call_variant_hits.sh
python src/4_hits/3_summarize_hits.py --config config/config.yaml
python src/5_report/build.py --config config/config.yaml
```
