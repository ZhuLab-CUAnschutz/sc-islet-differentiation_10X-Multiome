# Disease variant interpretation across developmental stages (Figure 4)
_Exploratory analysis · sandbox/2026_07_25_variant_effects · started 2026-07-25 · status: RUN-COMPLETE (partial)_

## Goal
Score T1D/T2D and glycemic (MAGIC) credible-set variants for predicted enhancer-activity change
across the 22 developmental islet cell groups, with two oracles (ChromBPNet + peak-regression),
and produce the source-data tables + a report for Figure 4: (B) cell-type-specific enrichment,
(C) clustering of variant effects, (D) representative stage-/lineage-specific loci. "Done" = frozen
credible-set table, variants×groups×oracles score tensor, enrichment/cluster/representative-loci
tables, and a readable report.

## Context
Repeat of the mature ChromBetaNet §9 variant-effect module
(`stimulated_sc-islets/manuscripts/ChromBetaNet/bin/9_variant_effect_prediction/`), which ran the
same 6 traits on ChromBetaNet's stimulation-condition groups. Here we swap in the 22 developmental
groups + add the peak-reg design oracle. Promote target: `manuscript/bin/7_variant_effects` (METHODS §8).
Earlier messy attempt: `sc-islet-differentiation_10X-Multiome/scratch/2025_11_03/`.

## Key questions
- Peak-reg significance null: dinucleotide-shuffled background to calibrate |delta| threshold (TODO).
- Confirm `finetuned_v2/28.keras` is the intended peak-reg checkpoint (best_checkpoint.txt points at v1/23).
- Per-trait vs pooled derived tables.

## Plan (phased)
### Phase 0 — inputs — DONE
- Harmonize 6 credible sets by column NAME (T2D is the 14-col outlier) on canonical id `chr:end:a1:a2`.
- `scripts/build_credible_sets.py` → `inputs/credible_sets.tsv` (56,736 variant×trait rows),
  `inputs/combined_credset.minimal.snps.chrombpnet.bed` (55,544 unique variants).
- Done when: T2D's 7,878 variants present; PIP coverage all 6 traits. ✓

### Phase 1 — ChromBPNet scoring — RUNNING
- `scripts/score_chrombpnet.sh` (variant-scorer, env `chrombpnet`, carter-gpu a30, array 1-22%4).
- Done when: 22 `{group}/{group}.variant_scores.tsv`, each 55,544 rows.
- Validated by: smoke (100 var × 2 models) before full launch.

### Phase 2 — Peak-regression scoring — CODE READY
- `scripts/score_peakreg.py` (+`.sh`): CREsted `28.keras` REF/ALT delta over 22 heads, crested `.venv`.
- Done when: `results/peakreg/peakreg.variant_scores.tsv` (55,544 × 22 delta).

### Phase 3 — Aggregate — CODE READY
- `scripts/aggregate_scores.py` → per-trait wide matrices + `variant_scores.h5` (variant×22×layers).

### Phase 4 — Derived tables — CODE READY
- `scripts/report_variant_effects.py` → `enrichment.tsv`, `group_clusters.tsv`,
  `representative_loci.tsv` + design-oracle clustermaps. Port `credset_prediction.py` for PIP-auPRC.

### Phase 5 — Report
- `REPORT.md` + `report.html` (data-forward). Promotion to manuscript is a later step.

## Guardrails (house rules)
- Validator + dev-agent gate per phase; `/grill-me` done; `/pre-flight` done (GO with fixes applied).
- No heavy local compute — all scoring on SLURM carter-gpu. Kept array throttle low (%4) while
  user's mt_full/mt_null array occupies the GPU queue.
- Evidence + provenance for every claim; canonical id join key everywhere.

## Deliverables
- `inputs/credible_sets.tsv`, `inputs/combined_credset...bed`
- `results/chrombpnet/`, `results/peakreg/`, `results/variant_scores.h5`
- `results/enrichment.tsv`, `results/group_clusters.tsv`, `results/representative_loci.tsv`, `figures/`
- `REPORT.md` + `report.html`

## Report spec
**Audience / depth:** self + collaborators; data-forward, plain language, adjustable controls, no interpretive takeaways.
**Standard sections:** TL;DR (+per-phase) · Results & interpretation · Validation scorecard · Caveats · Key paths · Reproduce.
**Analysis-specific:** per-trait enrichment tables, cluster maps, representative-loci table with attribution PDFs.
**Headline question:** Which disease variants have stage-/cell-type-specific predicted regulatory effects during islet differentiation?

## Run log
- Phase 0 — DONE · 55,544 unique variants, T2D trap handled · `inputs/`
- Phase 1 — smoke queued (jobs 13315979) behind user's GPU array
- Phase 2 — smoke queued (job 13316043); scorer code written
- Phases 3-4 — code written + syntax-checked in `eugene_tools`
