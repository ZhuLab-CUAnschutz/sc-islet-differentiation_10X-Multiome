# Handoff — Figure 4: Disease variant interpretation across developmental stages

_Paused 2026-07-27 to let the syntax-marginalization (`full_allct`) a30 run finish. Resume when a30 slots free._

## TL;DR

Building the source data for **manuscript Figure 4** (T1D/T2D + glycemic credible-set variants → predicted enhancer-activity change across the 22 developmental islet cell groups, two oracles). This is a **repeat of the ChromBetaNet §9 variant-effect module** on the developmental groups.

**Status:** everything is built and validated except the full GPU scoring, which is **cancelled/paused** (a30-only, and the a30s are occupied by your own `full_allct` syntax array + others).

- ✅ Inputs frozen (55,544 variants) · ✅ all 6 scripts written + **smoke-validated on real data** · ✅ downstream aggregate/report chain **synthetic-tested E2E**.
- ⏸ Full scoring NOT run yet — both jobs cancelled (`13323485` cbp, `13323486` peakreg). **This is the only thing left to execute**, then a fast, tested downstream.

**To resume:** re-submit the two scoring jobs (commands below), wait, then run 3 downstream scripts + write the report. No new code needed.

## Where everything lives

- **NRNB sandbox (source of truth):** `/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_25_variant_effects/`
  - `scripts/` — the 6 scripts (also copied into this handoff dir)
  - `inputs/credible_sets.tsv` (frozen annotation, also copied here) + `inputs/combined_credset.minimal.snps.chrombpnet.bed` (55,544-variant scoring list)
  - `results/` (empty — scoring was cancelled), `figures/`, `slurm_logs/`, `PLAN.md`
- **Promote target (later):** `manuscript/bin/7_variant_effects` + `results/7_variant_effects` (METHODS §8, Fig 4A–D).
- **Primary template ported from:** `.../stimulated_sc-islets/manuscripts/ChromBetaNet/bin/9_variant_effect_prediction/`.
- **This local bundle:** `scripts/`, `PLAN.md`, `credible_sets.tsv`, this doc. (Scripts are regenerable-safe; big inputs/models stay on NRNB.)

## Decisions (locked with user)

- **2 oracles:** ChromBPNet (22 single-task `chrombpnet_nobias.h5`) + peak-regression multi-task CREsted design oracle (`finetuned_v2/28.keras`). Topic CNN deferred.
- **6 traits:** T1D (Chiou 2021), T2D (DIAMANTE cred99), MAGIC FG/FI/HbA1c/2hGlu. All carry PIP (MAGIC = LD-proxy pseudo-PIP).
- **Null:** scorer **default `--num_shuf 10`** (10 nulls/SNP) — NOT the ChromBetaNet template's fixed `-t 1000000` (that was ~2 h/model).
- **Enrichment** = per-group significant-variant counts; **PIP-auPRC/Fisher** kept as model validation.
- **GPU:** **a30 only** (user pref — rtx5000 too slow for the 555k-shuffle null). Downside: a30 contention → long queue wait.
- **Emphasis:** source-data tables + a readable report. Figure-panel assembly deferred.

## Resume recipe (when a30 slots are free)

All SSH via a **single persistent master** (many rapid connections trip the login-node throttle → hangs):
```
ssh -o ControlMaster=yes -o ControlPersist=900 -o ControlPath=~/.ssh/cc-master -fN hpc
alias hssh='ssh -o ControlPath=~/.ssh/cc-master hpc'
```
SLURM binaries aren't on PATH: use `/cm/shared/apps/slurm/current/bin/`. Env for downstream = conda `eugene_tools`.

**1. Score (a30).** From the sandbox dir:
```
SB=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_25_variant_effects
BED=inputs/combined_credset.minimal.snps.chrombpnet.bed
# ChromBPNet: 22-model array (scripts already request #SBATCH --gpus=a30:1)
sbatch --array=1-22%8 --export=ALL,LIST=$BED,OUTROOT=results/chrombpnet scripts/score_chrombpnet.sh
# Peak-reg: one job, all 22 heads
sbatch --export=ALL,BED=$BED,OUT=results/peakreg/peakreg.variant_scores.tsv scripts/score_peakreg.sh
```
Expect ~1.5 h/model on a30 (≈several h wall for 22 at %8); peak-reg ~15 min. Smoke first if paranoid: same commands with `LIST=inputs/smoke_100.bed`, `OUTROOT=results/chrombpnet_smoke`, array `1-2%2`.

**2. Aggregate + report (env `eugene_tools`, CPU, minutes):**
```
python scripts/aggregate_scores.py --chrombpnet_dir results/chrombpnet \
  --peakreg_tsv results/peakreg/peakreg.variant_scores.tsv \
  --credible_sets inputs/credible_sets.tsv --outdir results
python scripts/report_variant_effects.py --h5 results/variant_scores.h5 \
  --credible_sets inputs/credible_sets.tsv --outdir results --figdir figures
python scripts/validate_pip_enrichment.py --matrix_dir results --outdir results
```
Produces: `variant_scores.h5` (variant × 22 groups × {chrombpnet_logfc, chrombpnet_ies_pval, peakreg_delta}), per-trait wide matrices, `enrichment.tsv`, `group_clusters.tsv`, `representative_loci.tsv`, `pip_enrichment.tsv`, clustermap PNGs.

**3. Write** `REPORT.md` + self-contained `report.html` (house data-forward style), then optionally promote to `manuscript/bin/7_variant_effects`.

## Hard-won gotchas (all already fixed in the scripts here)

1. **bash `GROUPS` is reserved** (holds Unix GIDs) — the cell-group array is named `CELLGROUPS`. Don't rename back.
2. **`/cellar` is NOT mounted on carter compute nodes** — scorer path must be `/carter/users/aklie/opt/variant-scorer/src/variant_scoring.py`. (Memory: carter nodes mount /carter.)
3. **`28.keras` was saved by keras-2** — `score_peakreg.py` monkeypatches `keras.layers.BatchNormalization.__init__` to drop `renorm*` kwargs (keras 3.13 rejects them; `custom_objects` is bypassed for built-ins).
4. **crested venv** — call `$VENV/bin/python` directly (`tools/crested/.venv`, py3.12, keras 3.13.2); `source activate` gets shadowed by conda under SLURM.
5. **SSH** — one persistent master; never `-o ControlMaster=no` spam. Clear stale socket with `rm ~/.ssh/master-*` if it wedges.
6. **T2D schema is the outlier** (14 cols, no `variant_id`, swapped rsID) — handled: everything joins on canonical id `chr:end:allele1:allele2` built from the bed.

## Open items

- **Peak-reg significance null:** the design-oracle `delta` has no built-in p-value. For its enrichment threshold, calibrate against dinucleotide-shuffled sequences (matches your null-calibration practice). Not yet implemented.
- Confirm `finetuned_v2/28.keras` is the intended peak-reg checkpoint (`best_checkpoint.txt` points at v1/23; `finetune_params.yaml` has stale `/cellar` paths — scripts hardcode `/carter`).
- Per-trait vs pooled derived tables (currently per-trait).

## Verification (post-scoring)

- Each `results/chrombpnet/{group}/{group}.variant_scores.tsv` has 55,545 rows (header + 55,544).
- `variant_scores.h5` shape = (55,544, 22, 3 layers).
- Known islet locus shows expected effect direction; ChromBPNet vs peak-reg top-effect overlap is non-trivial (sanity, not identity).
- Cross-check enrichment/cluster tables against ChromBetaNet's `cross_group_prediction` outputs.

_Related memory: `fig4-variant-effects`, `nrnb-slurm-carter-storage`, `reports-data-forward-no-takeaways`, `syntax-pairwise-marginalization`._
