# TF-binding syntax drives cell-type specificity in islet differentiation
### In-silico pairwise motif marginalization over the single-task ChromBPNet catalog (v1.1)

_Working report — session 2026-07-19/20. Sandbox: `sandbox/2026_07_19_syntax_pairwise_marg/`._

## Question
Do pairs of TF motifs **cooperate** (synergize beyond independent additive effects) in a
**cell-type-specific** way? Method reproduces **Liu et al., "Multiomics and deep learning dissect
regulatory syntax in human development," _Nature_ 2026** (doi 10.1038/s41586-026-10326-9; Betty B.
Liu, Kundaje/Farh/Greenleaf — the published version of bioRxiv 2025.04.30.651381) — in-silico
pairwise marginalization with ChromBPNet + tangermeme — applied to our 22 per-cell-type single-task
models and the v1.1 motif catalog.

_Note on the two "Liu et al."s:_ the **method** paper is **Betty B. Liu et al. (Nature 2026)** above.
The **β-vs-EC islet biology** used as interpretive anchor below is a *separate* paper — **Dingyu Liu
et al.**, "A stem cell knockout village reveals … a non-canonical islet cell fate in monogenic
diabetes" (bioRxiv 2025.12.23.696311). Different first-author Lius; method vs biology.

**What we did beyond the paper:** Liu et al. classified only **138 pre-filtered de-novo composite**
motif pairs and never ran an all-pairs sweep. We extend the exact method to an **unbiased 2211-pair
screen** over the islet keep-set — which is precisely why the empirical null calibration (below) was
required, and why the paper's own thresholds do not transfer unmodified.

## Method (reproduced from Liu et al. Nature 2026, adapted to our data)
For a motif pair (A,B) in a cell type's ChromBPNet model:
- **Backgrounds:** 100 (hero) / 64 (full screen) real **GC-matched inaccessible negatives**
  (`filtered.nonpeaks.bed`), 2114 bp.
- **Consensus:** argmax of the collapsed signed **CWM** (`cwms.npz`), importance-trimmed.
- **Sweep:** insert A and B (`tangermeme.ersatz.substitute`) at every **orientation** (≤4; palindrome-
  deduped) × **inner-edge gap 0–200 bp** (uniform grid). Predict Δlog-counts vs background.
- **Synergy:** ΔJ (joint) vs **ΔS = ΔA + ΔB** (log-additive). Optimal arrangement = argmax mean ΔJ.
  Wilcoxon signed-rank (joint vs independent) across backgrounds; **hard** syntax = arrangement
  Z>4 (bootstrap-checked, single fold); **soft** = (ΔJ−ΔS)>0.15 at 20–150 bp.
- **CT-specificity:** each pair's Δ across all 22 CTs, **z-scored excluding its own max CT** (de-bias).
- Single fold (`fold_0`); thresholds recalibrated against a lineage-mismatched **empirical null**.

The plan was **adversarially reviewed** before running; all blockers fixed (CWM-not-PPM consensus,
real negatives not shuffled peaks, single-fold bootstrap, empirical-null FDR, palindrome dedup,
job-index load balancing, selection-debiased specificity).

## Validation (all passed)
- **Correctness:** batched sweep reproduces the un-batched ΔJ exactly (ISL1×NKX6-1 β = 0.738).
- **Additivity control:** ΔJ → ΔS as gap → 200 bp for every pair (independent at distance).
- **Composite recovery (internal positive control):** motifs the curator independently flagged as
  Hetero/Homocomposite — FOXA2×OTX2, RFX6×NKX2-2, OTX2×GATA, MECOM×GATA — come back as the
  **strongest, hard-syntax** synergies (best Δ 1.06–1.53). Motifs annotated as composites are
  predicted to physically cooperate.

## Headline result — syntax is cell-type-specific along the β/EC axis
CT-specificity heatmap (`fig_ct_specificity_heatmap.png`): the β/endocrine co-regulator pairs
synergize **specifically in the β/α endocrine cell types**, EC pairs in **SC-EC**, progenitor
composites in the FOX/OTX/GATA endoderm cell types — surviving the own-max-CT de-biasing and lighting
up across *multiple* related CTs (not a single-CT artifact).

| syntax class | pairs (mean Δβ vs ΔEC) |
|---|---|
| **β-specific** | PDX1×NKX6-1 (0.52 / 0.09), ISL1×NKX6-1 (0.38 / 0.09), NKX6-1×NKX6-1, PAX6×NKX2-2, NKX6-1×PAX6, ISL1×PAX6, ISL1×ISL1 |
| **EC-specific** | LMX1A×RFX6 (0.32 / 0.40), LMX1A×RFX6_comp (0.24 / 0.34), MECOM×GATA |
| **cross-fate (β-motif × EC-motif)** | ISL1×LMX1A, NKX6-1×LMX1A — weak, no fate-boundary cooperativity |

This is the **Dingyu Liu et al.** (islet KO-village) β-vs-EC regulator logic (ISL1/NKX6-1/PDX1/PAX6 → β;
LMX1A/RFX6 → EC) recovered at the level of **cis-regulatory binding syntax** — cooperativity our
chromatin models predict, that their SCENIC+/KO-village approach could not resolve.

## Distance-dependent cooperativity (Fig-4k reproduction)
`fig_distance_curves.png`: sharp near-abutting peaks decaying to additive — RFX6×NKX2-2 spikes at
~5 bp (hard), NKX6-1×NKX6-1 / PDX1×NKX6-1 at 0–1 bp, LMX1A×RFX6 (EC) at ~4 bp.

## Null calibration (vindicates the adversarial review)
126 lineage-mismatched pairs (DE, early_SC_beta, early_SC_EC). At the **paper's own thresholds**,
**54%** of mismatched null pairs would be falsely called synergistic (Δ>0.15) and **14%** hard
(Z>4) — i.e. the paper's cutoffs are invalid for our single-fold / all-pairs regime. Empirical
95th-percentile null thresholds (~5% FDR) applied throughout: **Δ > 0.504**, **Z > 5.38**. (Even
mismatched pairs average Δ≈0.17: inserting any two motifs near an accessible center is mildly
non-additive; the null absorbs this baseline.)

## Genome-wide census — full keep-set sweep (2211 pairs, primary CT)
`fig_full_scatter.png` (ΔJ vs ΔS), `fig_full_census.png`, `fig_full_spacing.png`,
`full_top_synergy.tsv`, `full_calls.tsv`:
- **477 / 2211 (22%) synergistic**, **43 hard-syntax**, **433 soft-syntax** at 5% FDR vs null.
- Bulk of pairs sit on the additive diagonal; a well-separated synergistic tail. Strongest synergies:
  FOXA2×OTX2, TEAD homo/heterotypic pairs, RFX6×FOXA2 (EC), MECOM×FOXA2 — high-effect
  progenitor/lineage composites, most at close (<25 bp center-to-center) spacing.

## Runs (all complete)
- **Candidate/hero** (22 pairs × 22 CTs, full fidelity, n=100, 1 bp grid) — DONE.
- **Null** (42 pairs × 3 CTs) — DONE.
- **Full** (2211 keep-set pairs, primary-CT, n=64, uniform 2 bp grid) — DONE.

## Figures / tables (in `figures/`)
`fig_ct_specificity_heatmap.png`, `fig_distance_curves.png`, `fig_scatter_early_SC_beta.png`
(ΔJ vs ΔS), `fig_syntax_classes_*.png` (hard/soft/none barplot), `fig_spacing_hist.png`,
`syntax_summary_table.tsv`, `candidate_calls.tsv`, `ct_specificity_zmatrix.tsv`.

## Caveats
- **Single fold** (we have `fold_0` only; paper used 5) → no cross-fold error bars; per-arrangement
  ΔJ is a point estimate, variance estimated across the 100/64 backgrounds + bootstrap on the max-Z.
- **Consensus insertion** uses the CWM-argmax k-mer (one representative sequence), not sampled
  instances; affinity/flank effects not modelled. Backgrounds are the CT's GC-matched negatives.
- **Full screen** run at reduced fidelity (n=64, 2 bp uniform grid) for tractability; hero set is full
  fidelity (n=100, 1 bp). Uniform grid preserves the Z denominator.
- **Primary-CT** for the full screen = max hit co-occurrence CT; synergy is a per-CT statement.
- Not yet done (natural next steps): DeepLIFT confirmation that inserted motifs drive the predicted
  synergy (paper Fig 4e); homotypic/higher-order (>2) syntax; RNA co-expression cross-check of top
  synergistic TF pairs; promote the pipeline out of sandbox into `bin/3_single_task_models/`.

## Reproduce / where things live
Cluster: `sandbox/2026_07_19_syntax_pairwise_marg/` — `scripts/marginalize_pairs.py` (core),
`run_array.sh` / `run_full.sh` (SLURM carter-gpu), `inputs/pairs_{candidate,null,full_sorted}.tsv`,
`results/{candidate,null,full}/`, `figures/`. Env `eugene_tools` (tangermeme 1.0.3, bpnetlite).
