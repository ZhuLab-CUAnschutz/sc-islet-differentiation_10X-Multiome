# Determining TF-motif-pair synergism — methodology & rigor

_Method reproduces Betty B. Liu et al., "Multiomics and deep learning dissect regulatory syntax in
human development," Nature 2026 (doi 10.1038/s41586-026-10326-9), Methods "In silico marginalizations
to assess motif synergy," applied to our 22 single-task islet ChromBPNet models + v1.1 motif catalog._

## 1. In-silico pairwise marginalization
For a motif pair (A, B) in a given cell type's ChromBPNet model:

- **Backgrounds.** N GC-matched **inaccessible** genomic sequences (the model's own
  `filtered.nonpeaks.bed` negatives), 2114 bp — the model's input window. (Not peak sequences; not
  dinucleotide-shuffled peaks. This is the paper's background, and controls for local GC/base content.)
- **Inserted sequence.** Each motif's consensus = **argmax of its collapsed, importance-trimmed CWM**
  (contribution-weighted, not PPM-frequency argmax), matching the paper.
- **Predicted effect.** Insert a sequence into the center of every background; the marginal effect is
  the mean difference in predicted **natural-log counts** (accessibility), edited − background:
  ΔA = y_A − y₀, ΔB = y_B − y₀ (each motif alone), **ΔJ = y_J − y₀** (both inserted).
- **Additive (independence) model.** ΔS = ΔA + ΔB — a log-additive model = multiplicative in counts.
  **Synergy = ΔJ exceeding ΔS**: the two motifs together do more than the sum of their parts.

## 2. Arrangement sweep → optimal arrangement
Both motifs are inserted at every **orientation** (FF/FR/RF/RR for distinct motifs; fewer for
same/palindromic) × every **inner-edge spacing 0–200 bp** (uniform grid). Each (orientation, spacing)
is an "arrangement." The **optimal arrangement** = the (orientation, spacing) with the greatest mean ΔJ.
Δ = ΔJ(optimal) − ΔS is the reported synergy magnitude.

## 3. Calling synergy (statistics)
At the optimal arrangement, across the N background sequences, we test whether the **paired** joint
effect exceeds the independent effect with a **Wilcoxon signed-rank test** (ΔJ vs ΔA+ΔB per sequence),
Benjamini–Hochberg corrected. A pair is **synergistic** if adjusted p < 0.001 **and** Δ > threshold.

**Threshold calibration (a rigor requirement of our extension).** The paper classified only 138
pre-filtered de-novo composite motifs and used Δ > 0.15, Z > 4. We instead run an **unbiased all-pairs
screen** (2628 pairs). In that regime those cutoffs are far too permissive: on a **lineage-mismatched
null** (126 pairs of TFs from unrelated lineages), **54% would be called synergistic at Δ>0.15 and 14%
"hard" at Z>4.** We therefore set thresholds at the **95th percentile of the null** (~5% FDR):
**Δ > 0.504, Z > 5.38.** (Even mismatched pairs average Δ≈0.17 — inserting any two motifs near an
accessible center is mildly non-additive; the null absorbs this baseline.) The report exposes Δ/Z as
**live inputs** so the cutoff is transparent and adjustable, with null percentiles shown for reference.

## 4. Syntax class (spacing/orientation preference)
- **Hard syntax:** any arrangement's ΔJ is a Z-score > (cal.) above the mean over all arrangements —
  a sharp spacing/orientation preference (DNA-mediated cooperativity, typically <20 bp).
- **Soft syntax:** Δ > 0.15 at some arrangement with 20–150 bp spacing — flexible, longer-range
  (nucleosome-mediated).

## 5. Cell-type specificity
Two levels, both provided:
- **Fixed-arrangement (cheap, all 2628 pairs):** the pair's primary-CT optimal arrangement re-scored in
  every cell-type model → Δ per CT. Answers "where is this fixed grammar active?"
- **Re-optimized (582 synergistic pairs):** the full spacing×orientation sweep repeated **independently
  in each of the 22 cell types**, so the arrangement is re-optimized per CT → the true per-CT maximum.
  This is the matrix behind the synergy clustermaps.

## 6. Rigor / quality controls
- **Null calibration** (§3) — the central defense; without it the all-pairs regime is uninterpretable.
- **DeepLIFT confirmation** (the paper's key QC) — for each synergistic pair at its optimal arrangement
  we compute count-head attributions (tangermeme `deep_lift_shap`) on the edited sequence and measure
  the fraction of central (±100 bp) |attribution| falling on the **two inserted motif footprints**
  (`frac_in`). If most attribution is elsewhere (junction/spacer bases), the "synergy" is an accidental
  composite site, not A–B cooperativity. **Result: median frac_in = 0.58** — highly concentrated (the
  two motifs occupy ~30 of 200 central bp yet hold ~58% of attribution); **only 7 / 582 pairs (1%)
  fall below 0.40** and are flagged as possible artifacts (e.g. OTX2×ONECUT2, EOMES×OTX2, ONECUT2×FEV).
  Table: `figures/deeplift_confirm.tsv` (`deeplift_flag`).
- **Additivity control (built in):** for every pair, ΔJ → ΔS as spacing → 200 bp — motifs become
  independent at distance, the method's internal negative control (visible in every distance curve).
- **Composite positive control:** motifs the collaborator independently annotated as Hetero/Homocomposite
  (FOXA2×OTX2, RFX6×NKX2-2, OTX2×GATA, MECOM×GATA) are recovered as the strongest hard-syntax synergies.
- **Batching correctness:** the batched sweep reproduces the un-batched ΔJ exactly (ISL1×NKX6-1 β=0.738).

## 7. Limitations (stated, not hidden)
- **Single fold.** We have `fold_0` only; the paper averages 5 folds and shows cross-fold error bars. We
  cannot draw those; per-arrangement ΔJ is a point estimate. Mitigations: the Wilcoxon is across the 100
  sequences (fold-independent, still valid), and the hard-syntax max-Z is bootstrap-checked over
  backgrounds. Net effect: hard calls are the most single-fold-sensitive — treat Z near threshold as soft.
- **Consensus k-mer insertion** (one representative sequence, argmax CWM) rather than multiple sampled
  instances — ignores affinity/flank variation.
- **Primary CT** for the all-pairs screen is chosen by hit **co-occurrence**, not by max synergy; the
  optimal arrangement is therefore locked from that CT (the §5 re-optimized sweep removes this for the
  synergistic set).
- **Reproducibility knobs:** thresholds are data-driven from the null, not universal constants.
