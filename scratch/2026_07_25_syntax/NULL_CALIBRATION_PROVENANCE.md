# Null calibration — provenance & rationale

_What the empirical null is, how it was built, and where every number comes from._
_Written 2026-07-27 by tracing the source files on NRNB (see paths at bottom)._

## TL;DR

- The null is **42 hand-authored "lineage-mismatched" motif pairs** — a 7 × 6 grid of TFs that
  should not co-bind.
- Those same 42 pairs were calibrated **two different ways**, and the report text mixes them up:
  - **42 × 3 = 126** — original *global* calibration (3 cell states) → one threshold pair **Δ>0.504, Z>5.38**.
  - **42 × 22 = 924** — *per-cell-type* null (all 22 models) → per-state thresholds `d95`/`z95`.
- The specificity / tau work uses the **per-CT (42×22)** null. The 2026-07-19 report used the **global (42×3)** one.

---

## 1. The 42 pairs (the grid)

Source: `inputs/pairs_null.tsv` (hand-authored 2026-07-19, no builder script). 42 rows,
each tagged `group=null, note=lineage-mismatched`. It is a full cross of two hand-picked TF sets:

| side | role | TFs (motif ID) |
|---|---|---|
| **idA — lineage-restricted (7)** | islet/endocrine fate determinants | ISL1 (ST36_sub2), NKX6-1 (ST36_sub4), LMX1A (ST36_sub3), PAX6 (ST33), NEUROD1 (ST09), PDX1 (ST52), MIXL1 (ST36_sub1) |
| **idB — different domains (6)** | TFs from non-overlapping expression domains | SALL4 (ST12), TEAD4 (ST07), ZNF143 (ST01), NRF1 (ST61), HNF4G (ST05_sub1), FEV (ST08) |

7 × 6 = **42 pairs**. Examples: PDX1×HNF4G, ISL1×SALL4, NKX6-1×ZNF143, NEUROD1×NRF1.

### Rationale (what was principled vs arbitrary)

**Principled:**
- *Concept.* A null needs pairs with no biological opportunity to cooperate → "lineage-mismatched"
  = TFs whose expression domains do not overlap (different cell types / stages, never in the same
  nucleus). Whatever Δ they produce is the baseline for "inserting any two motifs near an accessible
  center is mildly non-additive" (mismatched pairs average Δ≈0.17).
- *Row TFs* are the real endocrine-fate regulators from the Dingyu Liu et al. β-vs-EC logic
  (ISL1/NKX6-1/PDX1/PAX6/NEUROD1 → β; LMX1A → EC). MIXL1 (mesendoderm) is the one loose pick.
- *Column TFs* deliberately span several unrelated domains — pluripotency (SALL4), Hippo (TEAD4),
  promoter-proximal/housekeeping (ZNF143, NRF1), hepatic-endoderm (HNF4G), serotonergic/EC (FEV) —
  so the null is not one lineage contrast repeated.

**Arbitrary (hand choices, no documented criterion):**
- **7 vs 6** — no rule; "grab a handful of clearly restricted TFs per side." 6×6 or 7×7 would be
  equally defensible. 42 is just what the grid produced.
- Exact column identities (why FEV not another EC TF; why HNF4G not HNF1B) — representative, not derived.
- "Non-overlapping" judged from **known TF biology**, not from a co-expression matrix in this dataset.

---

## 2. Two calibrations of the same 42 pairs

### (a) Global calibration — 42 × 3 = 126   [`results/null`]

Ran the 42 pairs through **3 cell states**: `DE` (early progenitor), `early_SC_beta` (β fate),
`early_SC_EC` (EC fate). `.npz` files named `{idA}__{idB}__{ct}` → 7 × 6 × 3 = **126 observations**.
Produced **one** global threshold pair at the 95th percentile of the pooled null (~5% FDR):

    Δ > 0.504,  Z > 5.38            (figures/null_calibration.txt: "null pairs: 126")

Motivation stat: at the paper's own Δ>0.15 / Z>4 cutoffs, **54%** of these mismatched pairs would be
falsely called synergistic and 14% "hard" — i.e. the paper's thresholds do not transfer to our
single-fold / all-pairs regime. This is the calibration used in the **2026-07-19 REPORT**.

### (b) Per-CT null — 42 × 22 = 924   [`results/null_allct`]

Ran the **same 42 pairs** through **all 22 single-task models** (`inputs/pairs_null_allct_sorted.tsv`,
verified 42 per CT × 22 CT = 924). For each cell state, from its 42 null values we compute:

- `dmean`, `dsd` → the standardization Δ* = (Δ − dmean)/dsd
- `d95` = 95th pct of null Δ  (effect-size call bar, per state)
- `z95` = 95th pct of null maxZ  (significance call bar, per state)

A pair is **synergistic in cell state X** iff `Δ > d95_X AND maxZ > z95_X`. Thresholds table:
`specificity/per_ct_null_thresholds.tsv`. This is the null feeding the **specificity / tau** work
(`specificity_analysis.py`, `group_specificity.py`).

---

## 3. Caveats to state in any methods write-up

1. **Pairs vs observations.** "126 lineage-mismatched pairs" (REPORT.md, null_calibration.txt) is
   really 42 *pairs* × 3 *cell states* = 126 *observations*. There are only 42 distinct pairs.
2. **Small tail for the per-CT null.** n = 42 per cell state; a 95th percentile rides on the top ~2
   points, so `d95`/`z95` are noisy per-state estimates. (Bigger concern than 6-vs-7.)
   Cheapest mitigation: bootstrap a CI on the threshold — no re-marginalization needed.
3. **Hand-authored.** The 42-pair grid has no builder script and no documented per-TF criterion;
   it is a curator's negative-control set.

---

## 4. "The null is a subset of what we test" — is that circular?

The 42 null pairs **are** in the full test set: each appears 22× (once per CT) in
`inputs/pairs_allpairs_allct_sorted.tsv`. So the null overlaps the tested pairs. Two points:

- **Subset-hood alone is not circular.** A designated negative-control set that overlaps the test set
  is standard practice (decoys / spike-ins): use pairs believed *a priori* to be null to set a bar,
  then apply it to everything. The ~5% of null pairs that self-clear the bar is just the FDR, not leakage
  into the pairs of interest.
- **BUT the negatives here are not clean.** Empirically, **6 / 42 (14%)** of the "mismatched" null
  pairs are called synergistic at their own primary CT: NEUROD1×HNF4G, NEUROD1×SALL4, NEUROD1×FEV,
  PAX6×TEAD4, MIXL1×TEAD4, NKX6-1×TEAD4. TEAD4 (promiscuous composite hub) is in 3; NEUROD1
  (endocrine, e.g. NEUROD1×FEV plausibly real) in 3. So the hand-curated "mismatch" label admits
  several pairs that can genuinely cooperate → the null is contaminated at ~14%, well above the 5% target.

### Two kinds of "mismatch" — don't conflate

- **TF×TF mismatch** — the *pair* shouldn't cooperate (what the 42-grid is).
- **pair × wrong-cell-type** — a real pair tested where it's inactive. Tempting as a null, but
  **"other cell type" ≠ "inactive"**: a β-pair is synergistic across several endocrine states, so
  naively using every non-primary CT as null folds real signal into the null. Any per-CT null must
  **exclude the CTs where the pair is active**, or it self-contaminates (the 14% leakage is this in miniature).

### Quantified: how much the spiked-in signal cost (2026-07-27)

Recomputed thresholds with vs without the 6 contaminating pairs (null Δ/Z pulled from
`results/null_allct`):

- **Global bar** (behind the 582 calls): 0.492 → 0.473, **+3.9%** inflation. Small.
- **Per-CT bars** (behind the 233 specificity set): **+8.4% mean**, very uneven —
  **DE +48%**, PGT2 +21%, early_ENP/PP2/liver +15%, <5% in ~half the states.
- **Direction is conservative:** signal in the null → higher bar → we UNDER-called synergy.
  No false positives were manufactured; the 582/233 are undercounts. Calls that were made are safe
  (cleared an inflated bar). Cost = missed synergies, concentrated in DE / early progenitors (where the
  TEAD4-containing "mismatched" pairs are actually active).
- **Headline branch-level conclusion unaffected** (doesn't depend on the exact bar); per-CT counts for
  DE / early progenitors are the soft spot.
- **Fix (no re-marginalization needed):** drop the 6 / trimmed estimator, recompute d95/z95, re-call —
  or move to the full-matrix empirical null (§5).

## 5. The eventual fix — full-matrix empirical null

Current state: all 2628 pairs scored **only at their primary CT** (most co-occurring by Fi-NeMo hits)
→ 582 synergy calls. The `full_allct` run scores **all 2628 × 22**. Once that lands, the hand-picked
42-null is retired: the null becomes the **bulk of the full matrix itself** — most pair×CT combinations
are non-synergistic background, the synergistic tail is the signal (Efron empirical / local FDR). In that
framing null ⊆ test is *by design* and correct, and it removes both current weaknesses (tiny n; trusting a
curator's mismatch call). **Carry-over caution:** fit the empirical null on the central bulk
(robust / trimmed), so the synergistic tail doesn't pull the bar up — and still don't let a pair be its
own null in the CTs where it is active.

## Source files (NRNB)

Project: `carter:/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg/`

- `inputs/pairs_null.tsv` — the 42-pair grid (copied local as `inputs_pairs_null.tsv`)
- `inputs/pairs_null_allct_sorted.tsv` — 42 × 22 driver for the per-CT null
- `results/null/` — global 3-CT calibration (126 obs)
- `results/null_allct/` — per-CT null (924 obs)
- `figures/null_calibration.txt` — "null pairs: 126  delta 95pct=0.504  maxZ 95pct=5.38"
- `analysis/METHODS_synergy.md`, `REPORT.md` — prose (mixes the two regimes; see caveat 1)
