# Defining cell-type-specific synergy — three methods + coherence

**Call layer (per-cell-type null).** The 42 lineage-mismatched null pairs were run through all 22
single-task ChromBPNet models (`null_allct`) and, for free via one pass, the 22 multitask heads
(`mt_null`). Each cell type gets its OWN Δ/Z 95th-percentile bar. This matters: the single-task null
Z95 spans ~4.2–6.1 and Δ95 varies per model; MT null Δ95 spans 87–285. A pair is "synergistic in cell
type X" iff, in X's model, Δ > X's Δ95 AND maxZ > X's Z95.

Recalibrating single-task from the old global bar (Δ>0.504, Z>5.38) to per-cell-type bars drops the
"synergistic in ≥1 CT" set from 582 to **233** (within the 582 all-CT universe; pairs outside it that
might newly qualify under a lowered per-CT bar are not in the single-task all-CT data — the multitask
run, which covers all 2628 pairs × 22 CT, is the check for those and calls **201**).

**Specificity layer — three definitions of "specific to X":**
- **M1 standardized-Δ clustering.** Δ* = (Δ − null_mean)/null_sd per cell type (magnitude-comparable
  across models); row-z across cell types; Ward clustering. Natural k=2 (single-task silhouette 0.20;
  multitask 0.61).
- **M2 breadth / calls.** # cell types a pair is synergistic in → specific (≤3) / restricted (4–11) /
  broad (≥12).
- **M3 differential contrast.** Tau specificity index on Δ* (0 broad … 1 one-cell-type), one-vs-rest
  score per cell type, and a per-lineage Mann–Whitney contrast.

**Coherence (what to trust):**
- The two *relative* methods (M1 clustering, M3 Tau/one-vs-rest) agree strongly — same top cell type
  for 100% of pairs, median Jaccard 0.60. They robustly say WHICH cell type each pair peaks in.
- The *categorical calls* (M2) are a partly separate axis: the relative-peak cell type is also a
  *called*-synergistic cell type for only ~52% of pairs (a pair can peak relatively where it doesn't
  clear the absolute bar, and vice versa).
- **Cross-model is the real test.** Single-task and multitask synergistic sets overlap modestly (72
  pairs). Their *fine* lineage assignment agrees only 18%, but the **coarse axis — endocrine vs
  foregut/non-endocrine — agrees 88%.**

**Bottom line for defining CT-specific synergy:** the robust, model- and method-independent signal is
the **major k=2 axis** (a pair's synergy is specific to the endocrine branch or the foregut/progenitor
branch). Finer per-lineage specificity is noisier and should be treated as a ranking, not a hard call.
Consensus cell-type-specific set = tag "specific" + Tau ≥ 0.6 (151 single-task pairs); recurrent TFs
FOXA2/OTX2/TEAD4/FOXC1 (foregut) and RFX6/MAFB (endocrine).
