# PROX1 variant effect + marginalization illustrations

Five files, one per question. Single-task ChromBPNet, fold_0, 22 developmental cell types.
Motif IDs are v1.1 catalog `ST##`. All predictions, not measurements.

---

### 1. `1_single_TF_marginalization_HNF1A.pdf` — single-TF marginalization

**ST15 : HNF1A** inserted at the centre of 32 inaccessible background sequences, cell type **early_ENP**.
Top track = average background prediction. Bottom track = average prediction with the motif inserted.
Shared y-axis; shaded band marks the inserted 13-bp consensus.

Predicted log counts 3.70 → 5.88, **Δ = +2.19**. The largest single-motif effect in the catalog.
The dip at the motif with shoulders either side is the footprint.

### 2. `2_cell_type_specificity_SOX9.pdf` — the same, across all 22 cell types

**ST04 : SOX9**, inserted into one shared background set so the 22 panels are directly comparable.
Left: enrichment over background per cell type. Right: Δ log counts.

PP2 +1.35, PP1 +1.22, exocrine +1.10, ENP_phase1 +0.98, and flat zero in the other 18 including
all 12 endocrine types (median +0.03, minimum −0.03). The most cell-type-restricted motif in the
catalog with a real effect.

### 3. `3_two_TF_synergy_SIX4_PDX1.pdf` — two-TF marginalization, four tracks

**ST11 : SIX4 × ST52 : PDX1**, cell type **early_SC_alpha**, at the optimal arrangement
(RR orientation, 0 bp gap, 11 bp centre-to-centre). Four tracks, shared y-axis, background in grey behind each:

| track | Δ log counts |
|---|---|
| SIX4 alone | +0.09 |
| PDX1 alone | +0.15 |
| additive expectation (ΔS) | +0.24 |
| **both together, observed (ΔJ)** | **+1.02** |

**Synergy = ΔJ − ΔS = +0.79.** Positive in 32 of 32 backgrounds; BH q = 0.016.

The additive track is the log-additive expectation, so `area(observed)/area(additive) = exp(synergy)`
exactly — the gap between the third and fourth tracks *is* the synergy score.

### 4. `4_PROX1_variant_effect_early_ENP.pdf` + `4_PROX1_celltype_ranking.tsv`

**chr1:213977102:T:A** = rs79687284, PROX1 credible set, T2D DIAMANTE multi-ancestry, PIP 0.333,
beta 0.177, p 2.4e-22. Cell type **early_ENP**.

Top: delta predicted accessibility (alternate − reference). Middle: reference attributions.
Bottom: alternate attributions, **on the same y-scale**. Dashed line = variant position.
Seqlet boxes are labelled with v1.1 catalog IDs.

Predicted log counts 6.61 (ref) → 5.75 (alt), **log fold change −0.86**. The reference attribution is
dominated by **ST15:HNF1A** directly under the variant; that seqlet collapses on the alternate allele.
Same motif as figure 1.

The TSV is the per-cell-type ranking across all 22 models, `-log10(abs_logfc_x_jsd.pval)`, most
affected first.

---

### Notes

- This is the previous PROX1 analysis re-run on the **current 22-cell-type model set**. The earlier
  version used the older 71-cluster models, so cell-type names differ (`D9_ENP` has no direct
  equivalent; the ENP-stage types here are `early_ENP`, `ENP_phase1`, `late_ENP`).
- `early_ENP` was chosen for the panel because it has the highest absolute accessibility at this
  locus and so the clearest reference/alternate separation. `late_ENP` is the highest-ranked ENP
  stage in the TSV; panels for it and for `ENP_phase1` are available on request.
- Marginalization backgrounds are GC-matched inaccessible regions (`filtered.nonpeaks.bed`), n = 32,
  matching the parameters of the existing synergy analysis.
- Single fold (fold_0), so no cross-fold error bars.
- Alternate TFs (RFX6, MAFB, NEUROD1, GRHL2) and synergy pairs (RFX3 × NKX2-2, endocrine-restricted;
  RELB × MZF1, synergistic in 18 of 22 cell types) are available, as are per-cell-type versions of
  every panel.
