# DE synergy of EOMES, MIXL1, SOX17, FOXH1 — catalog vs JASPAR inserts

_2026-09-06 · DE single-task ChromBPNet · full fidelity (n=100 backgrounds, gaps 0–200 bp, 1-bp, all orientations) · model predictions, not measurements. Sandbox `2026_09_06_de_synergy/`; parameters and JASPAR links in `README.md`._

## Bottom line

**None of the six pairs is called synergistic in DE, with either insert set.** Every Δ sits at 0.05–0.21, below the DE lineage-mismatched null 95th percentile (Δ > 0.52). The largest is catalog EOMES×MIXL1 (Δ = +0.21, RR, 1-bp gap, Z = 4.9, joint > additive in 83 of 100 backgrounds): a real but small close-spacing bump, 40 % of the threshold. The JASPAR version of the same pair is flat (Δ = +0.07).

The deeper issue is upstream of synergy: **the DE model barely responds to any of these inserts alone** (ΔA ≤ 0.15 log counts; for comparison the 42 null pairs' additive sums ΔS average 0.7 in DE). The JASPAR FOXH1 consensus gives ΔA = 0.0004, so every FOXH1 pair is a test against a motif the model does not see.

## Pairs (DE, both sets)

| pair | catalog Δ | arrangement | Z | JASPAR Δ | arrangement | Z | call |
|---|---|---|---|---|---|---|---|
| EOMES×MIXL1 | **+0.21** | RR @ 1 bp | 4.9 | +0.07 | FF @ 105 bp | 2.6 | none / none |
| EOMES×SOX17 | +0.10 | RR @ 105 bp | 4.1 | +0.12 | FF @ 19 bp | 3.9 | none / none |
| MIXL1×SOX17 | +0.13 | RR @ 0 bp | 5.7 | +0.05 | FR @ 18 bp | 2.8 | none / none |
| EOMES×FOXH1 | +0.10 | RR @ 16 bp | 3.7 | **+0.14** | FR @ 1 bp | 5.4 | none / none |
| MIXL1×FOXH1 | +0.10 | RR @ 3 bp | 4.3 | +0.05 | FR @ 103 bp | 2.8 | none / none |
| SOX17×FOXH1 | +0.08 | RR @ 98 bp | 4.7 | +0.06 | FR @ 97 bp | 3.3 | none / none |

Thresholds from the 42-pair DE null at identical fidelity: Δ > 0.52 and Wilcoxon p < 10⁻³ → synergistic; additionally Z > 4.2 → hard. Full columns (ΔA, ΔB, ΔS, ΔJ, p, frac_pos, first-pass cross-reference) in `tables/pairs_by_set.tsv` and `tables/calls_long.tsv`.

![Fig 1. Synergy Δ per pair, catalog (purple) vs JASPAR (orange) inserts; dashed = DE-null 95th percentile.](figures/fig1_delta_by_pair.png)

![Fig 2. ΔJ versus inner-edge gap for each orientation. Dashed = additive ΔS; grey = region below the null threshold; circle = optimum. Every curve returns to ΔS by 200 bp (additivity control).](figures/fig2_distance_curves.png)

## Catalog vs JASPAR

- **Arrangement preferences that do exist are close-spacing** (0–3 bp) for catalog EOMES×MIXL1, MIXL1×SOX17, MIXL1×FOXH1 and JASPAR EOMES×FOXH1. Optima at ~100 bp are noise on a flat curve.
- **EOMES:** the JASPAR insert (`AAGGTGTGAA`) is 3× the catalog half-site (`TTCACACC`) alone (ΔA 0.10 vs 0.03), yet the catalog pair with MIXL1 is the stronger one. EOMES×EOMES homodimers at 0 to 1 bp reach Δ ≈ 0.2–0.3 in both sets.
- **MIXL1:** the 12-bp catalog motif (`GGTGCTAATCAG`) is the strongest single insert (ΔA 0.15); the IC-trimmed JASPAR core `TAATTA` (6 bp) is ~10× weaker and drives nothing. The two MIXL1 sets are not comparable for this reason.
- **SOX17:** both inserts are near-inert alone (catalog `AGAATGC` 0.02, JASPAR `CATTGTC` 0.01). The catalog G in place of the canonical SOX C (GAATG vs CAATG) does not change that. But **catalog SOX17×SOX17 is the sharpest arrangement in the whole run** (Δ = +0.30, FF @ 2 bp, Z = 16, joint > additive in 84 of 100 backgrounds): two half-sites 2 bp apart behave like one dimer site, matching the curator's "half of an obligate homodimer" note. Magnitude still under threshold.
- **FOXH1:** zero on its own; the JASPAR EOMES×FOXH1 peak (FR @ 1 bp) is the only FOXH1 signal and is small.

![Fig 3. Each insert alone in DE.](figures/fig3_inserts.png)

![Fig 4. Homotypic pairs.](figures/fig4_homodimers.png)

## Consistency with the first pass

The three catalog pairs were in the 2,701-pair first pass (n=32, gaps 0–50) in DE: Δ 0.31 / 0.17 / 0.21 for EOMES×MIXL1 / EOMES×SOX17 / MIXL1×SOX17, none called. At n=100 they are 0.21 / 0.10 / 0.13, same ordering, all within 0.1, same RR orientation and near-abutting optimum for EOMES×MIXL1 (1 bp both runs) and MIXL1×SOX17 (4 bp → 0 bp). Nothing changes with full fidelity.

## Multitask-model replication

**The DE head of the CRESTED multitask model also calls none of the six heterotypic pairs synergistic, with either motif set.** This was a separate full-fidelity run of all 19 target pairs and the same 42-pair null (100 backgrounds, gaps 0–200 bp by 1 bp, all orientations). The matched multitask DE-null thresholds are Δ > 161.37 raw predicted counts and maxZ > 6.38.

The strongest multitask signals are JASPAR EOMES×SOX17 (Δ = 39.61; 0.25× the null threshold; FR at 0 bp; Z = 6.06) and catalog EOMES×SOX17 (Δ = 32.94; 0.20× threshold; RF at 0 bp; Z = 8.82). Catalog EOMES×MIXL1 is only 0.15× threshold. Thus, unlike the single-task model—where EOMES×MIXL1 was the largest weak bump—the multitask model ranks EOMES×SOX17 first, but both models agree that its magnitude is far below the empirical null cutoff.

The most notable homodimer changes are also subthreshold: catalog MIXL1×MIXL1 reaches 0.79× the null threshold (FR at 0 bp), while the strong arrangement-specific catalog ST66×ST66 signal seen in the single-task model is absent (0.05× threshold). JASPAR SOX17×SOX17 reaches 0.24× threshold.

![Fig 5. Multitask-model DE synergy normalized to the matched DE-null 95th percentile; dashed line is the calling threshold.](figures/fig5_multitask_delta_by_pair.png)

Absolute Δ values are **not comparable across the two models**: the multitask model emits raw predicted counts and uses shared dinucleotide-shuffled peak backgrounds, whereas the single-task model reports log-count effects on DE inaccessible backgrounds. The defensible cross-model comparison is each pair's strength relative to its model-matched null, plus rank/orientation/spacing. Full multitask values are in `tables/mt_pairs_de.tsv` and `tables/mt_calls_long.tsv`.

## Caveats

- Single fold; single consensus k-mer per motif (no affinity or flank variation). The 6-bp JASPAR MIXL1 core is the clearest casualty of consensus insertion.
- Backgrounds are DE inaccessible negatives; a motif whose effect depends on an already-open context (e.g. FOXH1 as a SMAD co-factor) is invisible in this assay.
- The multitask replication uses its established shared dinucleotide-shuffled peak backgrounds rather than DE inaccessible negatives. This preserves comparability to the multitask null but means single-task versus multitask differences combine architecture and background context.
- Null n = 42 pairs; the 0.52 cutoff is a 5 % FDR estimate, not a sharp line. No pair is near it.
- ST66 is a SOX half-site with a non-canonical G; ST18 is an 8-bp T-box half-site. Neither is a strong DE motif in this model despite being derived from DE.
