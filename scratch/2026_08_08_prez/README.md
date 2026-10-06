# Presentation figures — marginalization illustrations + PROX1 variant effect

Three deliverables, 2026-08-10.

1. **Single-TF marginalization** — background prediction track over motif-inserted prediction track.
2. **Two-TF synergy** — four tracks: TF A alone, TF B alone, additive expectation, observed together at maximum synergy.
3. **PROX1 variant effect** — the attached analysis repeated on the current 22-cell-type model set, on ENP.

All numbers below are in the TSVs under `tables/`. Vector PDFs and matching PNGs are in `figures/`.

---

## What had to be computed, and why

The published synergy pipeline never saved prediction profiles: `marginalize_pairs.py` keeps only the counts head (`y[1]`) and discards `y[0]` on every forward pass. Everything on disk — the 2,701 `.npz` grids, `calls_long.tsv` — is scalar, so no track could be reconstructed from existing results. The marginalizations were re-run with the profile head kept (`scripts/marginalize_tracks.py`), reproducing the published scalars exactly (see Verification).

- Model: single-task ChromBPNet, `chrombpnet_nobias.h5`, fold_0, 22 cell types.
- Backgrounds: **n = 32** per cell type from each model's own `filtered.nonpeaks.bed` — the same count and source as the published first pass, which is why the scalars reproduce.
- Motif inserts: argmax consensus of the signed CWM from catalog **v1.1** `cwms.npz`, importance-trimmed at `imp_frac = 0.1`. Reconstructed widths match the published `wa`/`wb` for all six pair motifs and the catalog `width` for all six illustration motifs.
- Compute: `carter-gpu`, one a30, ~6 min for all 22 cell types × 22 conditions × 2 background sets.

**Track definition.** Profiles are averaged in log space, renormalized, then scaled by the mean log-counts:

```
S(x) = mean_i log_softmax(profile_logits_i)(x)      # geometric-mean shape over the 32 backgrounds
q(x) = softmax(S)(x)                                 # renormalized, sums to 1
T(x) = q(x) * exp(mean_i log_counts_i)               # the plotted accessibility track
```

so that `area(T_motif) / area(T_background) = exp(Δ)` **exactly**. Averaging linearly instead would let the two or three highest-count backgrounds dominate the mean track and break that identity.

**Additive expectation.** Shape and area are decoupled:

```
u(x)     = log q_A(x) + log q_B(x) - log q_bg(x)     # clipped to median(u) ± 10
q_add(x) = softmax(u)(x)
T_add(x) = q_add(x) * exp(c_A + c_B - c_bg)
```

giving `area(T_add)/area(T_bg) = exp(ΔA+ΔB)` and `area(T_obs)/area(T_add) = exp(Δ)`, matching the published synergy score. The naive `y_A · y_B / y_bg` has area `exp(ΔA+ΔB) · area_bg · Σ q_A q_B / q_bg`, and that correction factor exceeds 1 exactly when both inserts concentrate signal centrally — it would push the additive track up and visually shrink the synergy gap.

---

## 1. Single-TF marginalization — `fig1_*`

Two stacked tracks on a shared y-axis: background above, motif-inserted below. Motif footprint shaded; `Δ` is the change in predicted log counts.

| figure | motif | TF | catalog category | width | cell type | log counts bg | log counts + motif | Δ |
|---|---|---|---|---|---|---|---|---|
| `fig1_single_ST15_HNF1A_early_ENP` | ST15 | HNF1A | Base obligate homodimer | 13 | early_ENP | 3.699 | 5.885 | **+2.185** |
| `fig1_single_ST21_GRHL2_PP2` | ST21 | GRHL2 | Base with flank | 22 | PP2 | 4.280 | 6.391 | **+2.111** |
| `fig1_single_ST45_RFX6_early_SC_EC` | ST45 | RFX6 | *Heterocomposite* | 23 | early_SC_EC | 3.763 | 5.302 | **+1.539** |
| `fig1_single_ST04_SOX9_PP2` | ST04 | SOX9 | Base | 17 | PP2 | 4.280 | 5.650 | **+1.370** |
| `fig1_single_ST10_MAFB_early_SC_beta` | ST10 | MAFB | Base with flank | 12 | early_SC_beta | 4.106 | 4.807 | **+0.701** |
| `fig1_single_ST09_NEUROD1_early_SC_beta` | ST09 | NEUROD1 | Base | 14 | early_SC_beta | 4.106 | 4.505 | **+0.399** |

Cell type shown is that motif's strongest of the 22 (own backgrounds). HNF1A is the largest single-motif effect in the catalog. Full per-cell-type values: `tables/single_tf_marginalization.tsv`.

Plotted window is **±450 bp** around the insertion site (the model output is 1000 bp, so 500 is the maximum). At ±150 the marginalization signal is still ~1.4× background at the frame edge, which makes the peak read as a plateau; at ±450 it decays back to background inside the frame. Adjustable via `--half`.

## 2. Cell-type specificity — `fig2_*`

The same insert run against all 22 models using **one shared background set** (DE negatives) so the panels are comparable across cell types; per-cell-type own backgrounds would differ for reasons unrelated to the model. Left column: log enrichment over background, ordered by developmental stage and coloured by `config/cell_type_metadata.tsv`. Right column: Δ log counts.

Range of Δ across the 22 models (shared backgrounds), sorted by how restricted the motif is:

| motif | TF | min | median | max | max / median | cell types above half-max |
|---|---|---|---|---|---|---|
| **ST04** | **SOX9** | −0.034 | **0.025** | 1.354 | **18.1** | **4 of 22** |
| ST21 | GRHL2 | 0.069 | 0.157 | 2.068 | 10.0 | 5 of 22 |
| ST45 | RFX6 | 0.269 | 0.534 | 1.655 | 2.8 | 8 of 22 |
| ST10 | MAFB | 0.046 | 0.306 | 0.798 | 2.2 | 9 of 22 |
| ST09 | NEUROD1 | 0.023 | 0.154 | 0.403 | 2.0 | 10 of 22 |
| ST15 | HNF1A | 0.002 | 1.578 | 2.152 | 1.3 | 20 of 22 |

**ST04 : SOX9 is the cell-type-specific example.** It is the most restricted motif in the catalog among those with a real effect (screened over all 73 motifs × 22 cell types using the homodimer rows of `calls_long`, where `dA = dS/2`): a large effect in the pancreatic-progenitor branch — PP2 +1.35, PP1 +1.22, exocrine +1.10, ENP_phase1 +0.98 — and flat zero everywhere else, including all 12 endocrine cell types (median +0.025 across the 22, minimum −0.034). It is also the cleanest signal of the ten motifs measured, with a central bump 67× the flank noise. Catalog category `Base`, and SOX9's own tomtom best match, so no composite caveat.

ST21 : GRHL2 is the runner-up — a bigger absolute effect (+2.11 in PP2) but a broader footprint (PP1/PP2/PFG1/PFG2/exocrine), so it reads as progenitor-wide rather than sharply restricted.

At the other extreme, HNF1A is near-ubiquitous (20 of 22 cell types above half-max) — the two panels side by side make the specificity contrast on their own. `FB_FLT1` (fibroblast) gives Δ = 0.002 for HNF1A, a biological zero and a useful negative control.

## 3. Two-TF synergy, four tracks — `fig3_*`

Four stacked tracks on a shared y-axis (A alone / B alone / additive expectation / observed together), background drawn in grey behind each, motif footprints shaded, plus a ΔA / ΔB / ΔS / ΔJ bar inset with the synergy gap annotated. Singles are placed at their **in-pair position and orientation**, not centred, so the four tracks are directly comparable. Window is ±300 bp here rather than ±450, so the motif footprints stay visible (`--pair-half`).

| figure | pair | TFs | cell type | arrangement | ΔA | ΔB | ΔS additive | ΔJ observed | synergy Δ | maxZ | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fig3_pair_ST11_ST52_early_SC_alpha` | ST11 × ST52 | SIX4 × PDX1 | early_SC_alpha | RR, 0 bp gap | +0.091 | +0.146 | +0.236 | **+1.024** | **+0.788** | 11.64 | 1.6e-2 |
| `fig3_pair_ST20_ST46_early_SC_EC` | ST20 × ST46 | RFX3 × NKX2-2 | early_SC_EC | FR, 1 bp gap | +1.280 | +0.057 | +1.337 | **+2.645** | **+1.308** | 7.61 | 5.4e-5 |
| `fig3_pair_ST29_ST58_late_SC_alpha` | ST29 × ST58 | RELB × MZF1 | late_SC_alpha | FF, 0 bp gap | +0.010 | +0.383 | +0.393 | **+1.824** | **+1.431** | 9.52 | 5.6e-10 |

Cell type shown is that pair's maximum-Δ cell type; arrangement is that (pair, cell type)'s published optimum. Published Δ from `calls_long.tsv` for the same three rows: +0.939, +1.289, +1.527 (these use centred, forward-orientation singles — see Caveats).

**`fig3b_*`** is the same comparison in log-enrichment space, `log q_c + c_c − log q_bg − c_bg`, where additivity is literal addition and the shaded band is the synergy. No renormalization argument is needed to read it.

## 4. Synergy across cell types — `fig4_*`

Additive expectation (ΔS, open marker) vs observed (ΔJ, filled) per cell type; the connecting line is the synergy. Filled circle = hard call, open = soft, blank = not significant (BH q < 0.05, published calls).

| pair | TFs | synergistic in | pattern |
|---|---|---|---|
| ST11 × ST52 | SIX4 × PDX1 | 2 of 22 (2 hard) | narrow — PP1 and early_SC_alpha only |
| ST20 × ST46 | RFX3 × NKX2-2 | 5 of 22 (4 hard) | endocrine-restricted — early/late_ENP, early_SC_beta, early_SC_alpha, early_SC_EC |
| ST29 × ST58 | RELB × MZF1 | 18 of 22 (18 hard) | broad |

## 5. PROX1 variant effect — `prox1_*`

**Variant:** `chr1:213977102:T:A` = rs79687284, PROX1 credible set, T2D DIAMANTE multi-ancestry, PIP 0.333, beta 0.177, p 2.4e-22.

Repeat of the attached analysis on the current **22-cell-type** model set, rendered for the ENP-stage cell types. Two files per cell type: `..._predictions_attributions.pdf` (reference vs alternate accessibility + both attribution logos) and `..._delta.pdf` (alt − ref).

| cell type | log counts ref | log counts alt | log fold change | rank of 22 |
|---|---|---|---|---|
| early_ENP | 6.608 | 5.748 | −0.859 | 15 |
| ENP_phase1 | 4.716 | 3.968 | −0.748 | 12 |
| late_ENP | 3.597 | 2.944 | −0.653 | 9 |
| proliferating_endocrine | 3.394 | 2.446 | −0.948 | 1 |

`proliferating_endocrine` is included because it carries `dev_stage = ENP` and `lineage = endocrine_progenitor` in the cell-type metadata *and* is the top-ranked cell type for this variant — but it is the proliferating compartment rather than a named ENP stage, so drop it if that is not what "ENP" meant. `early_ENP` has the highest absolute accessibility and shows the clearest reference/alternate separation.

Three changes from the previous render, all for legibility on a slide:

- **Shared y-limits across the reference and alternate attribution panels.** Autoscaling them independently made a genuine ~10× collapse of attribution at the HNF1 site read as merely "different letters".
- **`ST##:TF` catalog labels** on seqlet boxes, replacing the bare tomtom-style names.
- **Dashed rule at the variant position**, and more annotation rows so labels do not overprint.

The reference attribution is dominated by **ST15:HNF1A** directly under the variant; on the alternate allele that seqlet collapses. Note ST15's best tomtom match is HNF1B — the same motif family the old `D9_ENP` figure annotated as `FOSB`/`Foxd3` under the previous 71-cluster naming.

Also included: `tables/prox1_celltype_ranking_22models.tsv` (the `-log10(abs_logfc_x_jsd.pval)` ranking across all 22 models) and `figures/prox1_celltype_contrast.pdf` (the cell-type contrast panel), both from the 2026_08_02 run.

The two files originally attached are preserved unchanged in `reference/`.

---

## Verification

Run: `python3 scripts/validate_tracks.py --tracks results/tracks --pairs inputs/pair_arrangements.tsv`

| check | result |
|---|---|
| recomputed ΔS vs published `calls_long` (66 pair × cell-type rows) | max abs err **4.6e-4** |
| recomputed ΔJ vs published `dJ_opt` | max abs err **5.9e-4** |
| recomputed Δ vs published `delta` | max abs err **4.8e-4** |
| `area(T_add)/area(T_bg) == exp(ΔA+ΔB)` | max abs err **6.9e-16** |
| `area(T_obs)/area(T_add) == exp(Δ)` | max abs err **7.8e-16** |
| profile head: central bump vs flank noise | ratio **3.9 – 34×** across the 10 motifs |

Residual scalar error is at the rounding precision of the published TSV. Also asserted at run time: `model.trimming == 557`, profile output length 1000, reconstructed motif widths equal published `wa`/`wb`, reconstructed centre-to-centre distance equals published `opt_center_center` for all 66 rows, and the background tensor is unmodified after `tangermeme.substitute`.

## Caveats

- **ST45 (RFX6) is a Heterocomposite and ST20 (RFX3) is a Partial motif** in the v1.1 catalog — those inserts are composite elements, not clean single TF sites. The category is printed in each panel title.
- **Published Δ uses centred, forward-orientation single inserts**; the four-track figures use position- and orientation-matched singles, because the optima are RR, FR and FF and ChromBPNet is not reverse-complement equivariant. Both numbers are in `tables/pair_synergy_tracks.tsv` (`dA_centered` / `dA_in_pair`). The gap is at most 0.27 log counts, and the matched Δ agrees with the published Δ to within 0.03 for all three headline pairs.
- **Backgrounds are per-cell-type inaccessible negatives**, so the own-background track differs across cell types for reasons unrelated to the model. `fig2` uses the shared background set for that reason; `fig1`/`fig3`/`fig4` use own backgrounds so their numbers match the published table.
- **ST11 × ST52 in FB_FLT1** is noise (ΔJ 0.045, ΔS −0.010). It is kept and labelled in `fig4` rather than dropped.
- Single fold (fold_0) only; no cross-fold error bars. First-pass parameters: n = 32 backgrounds, gaps 0–50 bp.
- Model predictions, not experimental measurements.

## Reproducing

Cluster sandbox: `/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_08_08_prez`

```bash
# 0. preflight, login node, CPU
python scripts/preflight.py --models_root $MROOT --cwms $MROOT/motifs/v1.1/cwm/cwms.npz \
  -g /carter/users/aklie/data/ref/genomes/hg38/hg38.fa \
  --singles inputs/single_motifs.tsv --pairs inputs/pair_arrangements.tsv

# 1. tracks, carter-gpu, ~6 min  (do NOT pass BGSETS via --export: sbatch splits it on the comma)
sbatch scripts/marginalize_tracks.sh

# 2. PROX1 re-render, carter-gpu, ~4 min
sbatch scripts/plot_prox1.sh

# 3. locally, after rsyncing results/
python3 scripts/validate_tracks.py --tracks results/tracks --pairs inputs/pair_arrangements.tsv \
  --out tables/validation_scalars.tsv
python3 scripts/make_tables.py  --tracks results/tracks --pairs inputs/pair_arrangements.tsv \
  --singles inputs/single_motifs.tsv --ct_metadata config/cell_type_metadata.tsv -o tables
python3 scripts/plot_tracks.py  --tracks results/tracks --pairs inputs/pair_arrangements.tsv \
  --singles inputs/single_motifs.tsv --ct_metadata config/cell_type_metadata.tsv -o figures \
  --half 450 --pair-half 300      # bp either side of the insertion site; 500 is the maximum
```

## Layout

```
figures/    fig1_* fig2_* fig3_* fig3b_* fig4_*  (PDF + PNG), prox1_*.pdf
tables/     single_tf_marginalization.tsv, pair_synergy_tracks.tsv, validation_scalars.tsv,
            prox1_celltype_ranking_22models.tsv, prox1_logcounts.tsv
results/    tracks/tracks__<cell_type>.npz  -- raw per-background log-softmax profiles (32, 1000)
            and log counts (32,) for every condition, both background sets; every averaging and
            renormalization choice above is re-decidable from these without another GPU job
            prox1/  the rendered variant PDFs
inputs/     pair_arrangements.tsv (66 pair x cell-type rows), single_motifs.tsv
config/     cell_type_metadata.tsv, catalog_v1.1_metadata.tsv
scripts/    preflight.py, marginalize_tracks.py(.sh), tracks_io.py, validate_tracks.py,
            make_tables.py, plot_tracks.py, plot_prox1.py(.sh)
reference/  the two files originally attached, unchanged
```
