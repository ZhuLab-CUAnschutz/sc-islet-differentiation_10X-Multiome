# DE Higher-Order TF Synergy + Grant Figures

Extension of `2026_09_06_de_synergy` (pairwise screen) and `2026_08_08_prez` (presentation figures)
for the 4 definitive-endoderm TFs: **EOMES, MIXL1, SOX17, FOXH1**.

## Deliverables

1. **Stacked-track synergy figures** (fig3 style: +A, +B, Σ, observed) for all 10 pairs in DE
2. **3-TF combinatorial screen**: pairwise-anchored sweep of a third motif, with matched null
3. **Interpretive panels**: fig1 (single-TF tracks), fig2 (specificity across 22 cell types),
   fig3b (log-space overlay), fig4 (multitask synergy across 22 cell types)

## Motifs (set1: catalog + J_FOXH1)

| TF     | Short ID    | Source        | Width |
|--------|-------------|---------------|-------|
| EOMES  | ST18        | catalog v1.1  | 8     |
| MIXL1  | ST36_sub1   | catalog v1.1  | 12    |
| SOX17  | ST66        | catalog v1.1  | 7     |
| FOXH1  | J_FOXH1     | JASPAR MA0479 | 9     |

## Layout

```
scripts/
  build_inputs.py        local   Generate input TSVs from de_synergy results
  preflight.py           cluster Path/geometry checks before GPU spend
  marginalize_tracks.py  cluster Copied from prez — profile tracks for fig1/2/3/3b
  tracks_io.py           local   Copied from prez — track loading + math
  marginalize_triples.py cluster 3-way sweep engine (extends marginalize_pairs.py)
  run_tracks.sh          cluster SLURM wrapper for track marginalization
  run_triples.sh         cluster SLURM array: target triples + null
  analyze_triples.py     local   Null calibration + calling for triples
  plot_tracks.py         local   fig1, fig2, fig3 (pairs/triples/quad), fig3b
  plot_mt_fig4.py        local   fig4 dumbbell from multitask data
  plot_distance_curves.py local  Distance-curve figures from de_synergy grids
inputs/
  motifs_de4.npz         7 insert arrays (catalog + JASPAR)
  cwms_v1.1.npz          Full catalog v1.1 CWMs (for null motif selection)
  single_motifs.tsv      4 rows — single-TF figure manifest
  pair_arrangements.tsv  10 rows — pair figure manifest (6 hetero + 4 homo)
  triples.tsv            12 rows — target triple manifest (4 triples × 3 anchors)
  null_triples.tsv       42 rows — null triple manifest (6 pairs × 7 nulls)
config/
  cell_type_metadata.tsv 22 cell types with colours and display order
  catalog_v1.1_metadata.tsv  Full motif catalog metadata
results/
  tracks/                tracks__<ct>.npz × 22  (from run_tracks.sh)
  triples/               target triple sweep outputs
  null_triples/          null triple sweep outputs
tables/
figures/
  common_scale/          Shared y-limit versions
deliverable/             Curated subset for Han
logs/
```

## Reproduce

```bash
# ── local (Mac) ──
python scripts/build_inputs.py

# ── sync to cluster ──
rsync -az --exclude .DS_Store --exclude __pycache__ . narrows:$REPO/scratch/2026_09_20_de_higher_order/

# ── cluster: preflight ──
ssh narrows 'bash -lc "cd $REPO/scratch/2026_09_20_de_higher_order && \
  /carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python scripts/preflight.py \
  --models_root /carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models \
  --genome /carter/users/aklie/data/ref/genomes/hg38/hg38.fa --all-cts"'

# ── cluster: GPU jobs (submit in parallel) ──
ssh narrows 'bash -lc "cd $REPO/scratch/2026_09_20_de_higher_order && \
  sbatch scripts/run_tracks.sh && sbatch scripts/run_triples.sh"'

# ── sync results back ──
rsync -az narrows:$REPO/scratch/2026_09_20_de_higher_order/results/ results/

# ── local: plot ──
python scripts/plot_tracks.py \
  --tracks results/tracks --pairs inputs/pair_arrangements.tsv \
  --singles inputs/single_motifs.tsv --ct_metadata config/cell_type_metadata.tsv \
  -o figures --common-ylim-dir figures/common_scale

python scripts/plot_mt_fig4.py \
  --mt_summary ../2026_09_06_de_synergy/results/mt_de4/summary__all.tsv \
  --mt_null_thresholds ../2026_09_06_de_synergy/tables/mt_null_thresholds.txt \
  --ct_metadata config/cell_type_metadata.tsv \
  --pairs inputs/pair_arrangements.tsv -o figures

python scripts/plot_distance_curves.py \
  --grids ../2026_09_06_de_synergy/results/de4 \
  --calls inputs/pair_arrangements.tsv -o figures

# ── local: analyze triples (after results synced) ──
python scripts/analyze_triples.py
```

## 3-Way Design

- **Anchor**: lock pair AB at its published optimal (orientation, inner-edge gap)
- **Slide**: third motif C on both flanks (left of A, right of B), both orientations, 0–200 bp
- **Grid**: 2 flanks × 2 C-orientations × 201 gaps = 804 arrangements per (pair, third)
- **Calling metric**: incremental Δ = ΔJ_ABC_opt − (ΔJ_AB + ΔC)
- **Null**: 6 target pairs × 7 lineage-mismatched catalog motifs = 42 null triples; p95 thresholds
- **Also report**: full-additive Δ = ΔJ_ABC_opt − (ΔA + ΔB + ΔC)
