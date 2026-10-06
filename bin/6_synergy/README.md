# 6_synergy: in-silico pairwise motif synergy (TF-binding syntax)

Insert two catalog motifs into GC-matched inaccessible backgrounds across every orientation and
spacing, predict with the single-task ChromBPNet model of each cell type, and ask whether the joint
effect exceeds the additive expectation. The method follows Liu et al., *Nature* 2026,
"Multiomics and deep learning dissect regulatory syntax in human development", §"In silico
marginalizations to assess motif synergy". Here it is extended from their 138 pre-filtered
composites to **all pairs of the 73-motif catalog × all 22 cell types**.

## Run order

All paths come from `paths.sh`, the only file to edit on a new cluster. Submit SLURM jobs from
this directory (`mkdir -p slurm_logs` first). Outputs go to `results/6_synergy/` (`$SYN`), or
to `$SYNERGY_DIR` if set.

| # | Step | Where | Command |
|---|---|---|---|
| 0 | Install check | GPU, ~5 min | `sbatch slurm/run_smoke.sh` |
| 1 | Pair manifest + chunks | CPU, seconds | `python scripts/build_pair_manifest.py --meta $CATALOG/metadata.tsv --cell_types ../../config/cell_type_metadata.tsv -o $SYN/inputs` |
| 2 | Sweep | GPU array | `sbatch slurm/run_pairs.sh` (set `--array=1-<n_chunks>` from step 1) |
| 3 | Calls, clustering, signatures, report | CPU | `sbatch slurm/run_downstream.sh` |

A is always placed upstream, so A/B order sets the orientation labels and file names. If new
results are meant to merge with the first run, add `--orient_like <first-run tables/calls_long.tsv>`
in step 1. With that flag the builder reproduces the original manifest exactly.

Step 3 runs `extract_frac_pos.py`, `downstream.py`, `explore_signatures.py`, `plot_split.py`,
`make_logos.py` and `build_report.py` in that order. It works on a partial sweep, so you can look
at results while chunks are still landing.

**Cost:** about 8.5 s per pair × CT at n=32 backgrounds and gaps 0–50 on an A30. The full
2,701 pairs × 22 CT run is 91 chunks of about 1.8 h each, roughly 140 GPU-h, or about a day at
`%6` concurrency. To test new pairs, pass a smaller manifest (`--motifs`, `--cts`).

## Inputs

- Single-task models: `$MROOT/models/<ct>/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5`
- Backgrounds: `$MROOT/models/<ct>/fold_0/chrombpnet/0.5/auxiliary/filtered.nonpeaks.bed`
  (ChromBPNet's GC-matched negatives), plus hg38
- Motifs: catalog v1.1 `cwm/cwms.npz` (signed CWMs keyed by `short_id`) and `metadata.tsv`
- Cell-type facets: repo `config/` (`cell_type_metadata.tsv`, `lineage_metadata.tsv`,
  `endocrine_metadata.tsv`)

## Outputs (`$SYN`)

- `sweep/chunk_NNN/summary__all.tsv`: one row per pair × CT, giving dJ_opt, dS, delta, optimal
  orientation/gap/center-to-center distance, maxZ and Wilcoxon p.
- `sweep/chunk_NNN/<A>__<B>__<ct>.npz`: the **full orientation × gap ΔJ grid** plus per-background
  joint and additive effects at the optimum. Always keep these; the report draws from them.
- `tables/calls_long.tsv`: the main table, with p/q per pair × CT, `synergistic`, `call`
  (hard/soft), `frac_pos` and motif annotations. Also `pair_annot`, `delta_wide`,
  `cluster_labels`, `block_cluster_labels`, `lineage_signatures`, `pair_groups`,
  `pair_ct_synergy_binary`.
- `figures/`: clustermaps (all / endocrine / non-endocrine), lineage-signature matrices,
  orientation QC.
- `report/`: the interactive HTML bundle (index, methods overview, all-pairs table, by-cell-type
  table, xlsx). All thresholds are live controls in the page.

## Definitions

- **ΔJ**: log-count change for both motifs inserted at one arrangement, averaged over backgrounds.
  ΔA and ΔB are each motif inserted alone at the center, and **ΔS = ΔA + ΔB** is the additive
  expectation.
- **Δ = ΔJ_opt − ΔS**, where ΔJ_opt is the single best arrangement (argmax over orientation × gap).
  This is the same as the paper.
- **Orientations:** FF/FR/RF/RR with A always upstream. Palindromes are deduplicated: if A is
  palindromic, use FF/FR; if B is palindromic, use FF/RF; for a homodimer, use FF/FR/RF.
- **Calling:**
  - For each CT, a robust null (median + MAD σ) is fit on the central 2.5–97.5% of that CT's Δ.
    This gives a right-tail p for every pair × CT.
  - All pair × CT p-values are pooled for **one global BH**.
  - **Synergistic** = q_delta < 0.05.
  - **Hard** = synergistic and q_maxZ < 0.05, where maxZ is the argmax cell against the
    pair's own orientation × gap grid. **Soft** = the remaining synergistic calls.
- **frac_pos**: the fraction of backgrounds where joint > additive at the optimum. It is reported
  but **not used as a gate**. The Wilcoxon test is also reported but not gated, because at the
  argmax arrangement it saturates (q < 0.05 for 76% of all tests).

Reference numbers from the first full run: 2,180 / 59,422 pair × CT synergistic (403 hard, 1,777
soft) and 660 / 2,701 pairs synergistic in at least one CT. The endocrine vs non-endocrine split
is 192 endocrine-only, 268 non-endocrine-only and 200 shared.

## Gotchas

- **Do not use the paper's thresholds (Δ > 0.15, Z > 4) here.** With a single fold and all
  pairs, they call 54% of lineage-mismatched null pairs synergistic. The argmax over about 200
  arrangements from one fold suffers from the winner's curse. The in-matrix empirical null above
  replaces the earlier 42-pair hand-picked null, which was too small for BH and about 14%
  contaminated.
- **Sub-motifs have `curator_keep = NaN`.** The six ST05/ST36 splits (ISL1, NKX6-1, LMX1A, MIXL1,
  HNF4G, RARB) postdate curation. Filtering on `curator_keep == "keep"` silently drops them, and
  they are the β/EC determinants. `build_pair_manifest.py` only excludes explicitly
  ("No expression" → ST53).
- **The motif inserted is not the motif shown.** Synergy inserts the CWM argmax consensus trimmed
  at 10% of peak importance (`imp_frac=0.1`). For catalog v1.1 that equals the stored width for
  every motif, so the inserted motif and the logo agree. Re-check this if you change the catalog.
- **Homodimers are not under-called.** Their synergy rate is 9.7%, against 3.5% for distinct
  pairs.
- **Orientation:** among synergistic calls RR is enriched (722 RR vs 387 FF), against a flat
  baseline across all pairs. Whether that is real geometry or a consensus/strand asymmetry is
  still open.
- **Fine per-CT specificity is noisy.** The robust signal is lineage-branch level (endocrine vs
  foregut/progenitor). Single-task and multitask calls agree 88% on that axis but only 18% on fine
  lineage (July ST-vs-MT comparison).

## Scripts

| Script | What | From (sandbox) |
|---|---|---|
| `build_pair_manifest.py` | all-pairs × all-CT manifest + pair-major chunks | new; reproduces the `pairs_fp_pairmajor.tsv` / `fp_chunks/` used for the first run |
| `marginalize_pairs.py` | the sweep (single-task ChromBPNet, tangermeme) | `projects/.../sandbox/2026_07_19_syntax_pairwise_marg/scripts` (identical copy in `scratch/2026_09_06_de_synergy`) |
| `marginalize_pairs_mt.py` | same sweep through the CREsted multitask model (one pass scores all 22 CT); a cross-check, needs the `.venv-keras` env + MT checkpoint | same |
| `extract_frac_pos.py` | frac_pos from the per-pair npz | new; was an ad hoc cluster one-liner |
| `downstream.py` | aggregate → empirical null + BH calls → clustering | `sandbox/2026_08_06_syntax_final/scripts` |
| `explore_signatures.py`, `plot_split.py` | binary lineage signatures, developmental taxonomy, block figures | same |
| `make_logos.py` | CWM logos for the report | new; logos had been generated ad hoc |
| `build_report.py` | interactive HTML bundle | same as downstream |
| `build_higher_order_inputs.py` | focus-set manifests for tracks/triples | generalized from `scratch/2026_09_20_de_higher_order/scripts/build_inputs.py` (was DE-hardwired) |
| `marginalize_tracks.py`, `tracks_io.py` | profile-head marginalization + track math | `scratch/2026_09_20_de_higher_order` (newer copy of the `2026_08_08_prez` original) |
| `marginalize_triples.py`, `analyze_triples.py` | 3-way sweep and incremental calling | same |
| `preflight.py`, `plot_tracks.py`, `plot_distance_curves.py`, `plot_mt_fig4.py` | path/geometry checks and figures | same |

## Higher-order: profile tracks and 3-way synergy

Two extensions that run on a small focus set of TFs after the pairwise sweep has called their
arrangements. Both reuse `paths.sh`; work dir is `$HO_DIR` (default `$SYN/higher_order`).

**Profile tracks.** The pairwise sweep keeps only counts-head scalars, so no prediction track can
be reconstructed from it. `marginalize_tracks.py` re-runs chosen arrangements keeping the profile
head, which is what the stacked +A / +B / Σ / observed figures are drawn from. Track math lives in
`tracks_io.py`: average the profile in **log** space, renormalize, then scale by
`exp(mean log-counts)`. Averaging linearly instead lets the few highest-count backgrounds dominate
and breaks the identity `area(T_motif) / area(T_bg) == exp(delta_logcounts)`.

**3-way synergy.** `marginalize_triples.py` locks a pair at its pairwise optimum and slides a third
motif on both flanks in both orientations (2 orientations x 2 flanks x gaps, about the cost of one
pairwise sweep). Calling uses the **incremental** metric
`delta_incr = dJ_ABC_opt - (dJ_pair + dC)`, with thresholds at the 95th percentile of a matched
lineage-mismatched 3-way null, floored at the paper's 0.15 / Z 4. This null is matched but small,
so it is a cruder gate than the pairwise in-matrix null described above.

| # | Step | Where | Command |
|---|---|---|---|
| 1 | Build manifests | CPU | `python scripts/build_higher_order_inputs.py --motifs <focus.tsv> --calls <pairwise calls_long.tsv> --cwms <focus cwms.npz> --catalog $CATALOG/metadata.tsv --null_cwms <catalog cwms.npz> -o $HO/inputs` |
| 2 | Sanity-check paths and geometry | CPU | `python scripts/preflight.py` |
| 3 | Profile tracks | GPU | `sbatch slurm/run_tracks.sh` |
| 4 | Triples + matched null | GPU | `sbatch slurm/run_triples.sh` |
| 5 | Call triples | CPU | `python scripts/analyze_triples.py --triples $HO/results/triples --null $HO/results/null_triples --ct DE -o $HO/tables` |
| 6 | Figures | CPU | `plot_tracks.py`, `plot_distance_curves.py`, `plot_mt_fig4.py` |

The focus set is an argument, not baked in. `--motifs` takes a TSV of `short_id` and `curator_tf`
(optionally `curator_category`); the null third motifs are then drawn from the catalog excluding
anything annotated to one of those TFs, spread evenly across the width distribution so the null is
not confounded by motif length. The worked example is the four DE TFs (EOMES `ST18`,
MIXL1 `ST36_sub1`, SOX17 `ST66`, FOXH1 `J_FOXH1`), which gives 12 triples (C(4,3) x 3 anchor
choices) and a 42-row null (6 heterotypic pairs x 7 nulls).

**Gotcha:** `marginalize_triples.py` takes `--cwms` for the focus motifs and the null run needs a
catalog-wide npz, because the null third motifs are not in the focus set. The runner keeps these as
separate variables (`HO_CWMS`, `NULL_CWMS`) for that reason.

## Existing results

The first full run (n=32, gaps 0–50; 2,701 pairs × 22 CT) is under
`/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg/results/full_fp/`,
which uses the same `chunk_NNN/` layout as `sweep/` here. To reuse it without re-running, point
`run_downstream.sh` at it (`ln -s .../full_fp $SYN/sweep`).

## Environment

`envs/eugene_tools.yml`: Python 3.11, torch 2.4.1, tangermeme 1.0.3, and bpnet-lite 0.8.1 from the
adamklie fork at the commit used. `paths.sh` also prepends the `chrombpnet` env's `lib/` to
`LD_LIBRARY_PATH` for libcudnn. Leave `CHROMBPNET_LIB` empty if your torch build finds CUDA on its
own.
