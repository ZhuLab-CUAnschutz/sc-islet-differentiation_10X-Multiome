# 7_sequence_design: in-silico enhancer optimization

Optimize a candidate enhancer for predicted cell-type specificity using the 22-output CREsted
multitask accessibility model, then explain the result by attributing each step and calling
seqlets on it. The worked example is the experimentally active zebrafish SOX32 enhancer, optimized
for definitive endoderm (DE). It follows the CREsted sequence-evolution tutorial, but starts from
Han's tested construct rather than from random DNA.

## Run order

Cluster paths live in `paths.sh`, the only file to edit elsewhere. Outputs go to `$DESIGN_DIR`
(default `results/7_sequence_design/`); the repo holds only code and the frozen inputs. Submit
from this directory after `mkdir -p slurm_logs`.

| # | Step | Where | Command |
|---|---|---|---|
| — | One-time env build | login | `bash envs/setup_crested_env.sh` |
| 0–2 | Preflight → evolution → candidates | GPU, ~4 h | `sbatch slurm/run_optimize.sh` |
| 3–4 | Attributions → seqlet annotation | GPU | `sbatch slurm/run_attributions.sh` |

Offline checks, no GPU or model needed:

    python scripts/00_preflight.py --skip-model
    python -m unittest discover -s tests -v

`00_walkthrough.ipynb` interprets the results from cached outputs only.

## Design

- **Frozen input:** the 2,114-bp construct, pinned by SHA-256 in `config.py`.
- **Mutable interval:** construct positions 507–1607 (0-based, half-open), the 1,100-bp enhancer.
  Both 507-bp flanks are protected, covering vector, minimal promoter and eGFP.
- **Model:** finetuned multitask checkpoint 23, with the original 22-class output order.
  That order is the bigWig filename order used in training, **not** the display order in
  `config/cell_type_metadata.tsv`, so `config.MODEL_CLASSES` is the authority.
- **Evolution:** exhaustive greedy single-nucleotide mutation, 20 steps from wild type, with
  snapshots at steps 5, 10 and 20.

Three trajectories separate activity from specificity:

1. **DE activity** takes the mutation with the largest DE prediction.
2. **CREsted default** uses the package's weighted-difference optimizer.
3. **DE specificity** maximizes `log1p(DE)` minus an equally weighted combination of the mean and
   the maximum `log1p` off-target prediction, with a strong penalty for dropping below the
   wild-type DE prediction.

The maximum-off-target term matters because the wild-type construct is already strongest in DE
(169), while late SC-beta (111) and early SC-beta (88) remain substantial off-targets.

## Reference result

From SLURM job 1047552 (`carter-gpu-02`, CREsted 1.7.1, Keras PyTorch backend). The preflight
reproduced the June TensorFlow wild-type predictions to within 0.0144 prediction units.

| Candidate | Net substitutions | DE | Strongest non-DE | DE / strongest non-DE |
|---|---:|---:|---:|---:|
| WT | 0 | 169.1 | 111.4 (late SC-beta) | 1.52 |
| DE specificity, step 5 | 5 | 569.0 | 189.5 (PGT3) | 3.00 |
| DE specificity, step 10 | 10 | 748.6 | 231.1 (PGT3) | 3.24 |
| DE specificity, step 20 | 20 | 911.2 | 265.8 (PGT3) | 3.43 |
| CREsted default, step 20 | 20 | 3433.8 | 2791.1 (PGT2) | 1.23 |
| DE activity, step 20 | 20 | 3850.6 | 3908.1 (PGT2) | 0.99 |

The specificity trajectory improves the DE-to-strongest-off-target ratio 2.26-fold at step 20,
whereas optimizing raw DE activity eventually makes PGT2 stronger than DE. Step 10 is a reasonable
activity/edit-burden compromise. The first specificity mutation alone (construct T1322A, enhancer
base 815) is also attractive: DE is essentially unchanged (169.1 → 170.3) while the maximum
off-target falls from 111.4 to 65.9.

These are **model-nominated constructs, not reporter validation.** The full trajectory is kept so
any panel can be extracted without rerunning the GPU evolution.

## Outputs (`$DESIGN_DIR`)

- `results/evolution.npz`: every sequence and all 22 predictions at every step.
- `results/preflight.json`, `results/run_metadata.json`: validation, checkpoint hash, package
  versions, parameters, host.
- `tables/trajectory_metrics.tsv`, `trajectory_predictions_long.tsv`: per-step metrics and all
  per-cell-type predictions.
- `tables/candidate_summary.tsv`, `candidate_mutations.tsv`, `candidate_sequences.fa`: the WT plus
  nine step-budget candidates, their net substitutions, and synthesis-ready full constructs.
- `tables/pareto_steps.tsv`: non-dominated DE versus maximum-off-target points.
- `tables/delta_seqlets.tsv`, `seqlet_summary.tsv`, `beds/seqlet_annotations.bed`: motifs gained,
  lost or changed at each step.
- `figures/`: activity, specificity, per-cell-type and mutation summaries.

All construct positions are reported both 0-based and 1-based, and enhancer-relative positions
both ways, to keep design and ordering coordinates unambiguous.

## Gotchas

- **Two different CREsted checkpoints are in play.** Design uses `finetuned/checkpoints/23.keras`
  (`SOX32_MODEL`). The synergy multitask cross-check in `bin/6_synergy` uses
  `finetuned_v2/checkpoints/28.keras`. Do not mix them.
- **Model output order is not display order.** Always index predictions through
  `config.MODEL_CLASSES`, never through the sorted cell-type metadata.
- **Motif labels key on `short_id`.** The MEME names in `inputs/catalog_plus_foxh1.meme` are
  short IDs (`ST01`, `ST05_sub1`), whereas `motif_catalog.tsv`'s `motif_name` column holds the
  upstream cluster id (`Average_131`). The original Sep 20 script keyed the label map on
  `motif_name`, which matched only the 6 sub-motifs; the other 68 silently fell back to the bare
  short id. Fixed here, so seqlet labels in `delta_seqlets.tsv` are now resolved for all 74
  motifs. Earlier delivered tables have the unresolved labels.
- **The preflight is a real gate.** It requires the cross-backend wild-type predictions to
  reproduce the June TensorFlow reference within 1% or 0.5 prediction units before evolution
  starts. Do not skip it on a new cluster; `--skip-model` is for offline file checks only.
- **Don't automatically take step 20.** Choose from the activity/specificity trajectory.

## Provenance

Promoted from two sandboxes that were sequential phases of one project: steps 0–2 and the tests
from `2026_09_07_sox32_enhancer_optimization`, steps 3–4 from
`2026_09_20_sox32_enhancer_optimization`. The cross-sandbox paths (evolution npz, motif catalog)
now resolve through `config.py`, and the cell-type metadata comes from the repo `config/` instead
of a stale local copy.

Environment: `envs/setup_crested_env.sh` builds a venv on the base torch env, adding CREsted 1.7.1
and Keras 3.14 with the PyTorch backend.
