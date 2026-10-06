# SOX32 enhancer optimization

Optimize the experimentally active zebrafish SOX32 enhancer for predicted definitive-endoderm
(DE) specificity using the 22-output CREsted multitask accessibility model. This follows the
CREsted sequence-evolution tutorial while starting from Han's tested construct rather than from
random DNA.

## Design

- Frozen input: the same 2,114-bp construct used in the 2026-06-04 analysis.
- Mutable interval: construct positions 507–1607 (0-based, half-open), the 1,100-bp enhancer.
- Protected sequence: both 507-bp flanks, including vector, minimal promoter, and eGFP.
- Model: finetuned multitask checkpoint 23, with the original 22-class output order.
- Evolution: exhaustive greedy single-nucleotide mutation, 20 steps from WT.
- Candidate snapshots: steps 5, 10, and 20 from each trajectory.

Three trajectories distinguish activity from specificity:

1. **DE activity** selects the mutation with the largest DE prediction.
2. **CREsted default** uses the package's weighted-difference optimizer: increase DE while
   subtracting the summed off-target change scaled by the total number of model outputs.
3. **DE specificity** maximizes
   log1p(DE) minus an equally weighted combination of the mean and maximum log1p off-target
   prediction. A strong soft constraint discourages any drop below the WT DE prediction.

The maximum-off-target term matters here because the WT construct is already strongest in DE
(169), but late SC-beta (111) and early SC-beta (88) remain substantial off-targets. The final
selection should use the full activity–specificity trajectory, not automatically the 20th step.

## Result

SLURM job 1047552 completed on `carter-gpu-02` using CREsted 1.7.1 with the Keras PyTorch
backend. The model and sequence preflight passed, including reproduction of the June WT
predictions (maximum absolute difference 0.0144 prediction units).

| Candidate | Net substitutions | DE | Strongest non-DE | DE / strongest non-DE | DE fraction |
|---|---:|---:|---:|---:|---:|
| WT | 0 | 169.1 | 111.4 (late SC-beta) | 1.52 | 16.8% |
| DE specificity, step 5 | 5 | 569.0 | 189.5 (PGT3) | 3.00 | 33.5% |
| DE specificity, step 10 | 10 | 748.6 | 231.1 (PGT3) | 3.24 | 40.6% |
| DE specificity, step 20 | 20 | 911.2 | 265.8 (PGT3) | 3.43 | 46.9% |
| CREsted default, step 20 | 20 | 3433.8 | 2791.1 (PGT2) | 1.23 | 21.9% |
| DE activity, step 20 | 20 | 3850.6 | 3908.1 (PGT2) | 0.99 | 14.2% |

The custom specificity trajectory therefore improves the predicted DE-to-strongest-off-target
ratio by 2.26-fold at step 20, whereas optimizing raw DE activity eventually makes PGT2 stronger
than DE. Step 10 is a useful activity/edit-burden compromise: relative to WT, predicted DE is
4.43-fold higher and the DE-to-maximum-off-target ratio is 2.13-fold better. The first specificity
mutation alone (construct T1322A; enhancer base 815) is also experimentally attractive: DE remains
nearly unchanged (169.1 to 170.3) while the maximum off-target falls from 111.4 to 65.9.

These are model-nominated constructs, not reporter validation. A practical first panel would span
WT, the one-mutation specificity design, the 5- and 10-mutation specificity designs, and one
activity-oriented control. The complete trajectory is retained so the exact panel can be extracted
without rerunning the GPU evolution.

## Layout and run order

- inputs: frozen construct, June WT predictions, and cell-type metadata
- scripts/00_preflight.py: validates sequence identity, coordinates, API, model shape, output order,
  and reproduction of the June WT prediction
- scripts/01_optimize.py: GPU sequence evolution and raw trajectory cache
- scripts/02_analyze.py: candidate tables, FASTA, Pareto steps, and figures
- 00_walkthrough.ipynb: result interpretation using only cached outputs
- results: raw evolution cache and provenance
- tables: trajectories, candidates, mutations, and candidate FASTA
- figures: activity, specificity, cell-type, and mutation summaries
- logs: SLURM output

Run the file-only checks locally:

    python scripts/00_preflight.py --skip-model
    python -m unittest discover -s tests -v

On Narrows:

    bash scripts/setup_carter_env.sh
    sbatch scripts/run_gpu.sh

The one-time setup creates a scoped environment inside the experiment, reusing the existing
eugene_tools PyTorch installation and adding CREsted 1.7.1 plus Keras 3.14. The job uses the Keras
PyTorch backend because the original Cellar TensorFlow environment is not mounted on Carter GPU
nodes. Its preflight requires the cross-backend WT predictions to reproduce the June TensorFlow
reference within 1% or 0.5 prediction units before evolution begins.

The job requests one A30 GPU, four CPUs, 32 GB RAM, and four hours. Override the checkpoint without
editing code by setting SOX32_MODEL before submission.

## Outputs

- results/evolution.npz: every sequence and 22-cell-type prediction at every step
- results/preflight.json: package/API/model/reference-prediction validation
- results/run_metadata.json: checkpoint hash, package versions, parameters, and host
- tables/trajectory_metrics.tsv: DE, mean/max off-target, DE fraction/rank, mutation, and objective
- tables/trajectory_predictions_long.tsv: all per-cell-type predictions
- tables/candidate_summary.tsv: WT plus the nine step-budget candidates
- tables/candidate_mutations.tsv: net substitutions in construct and enhancer coordinates
- tables/candidate_sequences.fa: synthesis-ready full-construct sequences
- tables/pareto_steps.tsv: non-dominated DE versus maximum-off-target trajectory points
- figures/fig5_before_after_specificity_grid.*: WT bars versus optimized predictions at
  specificity steps 1, 5, 10, and 20
- figures/fig5a–fig5d: full-size before/after plots for each specificity candidate

All construct positions are reported both 0-based and 1-based. Enhancer-relative positions are
also reported both ways to keep design and ordering coordinates unambiguous.
