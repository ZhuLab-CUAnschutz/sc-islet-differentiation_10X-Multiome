#!/bin/bash
#SBATCH --job-name=syn_pairs
#SBATCH --account=carter-gpu
#SBATCH --partition=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --array=1-91%6
#SBATCH --output=slurm_logs/syn_pairs.%A_%a.out
# All-pairs x all-CT pairwise synergy sweep (single-task ChromBPNet).
# One array task = one pair-major chunk from build_pair_manifest.py (30 pairs x 22 CT, ~1.8 h on an A30
# at n=32 / gaps 0-50). Set --array to 1-<n_chunks> printed by the manifest builder.
# Submit from bin/6_synergy/:  mkdir -p slurm_logs && sbatch slurm/run_pairs.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"   # submit from bin/6_synergy/
date; echo "Job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-}"
$PY -c "import torch;print('cuda',torch.cuda.is_available())"

TASK=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
PAIRS=$SYN/inputs/chunks/chunk_$TASK.tsv
OUT=$SYN/sweep/chunk_$TASK              # per-chunk dir: batch mode always writes summary__all.tsv
mkdir -p "$OUT"

$PY $BIN/scripts/marginalize_pairs.py \
  --models_root $MROOT --cwms $CWMS -g $GENOME \
  --pairs "$PAIRS" -o "$OUT" \
  -n ${NBG:-32} --maxdist ${MAXDIST:-50} --gap_step 1 --batch_size 512

echo "DONE task $TASK"; date
