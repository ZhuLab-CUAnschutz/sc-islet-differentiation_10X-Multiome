#!/bin/bash
#SBATCH --job-name=sox32_attr
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x.%j.out
# Steps 3-4: CREsted EIG attributions (GPU) -> seqlet calling and motif annotation (CPU).
# Needs results/evolution.npz from run_optimize.sh.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"

if [ ! -x "$PY" ]; then
    echo "Missing $PY; run envs/setup_crested_env.sh before submission" >&2
    exit 1
fi

echo "=== Step 3: CREsted EIG attributions (GPU) ==="
"$PY" "$BIN/scripts/03_compute_attributions.py"

echo "=== Step 4: seqlet calling + annotation ==="
"$PY" "$BIN/scripts/04_annotate_seqlets.py"
