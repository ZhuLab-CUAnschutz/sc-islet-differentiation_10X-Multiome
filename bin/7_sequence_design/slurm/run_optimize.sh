#!/bin/bash
#SBATCH --job-name=sox32_evolve
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x.%j.out
# Steps 0-2: preflight -> greedy sequence evolution -> candidate tables and figures.
# Submit from bin/7_sequence_design/:  mkdir -p slurm_logs && sbatch slurm/run_optimize.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"

if [ ! -x "$PY" ]; then
    echo "Missing $PY; run envs/setup_crested_env.sh before submission" >&2
    exit 1
fi

"$PY" "$BIN/scripts/00_preflight.py"
"$PY" "$BIN/scripts/01_optimize.py"
"$PY" "$BIN/scripts/02_analyze.py"
