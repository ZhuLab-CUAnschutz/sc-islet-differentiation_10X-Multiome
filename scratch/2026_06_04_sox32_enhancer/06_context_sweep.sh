#!/bin/bash
#####
# Context sweep (GPU). MODE = slide | keepcenter.
#   sbatch --partition=carter-gpu --account=carter-gpu --gres=gpu:a30:1 \
#     --cpus-per-task=4 --mem=32G --time=2:00:00 --job-name=sox32_sweep \
#     --output=logs/%x.%j.out 06_context_sweep.sh keepcenter outputs/sox32_sweep_keepcenter.csv
#####
set -euo pipefail
date; echo "Job ${SLURM_JOB_ID:-local}"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
DIR=/cellar/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_06_04/sox32_enhancer
cd "$DIR"
MODE="${1:-keepcenter}"; OUT="${2:-outputs/sox32_sweep_${MODE}.csv}"
$PY 06_context_sweep.py --mode "$MODE" --out "$OUT" --seeds 0 1 2
date; echo DONE
