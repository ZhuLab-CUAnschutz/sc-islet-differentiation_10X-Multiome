#!/bin/bash
#SBATCH --job-name=sox32_attr
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

set -euo pipefail

RUN_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$RUN_DIR"

PY="$RUN_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Missing $PY; run scripts/setup_carter_env.sh before submission" >&2
    exit 1
fi
export KERAS_BACKEND=torch
export SOX32_MODEL=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model/crested/finetuned/checkpoints/23.keras

echo "=== Phase 1: CREsted EIG attributions (GPU) ==="
"$PY" scripts/compute_attributions.py

echo ""
echo "=== Phase 2: Seqlet calling + annotation (CPU) ==="
"$PY" scripts/annotate_seqlets.py

echo ""
echo "=== Done ==="
