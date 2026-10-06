#!/bin/bash
#####
# Predict accessibility + count contributions across cell types (GPU).
#   sbatch --partition=carter-gpu --account=carter-gpu --gres=gpu:a30:1 \
#     --cpus-per-task=4 --mem=32G --time=4:00:00 --job-name=sox32_predict \
#     --output=logs/%x.%j.out 01_predict.sh outputs/sox32_all.npz "CELLTYPES_OR_EMPTY"
#####
set -euo pipefail
date; echo "Job ${SLURM_JOB_ID:-local}"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
DIR=/cellar/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_06_04/sox32_enhancer
cd "$DIR"
OUT="${1:-outputs/sox32_all.npz}"; CTS="${2:-}"
CT_ARG=""; [ -n "$CTS" ] && CT_ARG="--celltypes $CTS"
$PY 01_predict.py --out "$OUT" $CT_ARG
date; echo DONE
