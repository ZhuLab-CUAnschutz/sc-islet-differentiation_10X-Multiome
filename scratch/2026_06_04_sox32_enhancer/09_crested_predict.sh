#!/bin/bash
set -euo pipefail
date; echo "Job ${SLURM_JOB_ID:-local}"
PYK=/cellar/users/aklie/opt/deeptopic/.venv-keras/bin/python
DIR=/cellar/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_06_04/sox32_enhancer
cd "$DIR"
$PYK 09_crested_predict.py
date; echo DONE
