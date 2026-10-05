#!/bin/bash
# Step 6 attribution smoke test on carter-gpu.
# Submit: gpu.sh -s attribute.sh -j decima_attr -m 64G -t 04:00:00 -c 4 -g a30:1 \
#           -- --ckpt <best_ckpt> --arm decima --regime lora
date
echo -e "Job ID: ${SLURM_JOB_ID:-local}\n"
source /carter/users/aklie/opt/miniconda3/etc/profile.d/conda.sh && conda activate decima
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python attribute.py "$@"
echo -e "\nDone."; date
