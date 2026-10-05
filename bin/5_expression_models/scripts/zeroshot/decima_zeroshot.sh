#!/bin/bash
# Zero-shot Decima baseline (Step 3). CPU-only in decima 0.7.2 (uses the precomputed
# preds layer), so submit to carter-compute, not a GPU node.
# Submit: gpu.sh -s <this>.sh -j decima_zeroshot -p carter-compute -m 64G -t 02:00:00 -c 4
#   (or: srun -p carter-compute --mem 64G -c 4 --time 2:00:00 bash decima_zeroshot.sh)
#
# Prereqs: decima env built; pseudobulk target built
# (bin/2_process_data/build_pseudobulk_expression.py); state_mapping.tsv curated.

date
echo -e "Job ID: ${SLURM_JOB_ID:-local}\n"

export HF_HOME=/carter/users/aklie/data/decima_cache
export HF_HUB_CACHE=/carter/users/aklie/data/decima_cache/hub
source /carter/users/aklie/opt/miniconda3/etc/profile.d/conda.sh && conda activate decima

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/run_decima_zeroshot.py"

echo -e "\nDone."
date
