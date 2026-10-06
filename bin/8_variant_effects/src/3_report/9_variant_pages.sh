#!/bin/bash
#SBATCH --job-name=vep_pages
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH --output=logs/3_report/%x.%A.%a.out
#####
# Per-variant browser page figures (Stage 3.9). From the sandbox dir:
#   one variant:   sbatch --export=ALL,VARIANT=chr13:75543094:T:A src/3_report/9_variant_pages.sh
#   full set:      sbatch --array=1-10%3 --export=ALL,MAX=50 src/3_report/9_variant_pages.sh
#####
date; echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-NA}"
source activate eugene_tools
export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "$VARIANT" ]; then
  python src/3_report/9_variant_pages.py --config config/config.yaml --variant "$VARIANT" ${FORCE:+--force}
else
  N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-1}
  python src/3_report/9_variant_pages.py --config config/config.yaml \
    --max-examples ${MAX:-50} --chunk $I/$N ${FORCE:+--force}
fi
rc=$?; echo "exit=$rc"; date; exit $rc
