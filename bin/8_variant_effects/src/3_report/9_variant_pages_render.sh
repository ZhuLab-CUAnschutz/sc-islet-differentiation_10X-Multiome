#!/bin/bash
#SBATCH --job-name=vep_pages_render
#SBATCH --account carter-compute
#SBATCH --partition carter-compute
#SBATCH --mem=16G
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH --output=logs/3_report/%x.%A.%a.out
#####
# CPU-only re-render of variant pages from cached intermediates (no GPU): use this after
# a style/layout change. Requires Stage 3.9 to have run once (intermediates.npz present).
#   sbatch --array=1-10 --export=ALL,MAX=50 src/3_report/9_variant_pages_render.sh
#####
date; echo "task ${SLURM_ARRAY_TASK_ID:-NA}"
source activate eugene_tools
export PYTHONPATH=src:$PYTHONPATH
N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-1}
python src/3_report/9_variant_pages.py --config config/config.yaml \
  --max-examples ${MAX:-50} --chunk $I/$N --force
rc=$?; echo "exit=$rc"; date; exit $rc
