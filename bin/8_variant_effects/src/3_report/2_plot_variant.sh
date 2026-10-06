#!/bin/bash
#SBATCH --job-name=vep_plot
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH --output=logs/3_report/%x.%j.out
#####
# Per-variant prediction + attribution plots (Stage 3.2). Run from the sandbox dir:
#   sbatch --export=ALL,TRAIT=T2D src/3_report/2_plot_variant.sh
#####
date; echo "Job ${SLURM_JOB_ID}"
source activate eugene_tools
export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/3_report/2_plot_variant.py --config config/config.yaml ${TRAIT:+--trait $TRAIT}
rc=$?; echo "exit=$rc"; date; exit $rc
