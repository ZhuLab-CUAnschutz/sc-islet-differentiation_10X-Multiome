#!/bin/bash
#SBATCH --job-name=vep_contrast
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH --output=logs/3_report/%x.%A.%a.out
#####
# Cell-type contrast panels (Stage 3.5). From the sandbox dir:
#   single/trait:  sbatch --export=ALL,TRAIT=T2D src/3_report/5_variant_celltype_contrast.sh
#   variety batch: sbatch --array=1-10 --export=ALL,VARIETY=1,MAX=80,PER_CT=5 src/3_report/5_variant_celltype_contrast.sh
#####
date; echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-NA}"
source activate eugene_tools
export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "$BROWSER" ]; then
  N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-1}
  python src/3_report/5_variant_celltype_contrast.py --config config/config.yaml \
    --rank-file results/variants/browser_variants.tsv --max-examples ${MAX:-50} \
    --all-celltypes --chunk $I/$N
elif [ -n "$RANK" ]; then
  N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-1}
  python src/3_report/5_variant_celltype_contrast.py --config config/config.yaml \
    --rank-file results/variants/example_ranking.tsv --max-examples ${MAX:-80} \
    --n-on ${N_ON:-2} --n-off ${N_OFF:-3} --chunk $I/$N
elif [ -n "$VARIETY" ]; then
  N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-1}
  python src/3_report/5_variant_celltype_contrast.py --config config/config.yaml \
    --variety-batch --max-variants ${MAX:-75} --per-celltype ${PER_CT:-5} \
    --n-on ${N_ON:-2} --n-off ${N_OFF:-3} --chunk $I/$N
else
  python src/3_report/5_variant_celltype_contrast.py --config config/config.yaml \
    ${TRAIT:+--trait $TRAIT} ${VARIANTS:+--variants $VARIANTS} --n-on ${N_ON:-3} --n-off ${N_OFF:-3}
fi
rc=$?; echo "exit=$rc"; date; exit $rc
