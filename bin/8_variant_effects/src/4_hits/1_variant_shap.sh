#!/bin/bash
#SBATCH --job-name=vep_shap
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 08:00:00
#SBATCH --output=logs/4_hits/%x.%A.%a.out
#####
# Stage 4.1: per-cell-type SHAP for the significant-variant set (TF, chrombpnet env).
# Run from the sandbox dir after Stage 4.0:
#   sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed src/4_hits/1_variant_shap.sh
#####
date; echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
source activate "$CHROMBPNET_ENV"

SHAP=$SCORER_DIR/variant_shap.py
# LC_ALL=C reproduces the original array order, so an --array index always maps to the
# same cell group. A locale-aware sort reorders them.
mapfile -t CELLGROUPS < <(awk -F'\t' 'NR>1{print $1}' "$DATASET/config/cell_type_metadata.tsv" | LC_ALL=C sort)

group=${CELLGROUPS[$SLURM_ARRAY_TASK_ID - 1]}
model=${MODEL_ROOT}/${group}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
outdir=results/hits/${group}
mkdir -p "$outdir"
[ -f "$model" ] || { echo "MISSING MODEL: $model"; exit 1; }

cmd="python $SHAP -l $LIST -g $GENOME -m $model -o $outdir/${group} -s $SIZES -sc bed"
echo -e "Running:\n$cmd\n"
eval $cmd
rc=$?; echo "exit=$rc"; date; exit $rc
