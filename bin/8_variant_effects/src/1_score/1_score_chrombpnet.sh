#!/bin/bash
#SBATCH --job-name=cbp_vscore
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --output=logs/1_score/%x.%A.%a.out
#####
# ChromBPNet variant scoring — one array task per cell group, all scoring the SAME
# combined credible-set list once. Repeat of ChromBetaNet §9 on the 22 developmental groups.
# Submit with:
#   sbatch --array=1-22%4 --export=ALL,LIST=inputs/combined_credset.minimal.snps.chrombpnet.bed,OUTROOT=results/chrombpnet score_chrombpnet.sh
# Smoke:
#   sbatch --array=1-2%2  --export=ALL,LIST=inputs/smoke_100.bed,OUTROOT=results/chrombpnet_smoke score_chrombpnet.sh
#####
date; echo -e "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}\n"

source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
source activate "$CHROMBPNET_ENV"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

SCORER=$SCORER_DIR/variant_scoring.py
RANDOM_SEED=1234

# --- cell groups (single-task ChromBPNet model names), from the shared metadata ---
# LC_ALL=C is required: it reproduces the original array order, so a given --array index
# always maps to the same cell group. A locale-aware sort reorders them.
mapfile -t CELLGROUPS < <(awk -F'\t' 'NR>1{print $1}' "$DATASET/config/cell_type_metadata.tsv" | LC_ALL=C sort)

group=${CELLGROUPS[$SLURM_ARRAY_TASK_ID - 1]}
model=${MODEL_ROOT}/${group}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
outdir=${OUTROOT}/${group}
mkdir -p "$outdir"

echo -e "Group:  $group\nModel:  $model\nList:   $LIST\nOut:    $outdir\n"
[ -f "$model" ] || { echo "MISSING MODEL: $model"; exit 1; }

cmd="python $SCORER \
-l $LIST \
-g $GENOME \
-m $model \
-o $outdir/$group \
-s $SIZES \
-r $RANDOM_SEED \
--no_hdf5 \
-sc bed"
# note: no -t/--total_shuf → uses the scorer default --num_shuf 10 (10 shuffled nulls per SNP,
# i.e. the background scales with the variant set rather than a fixed 1M)
echo -e "Running:\n$cmd\n"
eval $cmd
rc=$?
echo "exit=$rc"
date
exit $rc
