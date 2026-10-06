#!/bin/bash
#SBATCH --job-name=cbp_vscore
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --output=slurm_logs/%x.%A.%a.out
#####
# ChromBPNet variant scoring — one array task per cell group, all scoring the SAME
# combined credible-set list once. Repeat of ChromBetaNet §9 on the 22 developmental groups.
# Submit with:
#   sbatch --array=1-22%4 --export=ALL,LIST=inputs/combined_credset.minimal.snps.chrombpnet.bed,OUTROOT=results/chrombpnet score_chrombpnet.sh
# Smoke:
#   sbatch --array=1-2%2  --export=ALL,LIST=inputs/smoke_100.bed,OUTROOT=results/chrombpnet_smoke score_chrombpnet.sh
#####
date; echo -e "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}\n"

source activate chrombpnet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# --- constants ---
# NOTE: carter compute nodes mount /carter, NOT /cellar — use /carter paths for everything
SCORER=/carter/users/aklie/opt/variant-scorer/src/variant_scoring.py
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
SIZES=/carter/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes
MODEL_ROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models
RANDOM_SEED=1234

# --- 22 developmental cell groups (single-task ChromBPNet model names) ---
CELLGROUPS=(DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2 SC_delta_GHRL \
        early_ENP early_SC_EC early_SC_alpha early_SC_beta exocrine \
        late_ENP late_SC_EC late_SC_alpha late_SC_beta liver proliferating_endocrine)

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
echo "exit=$?"
date
