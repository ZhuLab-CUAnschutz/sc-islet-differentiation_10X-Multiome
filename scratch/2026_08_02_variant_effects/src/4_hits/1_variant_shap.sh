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
source activate chrombpnet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib

SHAP=/carter/users/aklie/opt/variant-scorer/src/variant_shap.py
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
SIZES=/carter/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes
MODEL_ROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models
CELLGROUPS=(DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2 SC_delta_GHRL \
        early_ENP early_SC_EC early_SC_alpha early_SC_beta exocrine \
        late_ENP late_SC_EC late_SC_alpha late_SC_beta liver proliferating_endocrine)

group=${CELLGROUPS[$SLURM_ARRAY_TASK_ID - 1]}
model=${MODEL_ROOT}/${group}/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
outdir=results/hits/${group}
mkdir -p "$outdir"
[ -f "$model" ] || { echo "MISSING MODEL: $model"; exit 1; }

cmd="python $SHAP -l $LIST -g $GENOME -m $model -o $outdir/${group} -s $SIZES -sc bed"
echo -e "Running:\n$cmd\n"
eval $cmd
rc=$?; echo "exit=$rc"; date; exit $rc
