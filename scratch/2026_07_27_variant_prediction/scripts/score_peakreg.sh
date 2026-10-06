#!/bin/bash
#SBATCH --job-name=pr_vscore
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH --output=slurm_logs/%x.%A.out
#####
# Peak-regression (multi-task CREsted) REF/ALT variant scoring — one job, one model, all groups.
# Submit with:  sbatch --export=ALL,BED=inputs/combined_credset.minimal.snps.chrombpnet.bed,OUT=results/peakreg/peakreg.variant_scores.tsv score_peakreg.sh
# Smoke:        sbatch --export=ALL,BED=inputs/smoke_100.bed,OUT=results/peakreg_smoke/peakreg.variant_scores.tsv score_peakreg.sh
#####
date; echo -e "Job ${SLURM_JOB_ID}\n"

# conda base shadows a `source activate` of this venv under SLURM, so call the venv python directly
VENV=/carter/users/aklie/projects/islet_organoid_differentiation/tools/crested/.venv
PY=$VENV/bin/python
NV=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib:$LD_LIBRARY_PATH
$PY -c "import keras; print('keras', keras.__version__, keras.config.backend())"

DATA=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model
CKPT=$DATA/crested/finetuned_v2/checkpoints/28.keras
BW=$DATA/bigwigs
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
SIZES=/carter/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes

mkdir -p "$(dirname "$OUT")"
$PY scripts/score_peakreg.py \
  --checkpoint $CKPT \
  --bed $BED \
  --genome $GENOME \
  --chrom_sizes $SIZES \
  --bigwigs_dir $BW \
  --seq_len 2114 \
  --batch_size 128 \
  --out $OUT
echo "exit=$?"
date
