#!/bin/bash
# Full-fidelity CRESTED multitask rerun: task 0 = 19 DE4 pairs; task 1 = 42-pair null.
# The model scores all 22 heads in each forward pass; downstream reporting selects DE.
#SBATCH --job-name=de4_mt
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/de4_mt.%A_%a.out

set -e
date
hostname
nvidia-smi -L

source /carter/users/aklie/opt/deeptopic/.venv-keras/bin/activate
NV=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib:$LD_LIBRARY_PATH
export KERAS_BACKEND=tensorflow

REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_09_06_de_synergy
DS=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome
MROOT=$DS/results/3_single_task_models
MT=$DS/results/4_multi_task_model/crested

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  PAIRS=$WD/inputs/pairs_de4.tsv
  OUT=$WD/results/mt_de4
  CWMS=$WD/inputs/motifs_de4.npz
  LABEL=targets
else
  PAIRS=$WD/inputs/pairs_mt_null.tsv
  OUT=$WD/results/mt_null_full
  CWMS=$MROOT/motifs/v1.1/cwm/cwms.npz
  LABEL=null
fi

mkdir -p "$OUT"
cd "$WD/scripts"
SECONDS=0
python "$WD/scripts/marginalize_pairs_mt.py" \
  --model "$MT/finetuned_v2/checkpoints/28.keras" \
  --cwms "$CWMS" \
  --adata "$MT/contributions_specific_v2/adata_with_predictions.h5ad" \
  -g /carter/users/aklie/data/ref/genomes/hg38/hg38.fa \
  --peak_glob "$DS/results/2_process_data/peak_calls/all/*.narrowPeak" \
  --pairs "$PAIRS" -o "$OUT" \
  -n 100 --maxdist 200 --gap_step 1 --batch_size 512 --seed 1234

echo "DONE $LABEL in ${SECONDS}s"
date
