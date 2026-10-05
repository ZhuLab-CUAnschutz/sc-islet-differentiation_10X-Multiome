#!/bin/bash
#####
# MT motif marginalization for the MC cluster catalog.
# Differences vs v3:
#   - motif catalog = motif_compendium_cluster/combined_mapped.meme (151 motifs)
#   - outputs to crested/marginalization_strain/{CT}/
# Otherwise identical (finetuned_v2 checkpoint, k=2 shuffled per-CT background peaks).
#
# Submit as a 22-CT array (concurrency 4):
#   sbatch --partition=carter-gpu --account=carter-gpu --gres=gpu:a30:1 \
#     --cpus-per-task=4 --mem=32G --time=14-00:00:00 \
#     --array=1-22%4 --job-name=pr_marg_cluster \
#     --output=$BIN/slurm_logs/4_multi_task_model/%x.%A.%a.out \
#     <this>.sh
#####

date
echo -e "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID\n"

source /cellar/users/aklie/opt/deeptopic/.venv-keras/bin/activate
NVIDIA_LIBS=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:$LD_LIBRARY_PATH
export KERAS_BACKEND=tensorflow

celltypes=(
    DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2
    SC_delta_GHRL early_ENP early_SC_EC early_SC_alpha early_SC_beta
    exocrine late_ENP late_SC_EC late_SC_alpha late_SC_beta liver
    proliferating_endocrine
)
celltype=${celltypes[$SLURM_ARRAY_TASK_ID - 1]}
echo "Cell type: $celltype"

SCRIPT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/bin/4_multi_task_model/scripts/marginalization/marginalize_multitask.py
BASE=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model
PEAK_DIR=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all

n_seqs=100

python ${SCRIPT} \
    --model ${BASE}/crested/finetuned_v2/checkpoints/28.keras \
    --motif_file ${BASE}/crested/motif_compendium_cluster/combined_mapped.meme \
    --genome_fasta /cellar/users/aklie/data/ref/genomes/hg38/hg38.fa \
    --peaks ${PEAK_DIR}/${celltype}.narrowPeak \
    --adata ${BASE}/crested/contributions_specific_v2/adata_with_predictions.h5ad \
    --celltype ${celltype} \
    --output_dir ${BASE}/crested/marginalization_strain \
    --num_seqs ${n_seqs} \
    --batch_size 64

date
