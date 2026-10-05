#!/bin/bash
#####
# Marginalize the `constituent` MEME (751 constituent+consensus motifs) across all
# 22 per-cell-type ChromBPNet models. Explicit python (conda activate fails in batch).
# USAGE:
# sbatch \
# --job-name=marg_constituent \
# --account carter-gpu \
# --partition carter-gpu \
# --gpus=a30:1 \
# --mem=32G \
# -n 4 \
# -t 14-00:00:00 \
# --array=1-22%10 \
# --output /carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/bin/slurm_logs/3_single_task_models/marginalization/constituent/%x.%A.%a.out \
# marginalize_global.sh
#####
set -e
date; echo -e "Job $SLURM_JOB_ID Task $SLURM_ARRAY_TASK_ID\n"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cellar/users/aklie/opt/miniconda3/envs/chrombpnet/lib
$PY -c "import torch; print('cuda', torch.cuda.is_available())"

DATA=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results
PEAK=$DATA/2_process_data/peak_calls/all
MROOT=$DATA/3_single_task_models
MOTIF=$MROOT/marginalization/constituent/query.meme
GENOME=/cellar/users/aklie/data/ref/genomes/hg38/hg38.fa

celltypes=(DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2 SC_delta_GHRL early_ENP early_SC_EC early_SC_alpha early_SC_beta exocrine late_ENP late_SC_EC late_SC_alpha late_SC_beta liver proliferating_endocrine)
ct=${celltypes[$SLURM_ARRAY_TASK_ID - 1]}
model=$MROOT/models/$ct/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
peak=$PEAK/$ct.narrowPeak
outdir=$MROOT/models/$ct/fold_0/chrombpnet/0.5/marginalization/constituent
echo -e "CT $ct\nmodel $model\npeak $peak\nout $outdir\n"; mkdir -p $outdir
$PY /cellar/users/aklie/projects/ML4GLand/chrombpnet/scripts/marginalization.py \
  -m $model -f $MOTIF -g $GENOME -p $peak -o $outdir -n 100
echo "DONE $ct"; date
