#!/bin/bash
#SBATCH --job-name=ho_tracks
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ho_tracks.%j.out

# ─── paths ───
REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_09_20_de_higher_order
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python

# chrombpnet LD path fix (needed for libcudnn)
export LD_LIBRARY_PATH=/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib:$LD_LIBRARY_PATH

cd $WD

ALL_CTS="DE,ENP_phase1,FB_FLT1,PFG1,PFG2,PGT1,PGT2,PGT3,PP1,PP2,SC_delta_GHRL,early_ENP,early_SC_EC,early_SC_alpha,early_SC_beta,exocrine,late_ENP,late_SC_EC,late_SC_alpha,late_SC_beta,liver,proliferating_endocrine"

$PY scripts/marginalize_tracks.py \
    --models_root $MROOT \
    --cwms inputs/motifs_de4.npz \
    --genome $GENOME \
    --singles inputs/single_motifs.tsv \
    --pairs inputs/pair_arrangements.tsv \
    --out results/tracks \
    --cts $ALL_CTS \
    --bg_sets own,shared \
    --shared_bg_ct DE \
    -n 100 \
    --seed 1234 \
    --batch_size 256
