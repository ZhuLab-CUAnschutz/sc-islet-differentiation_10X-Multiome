#!/bin/bash
#SBATCH --job-name=ho_triple_tracks
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/ho_triple_tracks.%j.out

REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_09_20_de_higher_order
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python

export LD_LIBRARY_PATH=/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib:$LD_LIBRARY_PATH

cd $WD

$PY scripts/marginalize_tracks.py \
    --models_root $MROOT \
    --cwms inputs/cwms_v1.1_plus_jaspar.npz \
    --genome $GENOME \
    --singles inputs/single_motifs.tsv \
    --pairs inputs/pair_arrangements.tsv \
    --triples inputs/best_triple_arrangements.tsv \
    --out results/tracks \
    --cts DE \
    --bg_sets own \
    --shared_bg_ct DE \
    -n 100 \
    --seed 1234 \
    --batch_size 256
