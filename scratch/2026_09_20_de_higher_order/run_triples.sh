#!/bin/bash
#SBATCH --job-name=ho_triples
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/ho_triples.%A_%a.out

REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_09_20_de_higher_order
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python

export LD_LIBRARY_PATH=/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib:$LD_LIBRARY_PATH

cd $WD

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== target triples ==="
    $PY scripts/marginalize_triples.py \
        --models_root $MROOT \
        --cwms inputs/motifs_de4.npz \
        --genome $GENOME \
        --triples inputs/triples.tsv \
        --out results/triples \
        --ct DE \
        -n 100 --seed 1234 --batch_size 512
else
    echo "=== null triples ==="
    $PY scripts/marginalize_triples.py \
        --models_root $MROOT \
        --cwms inputs/cwms_v1.1_plus_jaspar.npz \
        --genome $GENOME \
        --triples inputs/null_triples.tsv \
        --out results/null_triples \
        --ct DE \
        -n 100 --seed 1234 --batch_size 512
fi
