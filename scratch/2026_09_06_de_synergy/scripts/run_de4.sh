#!/bin/bash
# DE 4-TF synergy (EOMES/MIXL1/SOX17/FOXH1; catalog+JASPAR FOXH1 vs all-JASPAR), DE model only.
# Full paper fidelity, same as the hero-pair run (run_array.sh): n=100 GC-matched inaccessible
# backgrounds, inner-edge gaps 0-200 at 1-bp, <=4 orientations. 19 unique pairs, single-CT mode.
#SBATCH --job-name=de4_syn
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/de4_syn.%j.out
set -e; date; hostname; nvidia-smi -L
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_09_06_de_synergy
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
CT=DE
MODEL=$MROOT/models/$CT/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
BG=$MROOT/models/$CT/fold_0/chrombpnet/0.5/auxiliary/filtered.nonpeaks.bed
OUT=$WD/results/de4; mkdir -p "$OUT" "$WD/logs"
cd "$WD/scripts"
SECONDS=0
$PY "$WD/scripts/marginalize_pairs.py" \
  -m "$MODEL" --background "$BG" --ct "$CT" \
  --cwms "$WD/inputs/motifs_de4.npz" -g /carter/users/aklie/data/ref/genomes/hg38/hg38.fa \
  --pairs "$WD/inputs/pairs_de4.tsv" -o "$OUT" \
  -n 100 --maxdist 200 --gap_step 1 --batch_size 256 --seed 1234
echo "DONE $CT in ${SECONDS}s"; date
