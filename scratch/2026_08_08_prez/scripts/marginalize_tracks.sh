#!/bin/bash
# Profile-keeping marginalization for the presentation figures: 22 cell types x ~20 conditions
# x 2 background sets, n=32 (the published first-pass nbg, so scalars reproduce exactly).
# ~1400 forward passes per cell type -- seconds of a30 time each; wall clock is model loading.
#SBATCH --job-name=marg_tracks
#SBATCH --partition=carter-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/marg_tracks.%j.out
set -e; date
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
WD=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_08_08_prez
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
OUT=${OUT:-$WD/results/tracks}
mkdir -p "$OUT" "$WD/logs"
cd "$WD"
$PY "$WD/scripts/marginalize_tracks.py" \
  --models_root "$MROOT" --cwms "$MROOT/motifs/v1.1/cwm/cwms.npz" \
  -g /carter/users/aklie/data/ref/genomes/hg38/hg38.fa \
  --singles "$WD/inputs/single_motifs.tsv" \
  --pairs "$WD/inputs/pair_arrangements.tsv" \
  -o "$OUT" \
  -n "${NSEQS:-32}" --batch_size 256 \
  --bg_sets "${BGSETS:-own,shared}" --shared_bg_ct DE \
  --background_mode "${BGMODE:-nonpeak}" \
  ${CTS:+--cts "$CTS"}
echo DONE; date
