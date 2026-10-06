#!/bin/bash
# PROX1 variant (chr1:213977102:T:A, rs79687284, T2D DIAMANTE) re-rendered on the ENP-stage
# cell types of the 22-model set, with shared attribution y-limits for the slide.
#SBATCH --job-name=prox1_prez
#SBATCH --partition=carter-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/prox1_prez.%j.out
set -e; date
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
REPO=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome
WD=$REPO/scratch/2026_08_08_prez
VEP=$REPO/scratch/2026_08_02_variant_effects
mkdir -p "$WD/logs" "$WD/results/prox1"
cd "$WD"
$PY "$WD/scripts/plot_prox1.py" \
  --config "$VEP/config/config.yaml" --vep_src "$VEP/src" \
  --variants "${VARIANTS:-chr1:213977102:T:A}" \
  --cell-types "${CTS:-late_ENP,ENP_phase1,early_ENP,proliferating_endocrine}" \
  -o "$WD/results/prox1" --force
echo DONE; date
