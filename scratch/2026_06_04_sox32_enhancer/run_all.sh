#!/bin/bash
#####
# Drive the sox32 analysis ENTIRELY on SLURM (never run compute on the head node).
#   bash run_all.sh
# Submits, with dependencies:
#   GPU (a30): 01 predict, 06 sweep x2
#   CPU (carter-compute, big mem): 03 fimo, 07 map, 04 motif, 02 panels, 05 trajectory
# The CPU plotting steps that need the .npz wait on the predict job (afterok).
#####
set -euo pipefail
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
DIR=/cellar/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/scratch/2026_06_04/sox32_enhancer
cd "$DIR"; mkdir -p outputs figures logs
NPZ=outputs/sox32_all.npz
GPU="--partition=carter-gpu --account=carter-gpu --gres=gpu:a30:1 --cpus-per-task=4 --mem=32G --time=4:00:00"
CPU="--partition=carter-compute --account=carter-compute --cpus-per-task=4 --mem=64G --time=1:00:00"
sub() { sbatch --parsable "$@"; }   # echo jobid

# --- GPU: predict + sweeps ---
PRED=$(sub $GPU --job-name=sox32_predict --output=logs/%x.%j.out 01_predict.sh "$NPZ")
sub $GPU --job-name=sox32_keepcenter --output=logs/%x.%j.out 06_context_sweep.sh keepcenter outputs/sox32_sweep_keepcenter.csv
sub $GPU --job-name=sox32_slide      --output=logs/%x.%j.out 06_context_sweep.sh slide      outputs/sox32_sweep_slide.csv

run() { sub $CPU --job-name="$1" ${3:+--dependency=afterok:$3} --output=logs/%x.%j.out --wrap="cd $DIR && $PY -u $2"; }
# CPU steps (carter-compute, 64G). FIMO first; map+trajectory need its CSV; panels+traj need npz.
FIMO=$(run sox32_fimo  "03_fimo_scan.py")
run sox32_motif "04_motif_relationships.py"
run sox32_map    "07_construct_map.py"                        "$FIMO"
run sox32_panels "02_panels.py $NPZ construct"                "$PRED"
run sox32_traj   "05_contrib_trajectory.py $NPZ construct"    "$PRED:$FIMO"

squeue -u "$USER" -o "%.10i %.16j %.2t %.20E %R"
echo "All steps submitted to SLURM (predict jobid=$PRED; CPU plot steps gated on it)."
