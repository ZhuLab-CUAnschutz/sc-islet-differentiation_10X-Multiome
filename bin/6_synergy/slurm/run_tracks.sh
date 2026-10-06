#!/bin/bash
#SBATCH --job-name=syn_tracks
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/syn_tracks.%j.out
# Profile-track marginalization: re-runs the chosen arrangements keeping the PROFILE head, so real
# prediction tracks can be drawn (the pairwise sweep stores counts-head scalars only).
# Needs the manifests from scripts/build_higher_order_inputs.py in $HO/inputs.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
HO=${HO_DIR:-$SYN/higher_order}

ALL_CTS=$(awk -F'\t' 'NR>1{print $1}' "$DATASET/config/cell_type_metadata.tsv" | paste -sd, -)

$PY $BIN/scripts/marginalize_tracks.py \
    --models_root $MROOT --cwms ${HO_CWMS:-$HO/inputs/cwms_focus.npz} -g $GENOME \
    --singles $HO/inputs/single_motifs.tsv \
    --pairs $HO/inputs/pair_arrangements.tsv \
    ${HO_TRIPLES:+--triples $HO_TRIPLES} \
    --out $HO/results/tracks \
    --cts "${CTS:-$ALL_CTS}" \
    --bg_sets own,shared --shared_bg_ct ${SHARED_BG_CT:-DE} \
    -n ${NBG:-100} --seed 1234 --batch_size 256
