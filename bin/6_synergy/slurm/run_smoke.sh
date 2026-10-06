#!/bin/bash
#SBATCH --job-name=syn_smoke
#SBATCH --account=carter-gpu
#SBATCH --partition=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=16G
#SBATCH -n 2
#SBATCH -t 0:30:00
#SBATCH --output=slurm_logs/syn_smoke.%j.out
# Install check: FOXA2 x OTX2 (ST16 x ST30) in PGT2, the internal positive control.
# Reference (full run, n=32, gaps 0-50): delta 1.11, maxZ 4.93, called hard. With n=8 / gaps 0-10 expect
# the same sign and a delta in that neighborhood, not an exact match.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"   # submit from bin/6_synergy/
OUT=$SYN/smoke; mkdir -p "$OUT"
printf "idA\tidB\nST16\tST30\n" > "$OUT/pair.tsv"
$PY $BIN/scripts/marginalize_pairs.py \
  --models_root $MROOT --cwms $CWMS -g $GENOME \
  --pairs "$OUT/pair.tsv" --ct PGT2 -o "$OUT" \
  -n 8 --maxdist 10 --gap_step 1 --batch_size 256
cat "$OUT/summary__PGT2.tsv"
