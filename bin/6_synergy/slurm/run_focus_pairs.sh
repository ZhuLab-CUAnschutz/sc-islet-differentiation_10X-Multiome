#!/bin/bash
#SBATCH --job-name=syn_focus
#SBATCH --account=carter-gpu
#SBATCH --partition=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 8:00:00
#SBATCH --output=slurm_logs/syn_focus.%j.out
# Focus pairwise screen: a few TFs in ONE cell type at full fidelity (n=100, gaps 0-200, 1-bp),
# rather than the all-pairs first-pass settings. Task 0 of the pair is the target set; the matched
# lineage-mismatched null is run the same way and sets the calling thresholds.
#
#   FOCUS=$SYN/focus  CT=DE  sbatch slurm/run_focus_pairs.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
FOCUS=${FOCUS:-$SYN/focus}
CT=${CT:-DE}

echo "=== target pairs ==="
$PY $BIN/scripts/marginalize_pairs.py \
  --models_root $MROOT --cwms $FOCUS/inputs/motifs_focus.npz -g $GENOME \
  --pairs $FOCUS/inputs/pairs_run.tsv --ct $CT -o $FOCUS/results \
  -n ${NBG:-100} --maxdist ${MAXDIST:-200} --gap_step 1 --batch_size 512

# The null uses catalog motifs that are NOT in the focus set, so it needs the catalog cwms.
if [ -f "$FOCUS/inputs/pairs_null.tsv" ]; then
    echo "=== matched null ==="
    $PY $BIN/scripts/marginalize_pairs.py \
      --models_root $MROOT --cwms ${NULL_CWMS:-$CWMS} -g $GENOME \
      --pairs $FOCUS/inputs/pairs_null.tsv --ct $CT -o $FOCUS/results_null \
      -n ${NBG:-100} --maxdist ${MAXDIST:-200} --gap_step 1 --batch_size 512
else
    echo "no pairs_null.tsv; reuse an existing matched null summary for analyze_focus_pairs.py --null"
fi
