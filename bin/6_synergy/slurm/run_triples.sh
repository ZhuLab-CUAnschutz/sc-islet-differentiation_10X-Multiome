#!/bin/bash
#SBATCH --job-name=syn_triples
#SBATCH --partition=carter-gpu
#SBATCH --account=carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/syn_triples.%A_%a.out
# 3-way synergy: slide a third motif around a locked pair. Task 0 = target triples,
# task 1 = the lineage-mismatched null that sets the calling thresholds.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
HO=${HO_DIR:-$SYN/higher_order}
CT=${CT:-DE}

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== target triples ==="
    $PY $BIN/scripts/marginalize_triples.py \
        --models_root $MROOT --cwms ${HO_CWMS:-$HO/inputs/cwms_focus.npz} -g $GENOME \
        --triples $HO/inputs/triples.tsv --out $HO/results/triples \
        --ct $CT -n ${NBG:-100} --seed 1234 --batch_size 512
else
    echo "=== null triples ==="
    $PY $BIN/scripts/marginalize_triples.py \
        --models_root $MROOT --cwms ${NULL_CWMS:-$HO/inputs/cwms_catalog.npz} -g $GENOME \
        --triples $HO/inputs/null_triples.tsv --out $HO/results/null_triples \
        --ct $CT -n ${NBG:-100} --seed 1234 --batch_size 512
fi
