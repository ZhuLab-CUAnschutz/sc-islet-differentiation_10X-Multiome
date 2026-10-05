#!/bin/bash
#SBATCH --partition=carter-compute
#SBATCH --account=carter-compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --job-name=neg_pfms
#SBATCH -o slurm_logs/neg_pfms.%j.out
#SBATCH -e slurm_logs/neg_pfms.%j.err

set -euo pipefail

PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
SCRIPT_DIR=/carter/users/aklie/projects/islet_organoid_differentiation/sc-islet-differentiation_10X-Multiome/bin/3_single_task_models/scripts/motifs
SCRIPT=$SCRIPT_DIR/modisco_to_pfm_neg.py
MODELS_DIR=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models
META=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/config/cell_type_metadata.tsv

# Parameters matched to positives
T=0.45
ML=3
FLANK=2
MINSEQ=100

# Cell types = cell_type column (skip header)
CTS=$(tail -n +2 "$META" | cut -f1)

echo "===== neg PFM extraction ====="
for CT in $CTS; do
    H5=$MODELS_DIR/$CT/average/modisco/v1/modisco.object.h5
    OUT=$MODELS_DIR/$CT/average/modisco/v1/pfms/neg
    if [ ! -f "$H5" ]; then
        echo "$CT: MISSING h5 ($H5)"
        continue
    fi
    mkdir -p "$OUT"
    echo "----- $CT -----"
    $PY "$SCRIPT" \
        -m "$H5" \
        -o "$OUT" \
        -t $T \
        -ml $ML \
        -f $FLANK \
        -s $MINSEQ \
        -op "${CT}.counts"
done

echo "===== per-CT neg PFM counts ====="
for CT in $CTS; do
    OUT=$MODELS_DIR/$CT/average/modisco/v1/pfms/neg
    N=$(ls "$OUT"/*.pfm 2>/dev/null | wc -l)
    echo -e "${CT}\t${N}"
done
echo "===== done ====="
