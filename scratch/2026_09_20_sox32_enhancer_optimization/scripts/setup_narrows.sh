#!/bin/bash
set -euo pipefail
#
# Sync the Sep 20 seqlet-annotation project to narrows and set up its
# environment.  Run this LOCALLY (from your Mac) before sbatch.
#
# Usage:
#   bash sandbox/2026_09_20_sox32_enhancer_optimization/scripts/setup_narrows.sh
#

LOCAL_BASE="$HOME/Desktop/ucsd/islet_organoid_differentiation/sandbox"
REMOTE=aklie@narrows-login.sdsc.edu
REMOTE_BASE=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox

# ---------- 1. Sync dependencies that only exist locally --------------------

echo "=== Syncing Sep 7 optimization results (evolution.npz + trajectory) ==="
rsync -avz --relative \
  "$LOCAL_BASE/./2026_09_07_sox32_enhancer_optimization/results/evolution.npz" \
  "$LOCAL_BASE/./2026_09_07_sox32_enhancer_optimization/tables/trajectory_metrics.tsv" \
  "$LOCAL_BASE/./2026_09_07_sox32_enhancer_optimization/config.py" \
  "$LOCAL_BASE/./2026_09_07_sox32_enhancer_optimization/scripts/common.py" \
  "$LOCAL_BASE/./2026_09_07_sox32_enhancer_optimization/scripts/setup_carter_env.sh" \
  "$REMOTE:$REMOTE_BASE/"

echo ""
echo "=== Syncing Aug 17 motif catalog ==="
rsync -avz --relative \
  "$LOCAL_BASE/./2026_08_17_tf_hits/motif_catalog.tsv" \
  "$REMOTE:$REMOTE_BASE/"

echo ""
echo "=== Syncing Sep 20 project ==="
rsync -avz \
  --exclude='__pycache__' --exclude='.DS_Store' --exclude='.ipynb_checkpoints' \
  "$LOCAL_BASE/2026_09_20_sox32_enhancer_optimization/" \
  "$REMOTE:$REMOTE_BASE/2026_09_20_sox32_enhancer_optimization/"

# ---------- 2. Create output dirs + venv on narrows -------------------------

echo ""
echo "=== Setting up environment on narrows ==="
ssh "$REMOTE" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

BASE=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox
SEP20="$BASE/2026_09_20_sox32_enhancer_optimization"
SEP07="$BASE/2026_09_07_sox32_enhancer_optimization"

# Create output directories
mkdir -p "$SEP20"/{results,beds,tables,logs}

# Set up venv — reuse Sep 7's if it exists, otherwise build a fresh one
if [ -d "$SEP07/.venv" ]; then
    echo "Symlinking .venv → Sep 7 venv"
    ln -sfn "$SEP07/.venv" "$SEP20/.venv"
elif [ -x /carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python ]; then
    echo "Building fresh .venv from eugene_tools"
    bash "$SEP07/scripts/setup_carter_env.sh" 2>&1 | tail -5
    ln -sfn "$SEP07/.venv" "$SEP20/.venv"
else
    echo "WARNING: no existing venv found and eugene_tools not available"
    echo "  Run: bash $SEP07/scripts/setup_carter_env.sh"
    echo "  Then: ln -s $SEP07/.venv $SEP20/.venv"
fi

# Verify
echo ""
echo "Checking venv..."
"$SEP20/.venv/bin/python" -c "
import crested, keras, torch
from tangermeme.seqlet import recursive_seqlets
from tangermeme.annotate import annotate_seqlets
print(f'crested={crested.__version__}  keras={keras.__version__}  torch={torch.__version__}')
print('tangermeme OK')
"

echo ""
echo "Ready. Submit with:"
echo "  cd $SEP20 && sbatch scripts/run_attributions.sh"
REMOTE_SCRIPT
