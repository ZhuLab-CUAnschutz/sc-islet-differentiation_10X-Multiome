# Paths for the bin/7_sequence_design SLURM runners. Source this; edit the block below for a new cluster.

# ---- edit me -----------------------------------------------------------------
DATASET=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome   # repo / dataset root
BASE_PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python           # base env the venv is built from
# ------------------------------------------------------------------------------

BIN=$DATASET/bin/7_sequence_design
# The finetuned CREsted multitask checkpoint. NOTE: 23.keras (finetuned) is the design model;
# the synergy multitask cross-check uses a DIFFERENT checkpoint (finetuned_v2/28.keras).
export SOX32_MODEL=${SOX32_MODEL:-$DATASET/results/4_multi_task_model/crested/finetuned/checkpoints/23.keras}
export DESIGN_DIR=${DESIGN_DIR:-$DATASET/results/7_sequence_design}
export KERAS_BACKEND=torch
PY=$BIN/.venv/bin/python     # built by envs/setup_crested_env.sh
