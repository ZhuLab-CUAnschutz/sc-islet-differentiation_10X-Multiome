# Paths for the bin/6_synergy SLURM runners. Source this; edit the block below for a new cluster.
# Everything else in bin/6_synergy is argument-driven, so this is the only file to change.

# ---- edit me -----------------------------------------------------------------
DATASET=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome   # repo / dataset root
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python                 # tangermeme 1.0.3, bpnetlite 0.8.1
CHROMBPNET_LIB=/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib              # libcudnn for torch; leave empty if not needed
# ------------------------------------------------------------------------------

MROOT=$DATASET/results/3_single_task_models          # expects models/<ct>/fold_0/chrombpnet/0.5/{models,auxiliary}/
CATALOG=$MROOT/motifs/v1.1                           # metadata.tsv + cwm/cwms.npz
CWMS=$CATALOG/cwm/cwms.npz
SYN=${SYNERGY_DIR:-$DATASET/results/6_synergy}        # all synergy inputs/outputs live here
BIN=$DATASET/bin/6_synergy

[ -n "$CHROMBPNET_LIB" ] && export LD_LIBRARY_PATH=$CHROMBPNET_LIB:${LD_LIBRARY_PATH:-}
