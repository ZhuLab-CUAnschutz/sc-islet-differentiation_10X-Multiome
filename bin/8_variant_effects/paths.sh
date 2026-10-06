# Cluster paths for the bin/8_variant_effects shell stages (scoring, SHAP, hit calling).
# The Python stages read config/config.yaml instead; keep the two in sync.

# ---- edit me -----------------------------------------------------------------
DATASET=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome   # repo / dataset root
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
SIZES=/carter/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes
SCORER_DIR=/carter/users/aklie/opt/variant-scorer/src    # github.com/kundajelab/variant-scorer
CHROMBPNET_LIB=/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib             # libcudnn; empty if unneeded
CHROMBPNET_ENV=chrombpnet                                # conda env with tensorflow + chrombpnet
HITCALL_ENV=eugene_tools                                 # conda env used to run the hit caller
FINEMO_BIN=/carter/users/aklie/opt/miniconda3/envs/finemo_gpu/bin/finemo   # finemo matching its polars
# ------------------------------------------------------------------------------

MODEL_ROOT=$DATASET/results/3_single_task_models/models
CATALOG=$DATASET/results/3_single_task_models/motifs/v1.1
VE=${VARIANT_DIR:-$DATASET/results/8_variant_effects}     # work dir (matches config.yaml "sandbox")
BIN=$DATASET/bin/8_variant_effects

[ -n "$CHROMBPNET_LIB" ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CHROMBPNET_LIB
