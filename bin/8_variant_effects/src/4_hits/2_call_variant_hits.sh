#!/bin/bash
#SBATCH --job-name=vep_hits
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=a30:1
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH --output=logs/4_hits/%x.%A.%a.out
#####
# Stage 4.2: call motif hits overlapping each significant variant, per cell type,
# against the v1.1 catalog modisco (finemo, env eugene_tools). Run from sandbox after 4.1:
#   sbatch --array=1-22%8 --export=ALL,LIST=inputs/significant_variants.bed,HITS_PER_LOC=3 src/4_hits/2_call_variant_hits.sh
#####
date; echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
# Run the caller under eugene_tools (has pandas); use finemo from finemo_gpu (its
# finemo build matches its polars, unlike eugene_tools' older editable checkout).
source "${SLURM_SUBMIT_DIR:-.}/paths.sh"
source activate "$HITCALL_ENV"
FDIR=$(mktemp -d)
ln -sf "$FINEMO_BIN" "$FDIR/finemo"
export PATH="$FDIR:$PATH"

HITCALLER=$SCORER_DIR/hitcaller_variant.py
MODISCO=$CATALOG/cluster/clustered_motifs.modisco.h5
# LC_ALL=C reproduces the original array order, so an --array index always maps to the
# same cell group. A locale-aware sort reorders them.
mapfile -t CELLGROUPS < <(awk -F'\t' 'NR>1{print $1}' "$DATASET/config/cell_type_metadata.tsv" | LC_ALL=C sort)

group=${CELLGROUPS[$SLURM_ARRAY_TASK_ID - 1]}
outdir=results/hits/${group}
shap_h5=${outdir}/${group}.variant_shap.counts.h5
[ -f "$shap_h5" ] || { echo "MISSING SHAP: $shap_h5 (run Stage 4.1 first)"; exit 1; }

cmd="python $HITCALLER --shap_data $shap_h5 --input_type h5 --modisco_h5 $MODISCO \
--variant_file $LIST --output_dir $outdir --hits_per_loc ${HITS_PER_LOC:-3}"
echo -e "Running:\n$cmd\n"
eval $cmd
rc=$?; echo "exit=$rc"; date; exit $rc
