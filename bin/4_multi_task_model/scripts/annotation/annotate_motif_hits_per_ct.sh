#!/bin/bash
#####
# Per-CT motif-hit genomic-context annotation (Phase 7.2 of MC cluster pipeline).
# Submit as a 22-CT array (concurrency 8):
#   sbatch --partition=carter-compute --account=carter-compute \
#     --cpus-per-task=4 --mem=8G --time=01:00:00 \
#     --array=1-22%8 --job-name=pr_annot_cluster \
#     --output=$BIN/slurm_logs/4_multi_task_model/%x.%A_%a.out \
#     <this>.sh
#####
date
echo "Job ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"

celltypes=(
    DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2
    SC_delta_GHRL early_ENP early_SC_EC early_SC_alpha early_SC_beta
    exocrine late_ENP late_SC_EC late_SC_alpha late_SC_beta liver
    proliferating_endocrine
)
celltype=${celltypes[$SLURM_ARRAY_TASK_ID - 1]}
echo "Cell type: $celltype"

PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
SCRIPT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/bin/4_multi_task_model/scripts/annotation/annotate_motif_hits_per_ct.py
BASE=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome
# Catalog dir: pass as $1 to target a non-default catalog (e.g. ST gimme).
CAT=${1:-${BASE}/results/4_multi_task_model/crested/motif_compendium_cluster}
echo "Catalog: $CAT"

HITS_TSV=${CAT}/hits/${celltype}/hits.tsv
PEAK=${BASE}/results/2_process_data/peak_calls/all/${celltype}.narrowPeak
TSS=${CAT}/annotation/gencode_v43_tss.bed
FEAT=${CAT}/annotation/gencode_v43_features.bed
OUT=${CAT}/annotation/${celltype}

${PY} -u ${SCRIPT} \
    --hits_tsv ${HITS_TSV} \
    --peakfile ${PEAK} \
    --tss_bed ${TSS} \
    --features_bed ${FEAT} \
    --out_dir ${OUT}

date
