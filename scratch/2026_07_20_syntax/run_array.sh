#!/bin/bash
# Array wrapper: SLURM_ARRAY_TASK_ID -> cell type; run a pair manifest through that CT's model.
# Env: PAIRS, OUTSUB (subdir under results/), NSEQS, MAXDIST, BATCH
set -e
date; echo "Job $SLURM_JOB_ID Task $SLURM_ARRAY_TASK_ID"
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
WD=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
CWMS=$MROOT/motifs/v1.1/cwm/cwms.npz
cts=(DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2 SC_delta_GHRL early_ENP early_SC_EC early_SC_alpha early_SC_beta exocrine late_ENP late_SC_EC late_SC_alpha late_SC_beta liver proliferating_endocrine)
CT=${cts[$SLURM_ARRAY_TASK_ID-1]}
MODEL=$MROOT/models/$CT/fold_0/chrombpnet/0.5/models/chrombpnet_nobias.h5
BG=$MROOT/models/$CT/fold_0/chrombpnet/0.5/auxiliary/filtered.nonpeaks.bed
: ${NSEQS:=100}; : ${MAXDIST:=200}; : ${BATCH:=256}; : ${OUTSUB:=candidate}
OUT=$WD/results/$OUTSUB
mkdir -p "$OUT"
echo "CT=$CT MODEL=$MODEL"; echo "PAIRS=$PAIRS OUT=$OUT NSEQS=$NSEQS MAXDIST=$MAXDIST"
SECONDS=0
$PY $WD/scripts/marginalize_pairs.py \
  -m "$MODEL" --cwms "$CWMS" -g "$GENOME" --background "$BG" \
  --pairs "$PAIRS" -o "$OUT" --ct "$CT" -n "$NSEQS" --maxdist "$MAXDIST" --batch_size "$BATCH"
echo "DONE $CT in ${SECONDS}s"; date
