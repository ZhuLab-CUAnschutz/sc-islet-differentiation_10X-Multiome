#!/bin/bash
# Full all-pairs sweep (primary-CT only), job-index chunked (fix M2: balance by index, not by CT).
# Batch mode: marginalize_pairs.py reads ct from the (ct-sorted) manifest and lazily loads each
# CT's model once per chunk. Reduced-but-UNIFORM fidelity for the screen: n=64, gap_step=2.
set -e
date; echo "Job $SLURM_JOB_ID Chunk(task) $SLURM_ARRAY_TASK_ID / $NCHUNKS"
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
WD=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
GENOME=/carter/users/aklie/data/ref/genomes/hg38/hg38.fa
CWMS=$MROOT/motifs/v1.1/cwm/cwms.npz
: ${NSEQS:=64}; : ${MAXDIST:=200}; : ${GAPSTEP:=2}; : ${BATCH:=512}; : ${NCHUNKS:=44}
OUT=$WD/results/full; mkdir -p "$OUT"
CHUNK=$((SLURM_ARRAY_TASK_ID-1))
$PY $WD/scripts/marginalize_pairs.py \
  --models_root $MROOT --cwms "$CWMS" -g "$GENOME" \
  --pairs $WD/inputs/pairs_full_sorted.tsv -o "$OUT" \
  --chunk $CHUNK --nchunks $NCHUNKS \
  -n "$NSEQS" --maxdist "$MAXDIST" --gap_step "$GAPSTEP" --batch_size "$BATCH"
echo "DONE chunk $CHUNK"; date
