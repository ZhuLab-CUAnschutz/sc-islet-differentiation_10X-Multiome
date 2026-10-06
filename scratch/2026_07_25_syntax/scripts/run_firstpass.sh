#!/bin/bash
# First-pass 1-bp discovery run: 2701 pairs (2628 hetero + 73 homodimers) x 22 CT.
# n=32 backgrounds, gaps 0-50 (1-bp), fixed orientation, cwm-consensus insert (== report logo trim).
# Pair-major chunk files (whole pairs each) so completed chunks = fully-profiled pairs.
# Chunk count = 91 (30 pairs/chunk).  Generous 12h wall (chunks run ~1.8h) so nothing times out.
#SBATCH --job-name=fp_1bp
#SBATCH --partition=carter-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-91%6
#SBATCH --output=slurm_logs/fp_1bp.%A_%a.out
set -e; date; echo "chunk $SLURM_ARRAY_TASK_ID"
PY=/carter/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/carter/users/aklie/opt/miniconda3/envs/chrombpnet/lib
WD=/carter/users/aklie/projects/islet_organoid_differentiation/sandbox/2026_07_19_syntax_pairwise_marg
MROOT=/carter/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models
CH=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT=$WD/results/full_fp/chunk_$SLURM_ARRAY_TASK_ID; mkdir -p "$OUT"
cd "$WD/scripts"
$PY "$WD/scripts/marginalize_pairs.py" \
  --models_root "$MROOT" --cwms "$MROOT/motifs/v1.1/cwm/cwms.npz" \
  -g /carter/users/aklie/data/ref/genomes/hg38/hg38.fa \
  --pairs "$WD/inputs/fp_chunks/chunk_$CH.tsv" -o "$OUT" \
  -n 32 --maxdist 50 --gap_step 1 --batch_size 512
echo "DONE chunk $CH"; date
