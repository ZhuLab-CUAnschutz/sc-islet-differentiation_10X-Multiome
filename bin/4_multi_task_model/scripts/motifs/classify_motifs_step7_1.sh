#!/bin/bash
#####
# Step 7.1 — motif classification per Liu 2026 Fig 3d.
# Standalone (not part of build_mc_cluster.py) so it runs independently of
# the in-flight hits/marg/aggregator chain. Loose-E TomTom against Vierstra
# captures secondary sub-motifs per catalog motif.
#####
date
echo "Job ID: $SLURM_JOB_ID"

PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
SCRIPT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/bin/4_multi_task_model/scripts/motifs/classify_motifs_step7_1.py
# Catalog dir: pass as $1 to target a non-default catalog (e.g. ST gimme).
CAT=${1:-/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model/crested/motif_compendium_cluster}
echo "Catalog: $CAT"
# Vierstra v2.0beta archetype DB (matches step 5.3's reference).
DB=/cellar/users/aklie/data/ref/motifs/jvierstra/motif-clustering-v2.0beta/motifs.meme

# step 5.3 used chrombpnet's tomtom; put it on PATH
export PATH=/cellar/users/aklie/opt/miniconda3/envs/chrombpnet/bin:$PATH
echo "DB MEME: $DB"
which tomtom

${PY} -u ${SCRIPT} \
    --catalog_dir ${CAT} \
    --vierstra_meme ${DB} \
    --evalue 100 \
    --slot_tol 2

date
