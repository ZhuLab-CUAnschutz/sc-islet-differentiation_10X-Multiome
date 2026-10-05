#!/bin/bash
date
echo "Job ID: $SLURM_JOB_ID"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
SCRIPT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/bin/4_multi_task_model/scripts/annotation/aggregate_motif_context.py
# Annotation dir: pass as $1 to target a non-default catalog (e.g. ST gimme).
ANN=${1:-/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model/crested/motif_compendium_cluster/annotation}
echo "Annotation dir: $ANN"
${PY} -u ${SCRIPT} --ann_dir ${ANN}
date
