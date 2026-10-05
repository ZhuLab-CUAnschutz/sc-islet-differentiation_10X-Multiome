#!/bin/bash
date
echo "Job ID: $SLURM_JOB_ID"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
SCRIPT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/bin/4_multi_task_model/scripts/annotation/build_gencode_features.py
OUT=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model/crested/motif_compendium_cluster/annotation
${PY} -u ${SCRIPT} --out_dir ${OUT}
date
