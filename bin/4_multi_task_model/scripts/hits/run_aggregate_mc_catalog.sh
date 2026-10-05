#!/bin/bash
date
echo "Job ID: $SLURM_JOB_ID"
PY=/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/bin/python
BASE=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome
SCRIPT=${BASE}/bin/4_multi_task_model/scripts/hits/aggregate_hits.py
HITS=${BASE}/results/4_multi_task_model/crested/motif_compendium_cluster/hits
META=${BASE}/config/cell_type_metadata.tsv
${PY} ${SCRIPT} --hits_root ${HITS} --output_dir ${HITS} --metadata ${META}
date
