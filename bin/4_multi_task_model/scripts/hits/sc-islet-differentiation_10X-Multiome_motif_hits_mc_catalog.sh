#!/bin/bash
#####
# Motif hit calling against the MC-built catalog (sign-aware, motif_compendium_cluster/).
# Mirrors motif_hits_all_peaks.sh but reads contributions from
# contributions_all_peaks/ and writes hits to motif_compendium_cluster/hits/.
#####
date
echo -e "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID\n"

source activate finemo_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/opt/miniconda3/envs/finemo_gpu/lib/

celltypes=(
    DE ENP_phase1 FB_FLT1 PFG1 PFG2 PGT1 PGT2 PGT3 PP1 PP2
    SC_delta_GHRL early_ENP early_SC_EC early_SC_alpha early_SC_beta
    exocrine late_ENP late_SC_EC late_SC_alpha late_SC_beta liver
    proliferating_endocrine
)
celltype=${celltypes[$SLURM_ARRAY_TASK_ID - 1]}
echo "Cell type: $celltype"

BASE=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome
PEAK=${BASE}/results/2_process_data/peak_calls/all/${celltype}.narrowPeak
OH=${BASE}/results/4_multi_task_model/crested/contributions_all_peaks/${celltype}_oh.npz
CONTRIB=${BASE}/results/4_multi_task_model/crested/contributions_all_peaks/${celltype}_contrib.npz

# MC-built catalog (replaces motifs_all_peaks/ for this hit-calling pass)
modisco_h5=${BASE}/results/4_multi_task_model/crested/motif_compendium_cluster/clustered_motifs.modisco.h5
name_map=${BASE}/results/4_multi_task_model/crested/motif_compendium_cluster/clustered_motifs_name_map_unique.tsv
window=400

output_dir=${BASE}/results/4_multi_task_model/crested/motif_compendium_cluster/hits/${celltype}
mkdir -p $output_dir
echo "Output: $output_dir"

# Step 1: extract regions from oh/contrib npz pair
npz_path=$output_dir/${celltype}.finemo.npz
if [ ! -f "$npz_path" ]; then
    cmd="finemo extract-regions-modisco-fmt -s $OH -a $CONTRIB -o $npz_path -w $window"
    echo -e "Running extract:\n$cmd\n"
    eval $cmd
else
    echo "Reusing existing npz: $npz_path"
fi

# Step 2: call hits against the MC catalog
cmd="finemo call-hits -r $npz_path -m $modisco_h5 -o $output_dir -p $PEAK -N $name_map -l 0.8"
echo -e "Running call-hits:\n$cmd\n"
eval $cmd

# Step 3: report (best-effort)
cmd="finemo report -r $npz_path -p $PEAK -H $output_dir/hits.tsv -m $modisco_h5 -o $output_dir/report -W $window -N $name_map"
echo -e "Running report (best-effort):\n$cmd\n"
eval $cmd || echo "WARN: finemo report failed (expected for synthetic modisco h5). Continuing."

date
