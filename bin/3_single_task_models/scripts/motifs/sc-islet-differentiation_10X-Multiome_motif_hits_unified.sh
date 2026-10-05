#!/bin/bash

#####
# Motif hit calling with UNIFIED clustered motif catalog
# Runs finemo against the 70 gimme-clustered motifs (synthetic modisco h5)
# so hits are directly comparable across cell types
#
# Submit with gpu_array.sh: -n 22 -x 2 -t 14-00:00:00
#####

date
echo -e "Job ID: $SLURM_JOB_ID\n"

# Configure environment
source activate finemo_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/opt/miniconda3/envs/finemo_gpu/lib/

celltypes=(
	DE
	ENP_phase1
	FB_FLT1
	PFG1
	PFG2
	PGT1
	PGT2
	PGT3
	PP1
	PP2
	SC_delta_GHRL
	early_ENP
	early_SC_EC
	early_SC_alpha
	early_SC_beta
	exocrine
	late_ENP
	late_SC_EC
	late_SC_alpha
	late_SC_beta
	liver
	proliferating_endocrine
)
peaks=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/DE.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/ENP_phase1.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/FB_FLT1.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PFG1.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PFG2.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PGT1.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PGT2.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PGT3.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PP1.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/PP2.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/SC_delta_GHRL.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/early_ENP.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/early_SC_EC.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/early_SC_alpha.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/early_SC_beta.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/exocrine.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/late_ENP.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/late_SC_EC.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/late_SC_alpha.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/late_SC_beta.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/liver.narrowPeak
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/2_process_data/peak_calls/all/proliferating_endocrine.narrowPeak
)
input_h5s=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/DE/average/contributions/DE_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/ENP_phase1/average/contributions/ENP_phase1_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/FB_FLT1/average/contributions/FB_FLT1_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PFG1/average/contributions/PFG1_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PFG2/average/contributions/PFG2_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PGT1/average/contributions/PGT1_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PGT2/average/contributions/PGT2_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PGT3/average/contributions/PGT3_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PP1/average/contributions/PP1_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/PP2/average/contributions/PP2_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/SC_delta_GHRL/average/contributions/SC_delta_GHRL_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/early_ENP/average/contributions/early_ENP_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/early_SC_EC/average/contributions/early_SC_EC_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/early_SC_alpha/average/contributions/early_SC_alpha_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/early_SC_beta/average/contributions/early_SC_beta_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/exocrine/average/contributions/exocrine_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/late_ENP/average/contributions/late_ENP_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/late_SC_EC/average/contributions/late_SC_EC_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/late_SC_alpha/average/contributions/late_SC_alpha_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/late_SC_beta/average/contributions/late_SC_beta_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/liver/average/contributions/liver_counts.h5
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/proliferating_endocrine/average/contributions/proliferating_endocrine_counts.h5
)

# Unified motif catalog
modisco_h5=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/motifs/gimme_cluster/cluster/clustered_motifs.modisco.h5
name_map=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/motifs/gimme_cluster/cluster/clustered_motifs_name_map.tsv
window=400

# SLURM task-specific values
celltype=${celltypes[$SLURM_ARRAY_TASK_ID - 1]}
peak=${peaks[$SLURM_ARRAY_TASK_ID - 1]}
input_h5=${input_h5s[$SLURM_ARRAY_TASK_ID - 1]}
output_dir=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/${celltype}/average/motifs/hits/counts_unified

mkdir -p $output_dir

echo "Cell type: $celltype"
echo "Output: $output_dir"

# Step 1: Extract regions (reuse existing npz if available)
npz_path=$output_dir/${celltype}.counts.finemo.npz
if [ ! -f "$npz_path" ]; then
    cmd="finemo extract-regions-chrombpnet-h5 \
    -c $input_h5 \
    -o $npz_path \
    -w $window"
    echo -e "Running extract:\n$cmd\n"
    eval $cmd
else
    echo "Reusing existing npz: $npz_path"
fi

# Step 2: Call hits against unified catalog
cmd="finemo call-hits \
-r $npz_path \
-m $modisco_h5 \
-o $output_dir \
-p $peak \
-N $name_map \
-l 0.8"
echo -e "Running call-hits:\n$cmd\n"
eval $cmd

# Step 3: Report
cmd="finemo report \
-r $npz_path \
-p $peak \
-H $output_dir/hits.tsv \
-m $modisco_h5 \
-o $output_dir/report \
-W $window \
-N $name_map"
echo -e "Running report:\n$cmd\n"
eval $cmd

date
