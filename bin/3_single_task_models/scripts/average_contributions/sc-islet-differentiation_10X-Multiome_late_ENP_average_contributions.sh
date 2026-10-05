#! /bin/bash

#####
# Script to compute averaged contributions and profiles across folds
# USAGE: sbatch \
#--job-name=avg_contributions \
#--account=carter-compute \
#--output=slurm_logs/%x.%A.out \
#--mem=128G \
#-n 4 \
#-t 02-00:00:00 \
#average_contributions.sh
#####

date
echo -e "Job ID: $SLURM_JOB_ID\\n"

# Set up environment
source activate eugene_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/lib

# Define input files
celltype=late_ENP
counts=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/late_ENP/fold_0/chrombpnet/0.5/contributions/late_ENP.counts_scores.bw
)
profile=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/late_ENP/fold_0/chrombpnet/0.5/contributions/late_ENP.profile_scores.bw
)
counts_h5=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/late_ENP/fold_0/chrombpnet/0.5/contributions/late_ENP.counts_scores.h5
)
profile_h5=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/late_ENP/fold_0/chrombpnet/0.5/contributions/late_ENP.profile_scores.h5
)
output_dir=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/late_ENP/average/contributions
window=400

# Create output directory
mkdir -p $output_dir

# Define paths
wigToBigWig=/cellar/users/aklie/opt/wigToBigWig
chromsizes=/cellar/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes
path_script=/cellar/users/aklie/projects/ML4GLand/chrombpnet/scripts/average_importance_h5_over_folds.py

# Compute mean bigwig for counts. If there are multiple bias predictions, take the mean. Else, just copy the file
if [ ${#counts[@]} -eq 1 ]; then
    cmd="cp ${counts[0]} $output_dir/${celltype}_counts.bw"
else
    cmd="wiggletools mean ${counts[@]} > ${celltype}_temp_counts.wig && $wigToBigWig ${celltype}_temp_counts.wig $chromsizes $output_dir/${celltype}_counts.bw && rm -f ${celltype}_temp_counts.wig"
fi
echo -e "Running command:\\n$cmd\\n"
eval $cmd

# Compute mean bigwig for profile
if [ ${#profile[@]} -eq 1 ]; then
    cmd="cp ${profile[0]} $output_dir/${celltype}_profile.bw"
else
    cmd="wiggletools mean ${profile[@]} > ${celltype}_temp_profile.wig && $wigToBigWig ${celltype}_temp_profile.wig $chromsizes $output_dir/${celltype}_profile.bw && rm -f ${celltype}_temp_profile.wig"
fi
echo -e "Running command:\\n$cmd\\n"
eval $cmd

# Compute mean contributions in HDF5 format, if there are multiple files, run the script, else just copy the file
if [ ${#counts_h5[@]} -eq 1 ]; then
    cmd="cp ${counts_h5[0]} $output_dir/${celltype}_counts.h5"
else
    cmd="python $path_script --files ${counts_h5[@]} --output_file $output_dir/${celltype}_counts.h5 --window $window"
fi
echo -e "Running command:\\n$cmd\\n"
eval $cmd

# Compute mean profile in HDF5 format
if [ ${#profile_h5[@]} -eq 1 ]; then
    cmd="cp ${profile_h5[0]} $output_dir/${celltype}_profile.h5"
else
    cmd="python $path_script --files ${profile_h5[@]} --output_file $output_dir/${celltype}_profile.h5 --window $window"
fi
echo -e "Running command:\\n$cmd\\n"
eval $cmd

# Completion message
date
