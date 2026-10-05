#! /bin/bash

#####
# Script to make average prediction bigwigs over models
# USAGE: sbatch \
#--job-name=average_predictions \
#--partition carter-compute
#--output slurm_logs/%x.%A.%a.out \
#--mem=16G \
#-n 4 \
#-t 02-00:00:00 \
#average_predictions.sh
#####

date
echo -e "Job ID: $SLURM_JOB_ID\n"

# Set-up env
source activate eugene_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cellar/users/aklie/opt/miniconda3/envs/eugene_tools/lib

# file lists
celltype=PGT1
bias_preds=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/PGT1/fold_0/chrombpnet/0.5/predictions/PGT1_bias.bw
)
chrombpnet_nobias_preds=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/PGT1/fold_0/chrombpnet/0.5/predictions/PGT1_chrombpnet_nobias.bw
)
chrombpnet_preds=(
	/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/PGT1/fold_0/chrombpnet/0.5/predictions/PGT1_chrombpnet.bw
)
output_dir=/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/3_single_task_models/models/PGT1/average/predictions

# echo the celltype and peak
echo -e "Celltype: $celltype"
echo -e "Bias preds: ${bias_preds[@]}"
echo -e "Chrombpnet nobias preds: ${chrombpnet_nobias_preds[@]}"
echo -e "Chrombpnet preds: ${chrombpnet_preds[@]}"
echo -e "Output directory: $output_dir\n"

# wigToBigWig
wigToBigWig=/cellar/users/aklie/opt/wigToBigWig
chromsizes=/cellar/users/aklie/data/ref/genomes/hg38/hg38.chrom.sizes

# make the output directory
mkdir -p $output_dir

# Run cmd for bias. If there are multiple bias predictions, take the mean. Else, just copy the file
if [ ${#bias_preds[@]} -eq 1 ]; then
    cmd="cp ${bias_preds[0]} $output_dir/${celltype}_bias.bw"
else
    cmd="wiggletools mean ${bias_preds[@]} > ${celltype}_temp_bias.wig && $wigToBigWig ${celltype}_temp_bias.wig $chromsizes $output_dir/${celltype}_bias.bw && rm -f ${celltype}_temp_bias.wig"
fi
echo -e "Running command:\n$cmd\n"
eval $cmd

# Run cmd for chrombpnet_nobias
if [ ${#chrombpnet_nobias_preds[@]} -eq 1 ]; then
    cmd="cp ${chrombpnet_nobias_preds[0]} $output_dir/${celltype}_chrombpnet_nobias.bw"
else
    cmd="wiggletools mean ${chrombpnet_nobias_preds[@]} > ${celltype}_temp_chrombpnet_nobias.wig && $wigToBigWig ${celltype}_temp_chrombpnet_nobias.wig $chromsizes $output_dir/${celltype}_chrombpnet_nobias.bw && rm -f ${celltype}_temp_chrombpnet_nobias.wig"
fi
echo -e "Running command:\n$cmd\n"
eval $cmd

# Run cmd for chrombpnet
if [ ${#chrombpnet_preds[@]} -eq 1 ]; then
    cmd="cp ${chrombpnet_preds[0]} $output_dir/${celltype}_chrombpnet.bw"
else
    cmd="wiggletools mean ${chrombpnet_preds[@]} > ${celltype}_temp_chrombpnet.wig && $wigToBigWig ${celltype}_temp_chrombpnet.wig $chromsizes $output_dir/${celltype}_chrombpnet.bw && rm -f ${celltype}_temp_chrombpnet.wig"
fi
echo -e "Running command:\n$cmd\n"
eval $cmd

# Date
date
