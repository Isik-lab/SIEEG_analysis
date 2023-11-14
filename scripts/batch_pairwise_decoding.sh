#!/bin/bash -l

#SBATCH
#SBATCH --job-name=pairwise-decoding
#SBATCH --time=5:15:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
n_groups=$2
gaze=$3

ml anaconda
conda activate nibabel

python pairwise_decoding.py \
 --sid "$subj" --n_groups "$n_groups" "$gaze" \
 --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data \
 --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures  

