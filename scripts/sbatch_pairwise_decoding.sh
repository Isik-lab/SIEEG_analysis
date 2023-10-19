#!/bin/bash -l

#SBATCH
#SBATCH --job-name=pairwise-decoding
#SBATCH --time=1:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1
perm=$2

ml anaconda
conda activate nibabel

python voxel_permutation.py --sid "$subj" --perm "$perm"\
  --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim 
