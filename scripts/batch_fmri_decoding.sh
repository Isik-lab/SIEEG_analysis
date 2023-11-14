#!/bin/bash -l

#SBATCH
#SBATCH --job-name=glmsingle
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH -A lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

subj=$1

ml anaconda

conda activate nibabel

python fmri_decoding.py --sid $subj \
 --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data \
 --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures