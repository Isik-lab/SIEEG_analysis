#!/bin/bash -l

#SBATCH
#SBATCH --job-name=pairwise-decoding
#SBATCH --time=2:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

subj=$1

ml gcc/9.3.0
ml gcc/13.1.0
ml helpers/0.1.1
ml cuda/12.1.0
ml anaconda

conda activate nibabel

python fmri_decoding.py --sid $subj