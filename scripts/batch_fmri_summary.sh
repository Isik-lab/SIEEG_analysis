#!/bin/bash -l

#SBATCH
#SBATCH --time=30:00
#SBATCH --partition=parallel
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%j.out

ml anaconda
conda activate eeg

python fmri_whole_brain.py --top_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis