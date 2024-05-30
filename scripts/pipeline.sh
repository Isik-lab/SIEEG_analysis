#!/bin/bash -l

#SBATCH
#SBATCH --time=2:00:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-sub-%j.out

ml anaconda
conda activate eeg

user=$(whoami)
echo "user = $user"
project_folder="/home/$user/scratch4-lisik3/$user/SIEEG_analysis"

# fMRI encoding
