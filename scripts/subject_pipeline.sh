#!/bin/bash -l

#SBATCH
#SBATCH --time=2:00:00
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-sub-%j.out

set -e

ml anaconda
conda activate eeg

user=$(whoami)
echo "user = $user"
project_folder="/home/$user/scratch4-lisik3/$user/SIEEG_analysis"

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ=="

# Reorganize fMRI
python reorganize_fmri.py -d $project_folder
