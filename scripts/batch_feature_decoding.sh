#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-sub-%x-%j.out

subj=${1:-1}
echo "subj: $subj"

ml anaconda
conda activate eeg

python feature_decoding.py --sid $subj --overwrite