#!/bin/bash -l

#SBATCH
#SBATCH --time=30:00
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-sub-%x-%j.out

subj=${1:-1}
echo "subj: $subj"

ml anaconda
conda activate eeg

python feature_decoding.py --sid $subj \
 --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data \
 --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures
