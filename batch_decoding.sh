#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --job-name=eeg_decoding
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=6
set -e
ml anaconda
conda activate eeg
project_folder=/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis
eeg_preprocess=$1
eeg_files=($eeg_preprocess/*.csv.gz)
file=${eeg_files[${SLURM_ARRAY_TASK_ID}]}
python $project_folder/scripts/eeg_decoding.py -f /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI -e $file -o /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegDecoding -x eeg -y behavior
python $project_folder/scripts/eeg_decoding.py -f /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI -e $file -o /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegDecoding -x eeg -y fmri
python $project_folder/scripts/eeg_decoding.py -f /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI -e $file -o /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegDecoding -x eeg_behavior -y fmri
