#!/bin/bash -l

#SBATCH
#SBATCH --time=6:00:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-sub-%j.out

# TO RUN
# sbatch --array=1-21%5 --export=ALL batch_eeg_fmri.sh

# Call your encoding script, passing the SLURM_ARRAY_TASK_ID
source batch_fmri_decoding.sh $SLURM_ARRAY_TASK_ID
