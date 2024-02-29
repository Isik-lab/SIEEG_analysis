#!/bin/bash -l

subj=${1:-1}
echo "subj: $subj"

ml anaconda
conda activate eeg

python fmri_decoding.py --sid $subj \
 --data_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data \
 --figure_dir /home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures \
 --save_whole_brain --overwrite
