#!/bin/bash
eeg_preprocess=/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing
num_files=$(echo $eeg_preprocess/*.csv.gz | wc -w)
echo $num_files
sbatch --array=0-$((num_files-1))%50 batch_decoding.sh $eeg_preprocess
