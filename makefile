user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 01 02 03 04 05 06 08 09 10 11 12 13 14 15 16 17 18 19 20 21

# Dependencies
fmri_encoding=$(project_folder)/data/interim/fMRIEncoding
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing

# Find all dependent files
dependent_files=$(wildcard $(project_folder)/data/interim/eegPreprocessing/*/*.csv.gz)

# Extract targets from dependent files
decoding_targets=$(dependent_files:$(project_folder)/data/interim/eegPreprocessing/%.csv.gz=$(project_folder)/data/interim/eegDecoding/%)

# Steps to run
all: $(fmri_encoding) $(eeg_subs:%=$(eeg_preprocess)/sub-%) $(decoding_targets)

# Perform fMRI encoding with features
fmri_dir=$(project_folder)/data/interim/ReorganizefMRI
$(fmri_encoding): $(dependent_dir)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=fmri_encoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/fmri_encoding.py \
-a $(fmri_dir) -f $(dependent_dir) -o $(target)" | sbatch

# Prepare EEG data for regression
dependent_dir=$(project_folder)/data/interim/eegLab
$(eeg_preprocess)/sub-%: $(dependent_dir)
	@target_dir=$@; \
	s=$*; \
	if [ ! -d $$target_dir ] || [ $(dependent_dir) -nt $$target_dir ]; then \
		printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_preprocess\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(dependent_dir) -s $$s -o $$target_dir" | sbatch ; \
	fi

# EEG decoding analysis
$(project_folder)/data/interim/eegDecoding/%: $(project_folder)/data/interim/eegPreprocessing/%/%.csv.gz $(fmri_dir)
	@target_dir=$(project_folder)/data/interim/eegDecoding/$*; \
	dependent_file=$<; \
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_decoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/eeg_decoding.py \
-a $(fmri_dir) -e $$dependent_file -o $$target_dir" | sbatch
