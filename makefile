user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21

#Dependencies
fmri_encoding=$(project_folder)/data/interim/fMRIEncoding
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing

#Steps to run
all: $(fmri_encoding) preprocess

# Perform fMRI encoding with features
dependent_dir=$(project_folder)/data/interim/ReorganizefMRI
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
-a $(dependent_dir) -f $(dependent_dir) -o $(target)" | sbatch

# Prepare EEG data for regression
dependent_dir=$(project_folder)/data/interim/PreprocessData
preprocess:
	@for s in $(eeg_subs); do \
		target_dir=$(project_folder)/data/interim/PreprocessData/sub-$$s; \
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
	done
