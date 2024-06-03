user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
fmri_scores=$(project_folder)/data/interim/fMRIEncoding/scores.csv.gz

all: create_dirs $(project_folder)/data/interim/fMRIEncoding/scores.csv.gz

create_dirs:
	mkdir -p $(project_folder)/data/interim/fMRIEncoding

$(fmri_scores): create_dirs
	echo "project_folder = $(project_folder)"
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=reorganize_fmri\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
export NEPTUNE_CUSTOM_RUN_ID=fmri_encoding\n\
python $(project_folder)/scripts/fmri_encoding.py \
-a $(project_folder)/data/interim/ReorganizefMRI \
-f $(project_folder)/data/interim/ReorganizefMRI \
-o $(project_folder)/data/interim/fMRIEncoding/scores.csv.gz" | sbatch