user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
neptune_api_token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 01 02 03 04 05 06 08 09 10 11 12 13 14 15 16 17 18 19 20 21

# Dependencies
fmri_data=$(project_folder)/data/interim/ReorganizefMRI
fmri_behavior_encoding=$(project_folder)/data/interim/fmriBehaviorEncoding
matlab_eeg_path=$(project_folder)/data/interim/eegLab
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing
eeg_decoding=$(project_folder)/data/interim/eegDecoding
fmri_eeg_encoding=$(project_folder)/data/interim/fmriEEGEncoding
fmri_behavior_eeg_encoding=$(project_folder)/data/interim/fmriBehaviorEEGEncoding

# Steps to run
all: fmri_behavior_encoding eeg_preprocess eeg_decode clean

# Perform fMRI encoding with features
fmri_behavior_encoding: $(fmri_behavior_encoding)/.encoding_done
$(fmri_behavior_encoding)/.encoding_done: $(fmri_data)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=fmri_behavior_encoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/fmri_behavior_encoding.py \
-f $(fmri_data) -o $(target)" | sbatch
	touch $(fmri_behavior_encoding)/.encoding_done

# Preprocess EEG data for regression
eeg_preprocess: $(eeg_preprocess)/.preprocess_done
$(eeg_preprocess)/.preprocess_done: $(matlab_eeg_path) $(fmri_data)
	mkdir -p $(eeg_preprocess)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_preprocess\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(matlab_eeg_path) -s $$s -o $(eeg_preprocess)" | sbatch; \
	done
	touch $(eeg_preprocess)/.preprocess_done

# Decode EEG data
eeg_decode: $(eeg_decoding)/.decode_done 
$(eeg_decoding)/.decode_done: $(eeg_preprocess)/.preprocess_done $(matlab_eeg_path) $(fmri_data)
	mkdir -p $(eeg_decoding)

	@echo "#!/bin/bash" > submit_decoding_jobs.sh
	@echo "eeg_preprocess=$(eeg_preprocess)" >> submit_decoding_jobs.sh
	@echo "num_files=\$$(echo \$$eeg_preprocess/*.csv.gz | wc -w)" >> submit_decoding_jobs.sh
	@echo "echo \$$num_files" >> submit_decoding_jobs.sh
	@echo "sbatch --array=0-\$$((num_files-1))%50 batch_decoding.sh \$$eeg_preprocess" >> submit_decoding_jobs.sh
	@chmod +x submit_decoding_jobs.sh

	@echo "#!/bin/bash" > batch_decoding.sh
	@echo "#SBATCH --partition=shared" >> batch_decoding.sh
	@echo "#SBATCH --account=lisik33" >> batch_decoding.sh
	@echo "#SBATCH --job-name=eeg_decoding" >> batch_decoding.sh
	@echo "#SBATCH --ntasks=1" >> batch_decoding.sh
	@echo "#SBATCH --time=30:00" >> batch_decoding.sh
	@echo "#SBATCH --cpus-per-task=6" >> batch_decoding.sh
	@echo "set -e" >> batch_decoding.sh
	@echo "ml anaconda" >> batch_decoding.sh
	@echo "conda activate eeg" >> batch_decoding.sh
	@echo "project_folder=$(project_folder)" >> batch_decoding.sh
	@echo "eeg_preprocess=\$$1" >> batch_decoding.sh
	@echo "eeg_files=(\$$eeg_preprocess/*.csv.gz)" >> batch_decoding.sh
	@echo "file=\$${eeg_files[\$${SLURM_ARRAY_TASK_ID}]}" >> batch_decoding.sh
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x eeg -y fmri" >> batch_decoding.sh
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x eeg -y behavior" >> batch_decoding.sh
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x eeg_behavior -y fmri" >> batch_decoding.sh
	@chmod +x batch_decoding.sh

	./submit_decoding_jobs.sh
	touch $(eeg_decoding)/.decode_done


clean:
	rm *.sh