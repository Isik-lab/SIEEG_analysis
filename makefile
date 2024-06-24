user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
neptune_api_token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 01 02 03 04 05 06 08 09 10 11 12 13 14 15 16 17 18 19 20 21

# Dependencies
fmri_data=$(project_folder)/data/interim/ReorganizefMRI
fmri_encoding=$(project_folder)/data/interim/fmriEncoding
plot_encoding=$(project_folder)/data/interim/PlotEncoding

matlab_eeg_path=$(project_folder)/data/interim/eegLab
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing
eeg_decoding=$(project_folder)/data/interim/eegDecoding
plot_decoding=$(project_folder)/data/interim/PlotDecoding
plot_shared_variance=$(project_folder)/data/interim/PlotSharedVariance


# Steps to run
all: fmri_encoding eeg_preprocess eeg_decode plot_decoding plot_shared_variance

# Perform fMRI encoding with features
fmri_encoding: $(fmri_encoding)/.encoding_done $(fmri_data)
$(fmri_encoding)/.encoding_done: 
	mkdir -p $(fmri_encoding)
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
-f $(fmri_data) -o $(fmri_encoding) \
-x '[\"alexnet\", \"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/fmri_encoding.py \
-f $(fmri_data) -o $(fmri_encoding) \
-x '[\"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/fmri_encoding.py \
-f $(fmri_data) -o $(fmri_encoding) \
-x '[\"alexnet\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/fmri_encoding.py \
-f $(fmri_data) -o $(fmri_encoding) \
-x '[\"alexnet\", \"moten\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_encoding.py \
-f $(fmri_data) -e $(fmri_encoding) -o $(plot_encoding) \
-x '[\"alexnet\", \"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_encoding.py \
-f $(fmri_data) -e $(fmri_encoding) -o $(plot_encoding) \
-x '[\"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_encoding.py \
-f $(fmri_data) -e $(fmri_encoding) -o $(plot_encoding) \
-x '[\"alexnet\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_encoding.py \
-f $(fmri_data) -e $(fmri_encoding) -o $(plot_encoding) \
-x '[\"alexnet\", \"moten\"]' -y '[\"fmri\"]'" | sbatch
	touch $(fmri_encoding)/.encoding_done

# Preprocess EEG data for regression
eeg_preprocess: $(eeg_preprocess)/.preprocess_done $(matlab_eeg_path) $(fmri_data)
$(eeg_preprocess)/.preprocess_done: 
	mkdir -p $(eeg_preprocess)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_preprocess\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(matlab_eeg_path) -s $$s -o $(eeg_preprocess) \
-x '[\"alexnet\", \"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(matlab_eeg_path) -s $$s -o $(eeg_preprocess) \
-x '[\"moten\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(matlab_eeg_path) -s $$s -o $(eeg_preprocess) \
-x '[\"alexnet\", \"behavior\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/eeg_preprocessing.py \
-e $(matlab_eeg_path) -s $$s -o $(eeg_preprocess) \
-x '[\"alexnet\", \"moten\"]' -y '[\"fmri\"]'" | sbatch; \
	done
	touch $(eeg_preprocess)/.preprocess_done

# Decode EEG data
submit_file=submit_decoding_jobs.sh
batch_file=batch_decoding.sh
eeg_decode: $(eeg_decoding)/.decode_done $(eeg_preprocess)/.preprocess_done $(matlab_eeg_path) $(fmri_data)
$(eeg_decoding)/.decode_done: 
	mkdir -p $(eeg_decoding)

	@echo "#!/bin/bash" > $(submit_file)
	@echo "eeg_preprocess=$(eeg_preprocess)" >> $(submit_file)
	@echo "num_files=\$$(echo \$$eeg_preprocess/*.csv.gz | wc -w)" >> $(submit_file)
	@echo "echo \$$num_files" >> $(submit_file)
	@echo "sbatch --array=0-\$$((num_files-1))%55 $(batch_file) \$$eeg_preprocess" >> $(submit_file)
	@chmod +x $(submit_file)

	@echo "#!/bin/bash" > $(batch_file)
	@echo "#SBATCH --partition=shared" >> $(batch_file)
	@echo "#SBATCH --account=lisik33" >> $(batch_file)
	@echo "#SBATCH --job-name=eeg_decoding" >> $(batch_file)
	@echo "#SBATCH --ntasks=1" >> $(batch_file)
	@echo "#SBATCH --time=30:00" >> $(batch_file)
	@echo "#SBATCH --cpus-per-task=6" >> $(batch_file)
	@echo "set -e" >> $(batch_file)
	@echo "ml anaconda" >> $(batch_file)
	@echo "conda activate eeg" >> $(batch_file)
	@echo "project_folder=$(project_folder)" >> $(batch_file)
	@echo "eeg_preprocess=\$$1" >> $(batch_file)
	@echo "eeg_files=(\$$eeg_preprocess/*.csv.gz)" >> $(batch_file)
	@echo "file=\$${eeg_files[\$${SLURM_ARRAY_TASK_ID}]}" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\", \"alexnet\", \"moten\", \"behavior\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\", \"moten\", \"behavior\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\", \"alexnet\", \"behavior\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\", \"alexnet\", \"moten\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/eeg_decoding.py -f $(fmri_data) -e \$$file -o $(eeg_decoding) -x '[\"eeg\"]' -y '[\"behavior\"]'" >> $(batch_file)
	@chmod +x $(batch_file)

	@echo $(submit_file) 
	./$(submit_file)
	touch $(eeg_decoding)/.decode_done

plot_decoding: $(plot_decoding)/.done $(eeg_decoding)
$(plot_decoding)/.done: 
	mkdir -p $(plot_decoding)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=plot_decoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/plot_decoding.py -x eeg -y fmri" | sbatch
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=plot_decoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/plot_decoding.py -x eeg -y behavior" | sbatch
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=plot_decoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=30:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/plot_decoding.py -x eeg_behavior -y fmri" | sbatch

plot_shared_variance: $(plot_shared_variance)/.done $(eeg_decoding) $(plot_decoding)
$(plot_shared_variance)/.done: 
	mkdir -p $(plot_shared_variance)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=plot_shared_variance\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=5:00\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/plot_shared_variance.py\n\
python $(project_folder)/scripts/plot_shared_variance.py\n\
python $(project_folder)/scripts/plot_shared_variance.py\n\
python $(project_folder)/scripts/plot_shared_variance.py" | sbatch

clean:
	rm *.out
	rm *.sh
