user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
neptune_api_token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21
features := expanse agent_distance facingness joint_action communication valence arousal

# Dependencies
videos=$(project_folder)/data/raw/videos_3000ms
motion_energy=$(project_folder)/data/interim/MotionEnergyActivations
alexnet=$(project_folder)/data/interim/AlexNetActivations
fmri_data=$(project_folder)/data/interim/ReorganizefMRI
fmri_encoding=$(project_folder)/data/interim/encodeDecode/fmri
plot_encoding=$(project_folder)/data/interim/PlotROI

matlab_eeg_path=$(project_folder)/data/interim/SIdyads_EEG
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing
eeg_reliability=$(project_folder)/data/interim/eegReliability
eeg_decoding=$(project_folder)/data/interim/encodeDecode/eeg
eeg_stats=$(project_folder)/data/interim/eegStats
plot_decoding=$(project_folder)/data/interim/PlotTimeCourse
plot_shared_variance=$(project_folder)/data/interim/PlotSharedVariance
back_to_back=$(project_folder)/data/interim/Back2Back

# Steps to run
all: motion_energy alexnet fmri_encoding eeg_preprocess eeg_reliability eeg_decode eeg_stats plot_decoding plot_shared_variance back_to_back

# Get the motion energy for the 3 s videos
motion_energy: $(motion_energy)/.done $(videos)
$(motion_energy)/.done: 
	mkdir -p $(motion_energy)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=moten\n\
#SBATCH --ntasks=1\n\
#SBATCH --time 5:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/motion_energy_activations.py " | sbatch
	touch $(motion_energy)/.done

# Get the activations from AlexNet for the 3 s videos
alexnet: $(alexnet)/.done $(videos)
$(alexnet)/.done: 
	mkdir -p $(alexnet)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=moten\n\
#SBATCH --ntasks=1\n\
#SBATCH --time 1:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/alexnet_activations.py " | sbatch
	touch $(alexnet)/.done

# Perform fMRI encoding with features
fmri_encoding: $(fmri_encoding)/.encoding_done $(fmri_data)
$(fmri_encoding)/.encoding_done: 
	mkdir -p $(fmri_encoding)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=fmri_encoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=1:00:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
#Encode all the features\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"moten\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/encode_decode.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\"]' -y '[\"fmri\"]'\n\
#Plot the results\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"moten\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"affective\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_roi.py \
-x '[\"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_fmri_variance.py" | sbatch
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
#SBATCH --time=2:00:00\n\
#SBATCH --cpus-per-task=18\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_preprocessing.py -s $$s" | sbatch; \
	done
	touch $(eeg_preprocess)/.preprocess_done

#Compute the channel-wise EEG reliability
eeg_reliability: $(eeg_reliability)/.done $(eeg_preprocess)
$(eeg_reliability)/.done: 
	mkdir -p $(eeg_reliability)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_reliability\n\
#SBATCH --time=2:00:00\n\
#SBATCH --cpus-per-task=12\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_reliability.py -s $$s" | sbatch; \
	done
	touch $(eeg_reliability)/.done


#Compute the channel-wise EEG reliability
back_to_back: $(back_to_back)/.done $(eeg_preprocess)
$(back_to_back)/.done: 
	mkdir -p $(back_to_back)
	for x in $(features); do \
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=back_to_back\n\
#SBATCH --time=2:00:00\n\
#SBATCH --cpus-per-task=12\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/back_to_back.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).csv.gz -x2 '[\"$${x}\"]'" | sbatch; \
	done; \
	done
	touch $(back_to_back)/.done

# Decode EEG data
submit_file=submit_decoding_jobs.sh
batch_file=batch_decoding.sh
eeg_decode: $(eeg_decoding)/.decode_done $(eeg_preprocess)/.preprocess_done $(matlab_eeg_path) $(fmri_data)
$(eeg_decoding)/.decode_done: 
	mkdir -p $(eeg_decoding)
	@echo "#!/bin/bash" > $(submit_file)
	@echo "eeg_preprocess=$(eeg_preprocess)" >> $(submit_file)
	@echo "num_files=\$$(echo \$$eeg_preprocess/*.csv | wc -w)" >> $(submit_file)
	@echo "echo \$$num_files" >> $(submit_file)
	@echo "sbatch --array=0-\$$((num_files-1))%47 $(batch_file) \$$eeg_preprocess" >> $(submit_file)
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
	@echo "eeg_files=(\$$eeg_preprocess/*.csv)" >> $(batch_file)
	@echo "file=\$${eeg_files[\$${SLURM_ARRAY_TASK_ID}]}" >> $(batch_file)
	@echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"moten\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"social\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"affective\"]' -y '[\"fmri\"]'" >> $(batch_file)
	# @echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\"]' -y '[\"fmri\"]'" >> $(batch_file)
	@echo "python \$$project_folder/scripts/encode_decode.py -e \$$file -x '[\"eeg\"]' -y '[\"scene\", \"primitive\", \"social\", \"affective\"]'" >> $(batch_file)
	@chmod +x $(batch_file)

	@echo $(submit_file) 
	./$(submit_file)
	# touch $(eeg_decoding)/.decode_done


#Compute the channel-wise EEG reliability
eeg_stats: $(eeg_stats)/.done $(eeg_decoding)
$(eeg_stats)/.done: 
	mkdir -p $(eeg_stats)
	echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_stats\n\
#SBATCH --time=15:00\n\
#SBATCH --cpus-per-task=6\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_stats.py -p '$(fmri_encoding)/*x-alexnet-moten-scene-primitive-social-affective_y-fmri_yhat.csv.gz" | sbatch

	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=eeg_stats\n\
#SBATCH --time=2:00:00\n\
#SBATCH --cpus-per-task=12\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/eeg_stats.py -p '$(eeg_decoding)/sub-$$(printf '%02d' $${s})*_x-eeg_y-fmri_yhat.csv.gz'\n\
python $(project_folder)/scripts/eeg_stats.py -p '$(eeg_decoding)/sub-$$(printf '%02d' $${s})*_x-eeg-alexnet-moten-scene-primitive-social-affective_y-fmri_yhat.csv.gz'\n\
python $(project_folder)/scripts/eeg_stats.py -p '$(eeg_decoding)/sub-$$(printf '%02d' $${s})*_x-eeg_y-scene-primitive-social-affective_yhat.csv.gz'" | sbatch; \
	done
	touch $(eeg_stats)/.done


#Plot the eeg decoding results
plot_decoding: $(plot_decoding)/.done $(eeg_decoding)
$(plot_decoding)/.done: 
	mkdir -p $(plot_decoding)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=plot_decoding\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=45:00\n\
#SBATCH --cpus-per-task=6\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(token)\n\
python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\"]' -y '[\"scene\", \"primitive\", \"social\", \"affective\"]'\n\
python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"moten\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"scene\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"moten\", \"primitive\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"social\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"affective\"]' -y '[\"fmri\"]'\n\
# python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\", \"alexnet\", \"moten\", \"scene\", \"primitive\", \"social\"]' -y '[\"fmri\"]'\n\
python $(project_folder)/scripts/plot_timecourse.py -x '[\"eeg\"]' -y '[\"fmri\"]'"| sbatch

#Plot the Unique Shared variance between features, EEG, and fMRI
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
# python $(project_folder)/scripts/plot_shared_variance.py -u alexnet\n\
# python $(project_folder)/scripts/plot_shared_variance.py -u moten\n\
# python $(project_folder)/scripts/plot_shared_variance.py -u scene\n\
# python $(project_folder)/scripts/plot_shared_variance.py -u primitive\n\
# python $(project_folder)/scripts/plot_shared_variance.py -u social\n\
# python $(project_folder)/scripts/plot_shared_variance.py -u affective\n\
python $(project_folder)/scripts/plot_shared_variance.py" | sbatch

clean:
	rm *.out
	rm *.sh
