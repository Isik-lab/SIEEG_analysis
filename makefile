user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
neptune_api_token=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTQxNWI5MS0wZjk4LTQ2Y2YtYWVmMC1kNzM1ZWVmZGFhOWUifQ==
eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21
features := alexnet moten expanse object agent_distance facingness joint_action communication valence arousal

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
back_to_back_swapped=$(project_folder)/data/interim/Back2Back_swapped

# Steps to run
all: motion_energy alexnet eeg_preprocess eeg_reliability full_brain back_to_back back_to_back_swapped

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
#SBATCH --time=2:45:00\n\
#SBATCH --cpus-per-task=12\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
export NEPTUNE_API_TOKEN=$(neptune_api_token)\n\
python $(project_folder)/scripts/back_to_back.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet -x2 '[\"$${x}\"]'" | sbatch; \
	done; \
	done
	touch $(back_to_back)/.done


#Compute the channel-wise EEG reliability
back_to_back_swapped: $(back_to_back_swapped)/.done $(eeg_preprocess)
$(back_to_back_swapped)/.done: 
	mkdir -p $(back_to_back_swapped)
	for x in $(features); do \
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=back_to_back_swapped\n\
#SBATCH --time=45:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/back_to_back_swapped.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet -x1 '[\"$${x}\"]'" | sbatch; \
	done; \
	done
	touch $(back_to_back_swapped)/.done


#Compute the channel-wise EEG reliability
full_brain: $(full_brain)/.done $(eeg_preprocess)
$(full_brain)/.done: 
	mkdir -p $(full_brain)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=a100\n\
#SBATCH --account=lisik3_gpu\n\
#SBATCH --job-name=full_brain\n\
#SBATCH --time=4:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/eeg_decode.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet --no-roi_mean --smoothing" | sbatch; \
	done
	touch $(back_to_back_swapped)/.done

clean:
	rm *.out
	rm *.sh
