user=$(shell whoami)
project_folder=/home/$(user)/scratch4-lisik3/$(user)/SIEEG_analysis
eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21
fmri_subs := 1 2 3 4
features := alexnet moten expanse object agent_distance facingness joint_action communication valence arousal

# Dependencies
videos=$(project_folder)/data/raw/videos_3000ms
motion_energy=$(project_folder)/data/interim/MotionEnergyActivations
alexnet=$(project_folder)/data/interim/AlexNetActivations
fmri_data=$(project_folder)/data/interim/ReorganizefMRI

matlab_eeg_path=$(project_folder)/data/interim/SIdyads_EEG
eeg_preprocess=$(project_folder)/data/interim/eegPreprocessing
eeg_reliability=$(project_folder)/data/interim/eegReliability
back_to_back=$(project_folder)/data/interim/Back2Back
fmri_regression=$(project_folder)/data/interim/fMRIRegression
feature_regression=$(project_folder)/data/interim/FeatureRegression
feature_plotting=$(project_folder)/data/interim/PlotFeatureDecoding
roi_plotting=$(project_folder)/data/interim/PlotROIDecoding
back2back_plotting=$(project_folder)/data/interim/PlotBack2Back

# Steps to run
all: motion_energy alexnet eeg_preprocess eeg_reliability feature_decoding roi_decoding full_brain back_to_back plot_rois plot_features plot_back2back


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
#SBATCH --time=6:00:00\n\
#SBATCH --cpus-per-task=16\n\
#SBATCH --exclusive=user\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
echo $${s}\n\
python $(project_folder)/scripts/eeg_reliability.py -s $$s" | sbatch; \
	done
	# touch $(eeg_reliability)/.done


#Compute b2b regression with EEG first then annotated features
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
#SBATCH --cpus-per-task=16\n\
#SBATCH --exclusive=user\n\
set -e\n\
ml anaconda\n\
conda activate eeg\n\
echo $${x}\n\
python $(project_folder)/scripts/back_to_back.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet -x2 '[\"$${x}\"]'" | sbatch; \
	done; \
	done
	# touch $(back_to_back)/.done


#Compute b2b regression with EEG first then annotated features
feature_decoding: $(feature_regression)/.feature_decoding $(eeg_preprocess)
$(feature_regression)/.feature_decoding: 
	mkdir -p $(feature_regression)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=feature_decoding\n\
#SBATCH --time=2:45:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/feature_regression.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet" | sbatch; \
	done
	# touch $(feature_regression)/.feature_decoding


#Compute the channel-wise EEG reliability
roi_decoding: $(fmri_regression)/.done $(eeg_preprocess)
$(fmri_regression)/.done: 
	mkdir -p $(fmri_regression)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=roi_decoding\n\
#SBATCH --time=2:45:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/fmri_regression.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet" | sbatch; \
	done
	# touch $(fmri_regression)/.roi_decoding


#Full brain EEG to fMRI regression
full_brain: $(fmri_regression)/.full_brain $(eeg_preprocess)
$(fmri_regression)/.full_brain: 
	mkdir -p $(fmri_regression)
	for s in $(eeg_subs); do \
		echo -e "#!/bin/bash\n\
#SBATCH --partition=a100\n\
#SBATCH --account=lisik3_gpu\n\
#SBATCH --job-name=full_brain\n\
#SBATCH --time=45:00\n\
#SBATCH --cpus-per-task=12\n\
#SBATCH --gres=gpu:1\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/fmri_regression.py -e $(eeg_preprocess)/all_trials/sub-$$(printf '%02d' $${s}).parquet --no-roi_mean --smoothing" | sbatch; \
	done
	# touch $(fmri_regression)/.full_brain


#Plot the ROI timecourses 
plot_rois: $(roi_plotting)/.plotted $(fmri_regression)
$(roi_plotting)/.plotted: 
	mkdir -p $(roi_plotting)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=roi_plotting\n\
#SBATCH --ntasks=1\n\
#SBATCH --time 1:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/plot_nuisance_roi_decoding.py --overwrite" | sbatch
	# touch $(roi_plotting)/.plotted


#Plot the ROI timecourses 
plot_features: $(feature_plotting)/.plotted $(feature_decoding)
$(feature_plotting)/.plotted: 
	mkdir -p $(feature_plotting)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=feature_plotting\n\
#SBATCH --ntasks=1\n\
#SBATCH --time 1:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/plot_nuisance_feature_decoding.py --overwrite" | sbatch
	# touch $(feature_plotting)/.plotted


#Plot the Back2Back timecourses 
plot_back2back: $(back2back_plotting)/.plotted $(back_to_back)
$(back2back_plotting)/.plotted: 
	mkdir -p $(back2back_plotting)
	printf "#!/bin/bash\n\
#SBATCH --partition=shared\n\
#SBATCH --account=lisik33\n\
#SBATCH --job-name=back2back_plotting\n\
#SBATCH --ntasks=1\n\
#SBATCH --time 3:00:00\n\
#SBATCH --cpus-per-task=12\n\
ml anaconda\n\
conda activate eeg\n\
python $(project_folder)/scripts/plot_back2back.py --overwrite" | sbatch
	# touch $(back2back_plotting)/.plotted


clean:
	rm *.out
	rm *.sh
