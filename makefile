user=$(shell whoami)
project_folder=/orcd/data/ngk/001/users/$(user)/SIEEG_analysis
conda_python=/orcd/data/ngk/001/users/$(user)/miniconda3/envs/eeg/bin/python
# eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21
eeg_subs := 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21
fmri_subs := 1 2 3 4
features := alexnet moten expanse object agent_distance facingness joint_action communication valence arousal

# Dependencies
videos=$(project_folder)/data/raw/videos_3000ms
motion_energy=$(project_folder)/data/interim/MotionEnergyActivations
alexnet=$(project_folder)/data/interim/AlexNetActivations

fmri_data=$(project_folder)/data/interim/ReorganizefMRI
reorganize_eeg=$(project_folder)/data/interim/ReorganizeEEG


preprocess_raw=$(project_folder)/data/interim/PreprocessRaw
eeg_reliability=$(project_folder)/data/interim/eegReliability
back_to_back=$(project_folder)/data/interim/Back2Back
fmri_regression=$(project_folder)/data/interim/fMRIRegression
whole_brain=$(project_folder)/data/interim/fMRIWholeBrain
feature_regression=$(project_folder)/data/interim/FeatureRegression
feature_plotting=$(project_folder)/data/interim/PlotFeatureDecoding
roi_plotting=$(project_folder)/data/interim/PlotROIDecoding
back2back_plotting=$(project_folder)/data/interim/PlotBack2Back
reliability_plotting=$(project_folder)/data/interim/PlotReliability


# Steps to run
all: reorg_fmri motion_energy alexnet preprocess_raw reorganize_eeg eeg_reliability feature_decoding roi_decoding full_brain back_to_back plot_rois plot_features plot_back2back plot_reliability 


# Get the activations from AlexNet for the 500 ms videos
reorg_fmri: $(fmri_data)/.done $(videos)
$(fmri_data)/.done: 
	mkdir -p $(fmri_data)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh reorg_fmri 1:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/reorganize_fmri.py; $(conda_python) $(project_folder)/scripts/reorganize_fmri.py --fwhm 12" ""
	touch $(fmri_data)/.done

# Get the motion energy for the 3 s videos
motion_energy: $(motion_energy)/.done $(videos)
$(motion_energy)/.done: 
	mkdir -p $(motion_energy)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh motion_energy 5:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/motion_energy_activations.py" ""
	touch $(motion_energy)/.done



# Build and run AlexNet activations (with shim)
# Ensure the ijit shim is built before submitting the job so nodes without VTune libs
# will still resolve the iJIT symbols via LD_PRELOAD.
alexnet: ijit-shim $(alexnet)/.done $(videos)
$(alexnet)/.done: 
	mkdir -p $(alexnet)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh alexnet 1:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/alexnet_activations.py" "" "ml ffmpeg; export LD_PRELOAD=$(project_folder)/scripts/libijitshim.so"
	touch $(alexnet)/.done


# Preprocess raw EEG data
preprocess_raw: $(preprocess_raw)/.done
$(preprocess_raw)/.done: 
	mkdir -p $(preprocess_raw)
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh preprocess_raw 10:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/preprocess_raw.py -s $$s" ""; \
	done
# 	touch $(preprocess_raw)/.done

# Exclude bad subjects identified during preprocessing
bad_eeg_subs := 10
eeg_subs := $(filter-out $(bad_eeg_subs), $(eeg_subs))


# Preprocess EEG data for regression
reorganize_eeg: $(reorganize_eeg)/.preprocess_done $(preprocess_raw) $(fmri_data)
$(reorganize_eeg)/.preprocess_done: 
	mkdir -p $(reorganize_eeg)
	for s in $(eeg_subs); do \
		echo -e "Submitting reorganize_eeg job for $$s"; \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh reorganize_eeg 3:00:00 32 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/reorganize_eeg.py -s $$s" "--mem=64G"; \
	done
# 	touch $(reorganize_eeg)/.preprocess_done


#Compute the channel-wise EEG reliability
eeg_reliability: $(eeg_reliability)/.done $(reorganize_eeg)
$(eeg_reliability)/.done: 
	mkdir -p $(eeg_reliability)
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh eeg_reliability 2:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/eeg_reliability.py -s $$s" "echo $${s}"
	done
	touch $(eeg_reliability)/.done


#Compute b2b regression with EEG first then annotated features
back_to_back: $(back_to_back)/.done $(reorganize_eeg)
$(back_to_back)/.done: 
	mkdir -p $(back_to_back)
	for x in $(features); do \
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh back_to_back 2:45:00 16 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/back_to_back.py -e $(reorganize_eeg)/all_trials/sub-$$(printf '%02d' $${s}).parquet -x '["$${x}"]'" "" "echo $${x}"; \
			done; \
			done
	touch $(back_to_back)/.done


#Compute EEG feature regression
feature_decoding: $(feature_regression)/.feature_decoding $(reorganize_eeg)
$(feature_regression)/.feature_decoding: 
	mkdir -p $(feature_regression)
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh feature_decoding 30:00 16 ou_bcs_low "$(conda_python) $(project_folder)/scripts/feature_regression.py -e $(reorganize_eeg)/all_trials/sub-$$(printf '%02d' $${s}).parquet" ""; \
	done
# 	touch $(feature_regression)/.feature_decoding


#Compute the channel-wise roi_decoding
roi_decoding: $(fmri_regression)/.done $(reorganize_eeg)
$(fmri_regression)/.done: 
	mkdir -p $(fmri_regression)
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh roi_decoding 45:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/joint_regression.py -e $(reorganize_eeg)/all_trials/sub-$$(printf '%02d' $${s}).parquet" ""; \
	done
	touch $(fmri_regression)/.roi_decoding


#Full brain EEG to fMRI regression
full_brain: $(fmri_regression)/.full_brain $(reorganize_eeg)
$(fmri_regression)/.full_brain: 
	mkdir -p $(fmri_regression)
	for s in $(eeg_subs); do \
		bash $(project_folder)/batch_scripts/submit_sbatch.sh full_brain 45:00 12 ou_bcs_low "$(conda_python) $(project_folder)/scripts/fmri_regression.py -e $(reorganize_eeg)/all_trials/sub-$$(printf '%02d' $${s}).parquet --no-roi_mean --smoothing" "--gres=gpu:1"; \
	done
	touch $(fmri_regression)/.full_brain


#Full brain Averaging and NIFTI image saving
full_brain_avg: $(whole_brain)/.full_brain $(fmri_regression)
$(whole_brain)/.full_brain: 
	mkdir -p $(whole_brain)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh whole_brain 15:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/fmri_whole_brain.py" "--gres=gpu:1"
	touch $(whole_brain)/.full_brain


#Plot the EEG Reliability
plot_reliability: $(reliability_plotting)/.plotted $(eeg_reliability)
$(reliability_plotting)/.plotted: 
	mkdir -p $(reliability_plotting)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh reliability_plotting 1:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/plot_reliability.py --overwrite" ""
	touch $(reliability_plotting)/.plotted


#Plot the ROI timecourses 
plot_rois: $(roi_plotting)/.plotted $(fmri_regression)
$(roi_plotting)/.plotted: 
	mkdir -p $(roi_plotting)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh roi_plotting 1:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/plot_roi_decoding.py --overwrite; $(conda_python) $(project_folder)/scripts/plot_roi_decoding.py --simplified_plotting" ""
	touch $(roi_plotting)/.plotted


#Plot the ROI timecourses 
plot_features: $(feature_plotting)/.plotted $(feature_decoding)
$(feature_plotting)/.plotted: 
	mkdir -p $(feature_plotting)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh stats_testing 3:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/plot_feature_decoding.py --overwrite" ""
# 	touch $(feature_plotting)/.plotted


#Plot the Back2Back timecourses 
plot_back2back: $(back2back_plotting)/.plotted $(back_to_back)
$(back2back_plotting)/.plotted: 
	mkdir -p $(back2back_plotting)
	bash $(project_folder)/batch_scripts/submit_sbatch.sh back2back_plotting 3:00:00 12 ou_bcs_normal "$(conda_python) $(project_folder)/scripts/plot_back2back.py --overwrite" ""
	touch $(back2back_plotting)/.plotted


clean:
	rm *.out
	rm *.sh
	rm -f $(project_folder)/scripts/libijitshim.so