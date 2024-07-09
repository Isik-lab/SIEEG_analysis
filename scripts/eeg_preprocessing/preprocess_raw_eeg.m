%%specify the header and EEG files to read
addpath /Applications/fieldtrip-20230926/

input_path = '../../data/raw/SIdyads_EEG/';
out_path = '../../data/interim/SIdyads_EEG/';
%subj list after removing bad subject
s_list = {'01', '02', '03', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'};

prestim_time = 0.2;
photo_threshold = 0;
photodiode_lpf = 50;
time_exclude = .1;
plotting = 0;
raw_poststim_tim = 1.25;
aligned_poststim_time = 1;
down = 1;

for i=1:20
    %% Setup
    subj_file = ['sub-', s_list{i}];
    fprintf('Starting', subj_file, '\n');
    hdrfile = [input_path, subj_file, '/', [subj_file, '.vhdr']];
    eegfile = [input_path, subj_file, '/', [subj_file, '.eeg']];

    %Mark trials for removal. These trials should only be removed
    %because the run was aborted or the participant was asleep
    if contains(subj_file, 'sub-01')
        trials_to_remove = 881; %Aborted run
    else
        trials_to_remove = [];
    end

    %% define trials and realign to photodiode onset
    cfg = [];
    cfg.headerfile = hdrfile;
    cfg.datafile = eegfile;
    % cfg.representation = 'table';
    cfg.trialdef.eventtype = 'Stimulus';
    cfg.trialdef.eventvalue = 'S  1';
    cfg.trialdef.prestim = prestim_time;
    cfg.trialdef.poststim = raw_poststim_tim; %allow a buffer for realigning trials
    cfg = ft_definetrial(cfg);
    data = ft_preprocessing(cfg);
    n_trls = size(data.sampleinfo, 1);

    %% remove predefined trials
    all_trials = ones(1, n_trls);
    all_trials(trials_to_remove) = 0;
    cfg.trials = logical(all_trials');
    data = ft_preprocessing(cfg, data);

    %% Adjust onset based on photodiode
    toilim = [-1*prestim_time aligned_poststim_time];
    frames_per_second = data.hdr.Fs;
    onset_sample_number = prestim_time * frames_per_second;
    [data_aligned, offsets, badtrl_photo] = eeg_alignphoto(data, toilim,...
        photo_threshold, down, ...
        onset_sample_number, frames_per_second, ...
        plotting,...
        photodiode_lpf, time_exclude); %custom function to fix triggers

    fprintf([num2str(round((length(badtrl_photo)/n_trls)*100)),...
        '%% bad photo trials\n']);
    close all;

    %% minimally preprocess data before artefact rejection
    cfg = [];
    cfg.channel = {'all', '-photodiode'}; %remove photodiode channel
    cfg.demean = 'yes'; %demean data
    cfg.baselinewindow = [-0.2 0]; %use pre-trigger period for baselining
    cfg.reref = 'yes'; %rereference data
    cfg.implicitref = 'Cz'; %add the implicit online ref back in
    cfg.refchannel = 'all'; %use average across all channels
    cfg.refmethod = 'avg';
    cfg.hpfilter = 'yes'; %high-pass filter to remove slow drifts
    cfg.hpfreq = 0.1;
    cfg.hpfiltord = 4;
    cfg.lpfilter = 'yes';
    cfg.lpfreq = 60;
    data_minpreproc = ft_preprocessing(cfg, data_aligned);

    %% Save
    %save outputs
    preproc_file = [out_path, subj_file, '/', [subj_file, '_preproc.mat']];
    save(preproc_file,'-struct','data_minpreproc');
    fprintf(subj_file, 'saved and complete \n') ;
end
