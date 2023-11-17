%%specify the header and EEG files to read
addpath /Applications/fieldtrip-20230926/
addpath progressbar/

input_path = '../../data/raw/SIdyads_EEG_pilot/';
out_path = '../../data/interim/SIdyads_EEG_pilot/';
subj_file = 'subj014';
hdrfile = [input_path, subj_file, '/', [subj_file, '.vhdr']];
eegfile = [input_path, subj_file, '/', [subj_file, '.eeg']];

prestim_time = 0.2;
photo_threshold = 0;
photodiode_lpf = 50; 
time_exclude = .1;
plotting = 0;
poststim_time = 1.25;
new_poststim_time = 1;
down = 1; 

%Set the direction of photodiode and the timing
if contains(subj_file, 'subj001')
    down = 0;
    poststim_time = 0.75;
    new_poststim_time = 0.6; 
elseif contains(subj_file, 'subj002')
    down = 0;
end

%Mark trials for removal. These trials should only be removed
%because the run was aborted or the participant was asleep
if contains(subj_file, 'subj005')
    trials_to_remove = 661:661+52; 
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
cfg.trialdef.poststim = poststim_time; %allow a buffer for realigning trials
cfg = ft_definetrial(cfg);
data = ft_preprocessing(cfg);
n_trls = size(data.sampleinfo, 1);

%% remove predefined trials
all_trials = ones(1, n_trls); 
all_trials(trials_to_remove) = 0; 
cfg.trials = logical(all_trials');
data = ft_preprocessing(cfg, data);

%% Adjust onset based on photodiode
toilim = [-1*prestim_time poststim_time];
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
data_minpreproc = ft_preprocessing(cfg, data_aligned);

%% visualize artifacts

if plotting
    cfg = [];
    cfg.continuous = 'no';
    cfg.preproc.demean = 'no';
    cfg.viewmode = 'butterfly';
    ft_databrowser(cfg, data_minpreproc);

    cfg = [];
    cfg.continuous = 'no';
    cfg.preproc.demean = 'no';
    cfg.viewmode = 'vertical';
    ft_databrowser(cfg, data_minpreproc);
end

%% detect muscle artifacts and replace them with NaNs

cfg_art = []; %create a configuration to store artifact definitions
cfg_art.artfctdef.muscle.channel = 'EEG';
cfg_art.artfctdef.muscle.continuous = 'no';
cfg_art.artfctdef.muscle.cutoff = 15; %z-score cutoff
cfg_art.artfctdef.muscle.bpfilter = 'yes';
cfg_art.artfctdef.muscle.bpfreq = [110 140];%high freq filtering
cfg_art.artfctdef.muscle.bpfiltord = 8;
cfg_art.artfctdef.muscle.hilbert = 'yes';
cfg_art.artfctdef.muscle.boxcar = 0.2;
cfg_art.artfctdef.muscle.artpadding = 0.1;%pad the detected artifacts
cfg_art.artfctdef.muscle.interactive = 'yes'; %adjust threshold if needed
cfg_art = ft_artifact_muscle(cfg_art, data_minpreproc);

%replace artifacts with NaNs
cfg_art.artfctdef.reject = 'nan';
data_muscle_preproc = ft_rejectartifact(cfg_art, data_minpreproc);

%save indices of trials with muscle artefacts
art = cfg_art.artfctdef.muscle.artifact; %get artefact latency info
badtrl_msc = eeg_badtrialidx(art, data_muscle_preproc); %gets index of trials with artefacts
zval = cfg_art.artfctdef.muscle.cutoff; %save zvalue

fprintf([num2str(round((length(badtrl_msc)/n_trls)*100)),...
    '%% of trials with muscle artifacts\n']);

%% reject bad channels/trials

cfg = [];
cfg.method = 'summary';
cfg.keeptrial = 'nan'; %trials are replaced with ‘nan’; channels are removed
data_variance_preproc = ft_rejectvisual(cfg, data_muscle_preproc);

%get remaining channel indices – so we know which channels to keep
chan = data_variance_preproc.label;

%get indices of trials with too-high variance
art = data_variance_preproc.cfg.artfctdef.summary.artifact;
badtrl_var = eeg_badtrialidx(art, data_variance_preproc);

fprintf([num2str(round((length(badtrl_var)/n_trls)*100)),...
    '%% of trials with high variance\n']);

%% make a combined trial index from photodiode, muscle, and variance rejection

badtrial_idx = false(numel(data.trial),1);
badtrial_idx(unique([badtrl_var; badtrl_msc; badtrl_photo])) = 1;

%keep only trials and channels marked as good
cfg = [];
cfg.trials = find(~badtrial_idx);
cfg.channel = chan;
data_clean = ft_preprocessing(cfg, data_variance_preproc);

%remove bad trials from offsets
offsets_clean = offsets(~badtrial_idx);

%optionally - rereference again - will be more robust after data cleaning
cfg = [];
cfg.channel = 'all';
cfg.reref = 'yes';
cfg.refchannel = 'all';
cfg.refmethod = 'avg';
data_clean = ft_preprocessing(cfg, data_clean);

%% ICA

%first - downsample to speed up ICA
cfg = [];
cfg.resamplefs = 150;
cfg.detrend = 'no';
data_clean_ds = ft_resampledata(cfg, data_clean);


%compute the rank of the data to constrain number of components
data_cat = cat(2,data_clean_ds.trial{:});
data_cat(isnan(data_cat)) = 0;
num_comp = rank(data_cat);

%now run ICA
cfg= [];
cfg.method = 'runica';
cfg.numcomponent = num_comp;
comp = ft_componentanalysis(cfg, data_clean_ds);

%plot components with their time-courses
cfg = [];
cfg.layout = 'acticap-64ch-standard2';
cfg.viewmode = 'component';
cfg.continuous = 'yes';
cfg.blocksize = 30; %use blocks of 30 s
ft_databrowser(cfg, comp);

% plot topographies for first components
figure
cfg = [];
cfg.component = 1:32;
cfg.layout = 'acticap-64ch-standard2';
cfg.comment = 'no';
ft_topoplotIC(cfg, comp)

%here give the component numbers to be removed, e.g. [7 10 34]
comp_rmv = input('Components to be removed (use square brackets if several): ');

%this projects the artefactual components out of the original data
cfg = [];
cfg.unmixing = comp.unmixing;
cfg.topolabel = comp.topolabel;
cfg.demean = 'no';
comp_orig = ft_componentanalysis(cfg, data_clean);

cfg = [];
cfg.component = comp_rmv;
cfg.demean = 'no'; %note - data is demeaned by default
data_ica_preproc = ft_rejectcomponent(cfg, comp_orig, data_clean);
close all

%% low-pass filter data
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 30;
data_lp_filtered = ft_preprocessing(cfg, data_ica_preproc);

%% Save data frame
progressbar
df = table();
labels = cellfun(@(c)['channel_' c], data_lp_filtered.label, 'uni', false);
for i=1:length(data_lp_filtered.trial)
    d = data_lp_filtered.trial{i}';
    time = data_lp_filtered.time{i};
    rows = cellstr(num2str(time(1:end)'));
    cols = labels;
    T = array2table(d(1:end, :), 'VariableNames', cols);
    T.time = rows;
    T.trial = ones(size(T,1),1)*(i-1);
    T.offset = ones(size(T,1),1)*offsets_clean(i); 
    df = vertcat(df, T); 
    progressbar(i/length(data_lp_filtered.trial)); %update progress bar
end

if ~exist([out_path, subj_file], 'dir')
    mkdir([out_path, subj_file]);
end
trial_file = [out_path, subj_file, '/', [subj_file, '_trials.csv']];
writetable(df, trial_file);
gzip(trial_file);
delete(trial_file)
clear df
fprintf('dataframe saved \n');


%% Save other formats
%save the artifact rejection info in a preproc structure that can be reused
fprintf('here \n')
comp = rmfield(comp, 'time');
comp = rmfield(comp, 'trial');
preproc.idx_badtrial = badtrial_idx;
preproc.badtrial_variance = badtrl_var;
preproc.badtrial_muscle = badtrl_msc;
preproc.muscle_zvalue = zval;
preproc.icacomponent = comp;
preproc.comp_rmv = comp_rmv;
preproc.chan = data_lp_filtered.label;
preproc.time = data_lp_filtered.time{1};

%save outputs
data_file = [out_path, subj_file, '/', [subj_file, '_data.mat']];
preproc_file = [out_path, subj_file, '/', [subj_file, '_preproc.mat']];
save(data_file, '-v7.3', '-struct', 'data_lp_filtered');
save(preproc_file,'-struct','preproc');
fprintf('saved and complete \n');

%% Functions
function [badtrl] = eeg_badtrialidx(art,rawdata)
badtrl = [];
if ~isempty(art)
    for i = 1:size(art,1)
        badtrl = [badtrl; find(rawdata.sampleinfo(:,1)<=art(i,1)&rawdata.sampleinfo(:,2)>=art(i,2))];
    end
end
end