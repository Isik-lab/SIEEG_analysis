%%specify the header and EEG files to read
addpath fieldtrip/

input_path = '../../data/raw/SIdyads_EEG/';
out_path = '../../data/interim/SIdyads_EEG/';
%subj list after removing bad subject
s_list = {'01', '02', '03', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'};
% s_list = {'01', '02'};
n_subjs = length(s_list); 


prestim_time = 0.2;
photo_threshold = 0;
photodiode_lpf = 50;
time_exclude = .1;
plotting = 0;
raw_poststim_tim = 1.25;
aligned_poststim_time = 1;
down = 1;

all_data_avg = cell(1, n_subjs);  % Preallocate for 20 subjects
for i=1:n_subjs
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

    %% minimally preprocess data before
    cfg = [];
    cfg.channel = {'all', '-photodiode'}; %remove photodiode channel
    cfg.refmethod = 'avg';
    cfg.preproc.demean = 'yes'; %demean data
    cfg.preproc.baselinewindow = [-0.2 0]; %use pre-trigger period for baselining
    cfg.preproc.reref = 'yes'; %rereference data
    cfg.preproc.implicitref = 'Cz'; %add the implicit online ref back in
    cfg.preproc.refchannel = 'all'; %use average across all channels
    cfg.preproc.hpfilter = 'yes'; %high-pass filter to remove slow drifts
    cfg.preproc.hpfreq = 0.1;
    cfg.preproc.hpfiltord = 4;
    cfg.preproc.lpfilter = 'yes';
    cfg.preproc.lpfreq = 60;
    data_minpreproc = ft_preprocessing(cfg, data_aligned);

    %% Average over trials for grandaverage topography visualization
    data_avg = ft_timelockanalysis(cfg, data_minpreproc);

    % Store in the cell array (with subject identifier)
    all_data_avg{i} = data_avg;
    all_data_avg{i}.subject = s_list{i};  % Optional: Track subject ID

    %% Save
    %save outputs
    preproc_file = [out_path, subj_file, '/', [subj_file, '_preproc.mat']];
    save(preproc_file,'-struct','data_minpreproc');
    fprintf(subj_file, 'saved and complete \n') ;
end

%% Grand average
grand_avg = ft_timelockgrandaverage(cfg, all_data_avg{:});

%% Topo Plot
close all;
time_step = 0.1;
cfg = [];
cfg.xlim = [-0.2:time_step:1.0];  % Time intervals
cfg.zlim = [-10 10];         % Color limits
cfg.layout = 'acticap-64ch-standard2.mat';
cfg.colorbar = 'no';         % We'll add it manually later
cfg.figure = figure('Color', 'white');  % Force white background

% Key addition: Disable all FieldTrip text annotations
cfg.interactive = 'no';      % Disables interactive features that add text
cfg.showlabels = 'no';       % Explicitly disable channel labels
cfg.comment = 'no';          % Disable automatic comment (time range text)
cfg.commentpos = 'title';    % Even if comment was enabled, put it in title (but we disabled it)

% Plot topoplots
ft_topoplotER(cfg, grand_avg);

% Post-processing to clean up the figure
nPlots = length(cfg.xlim)-1;
n_rows = 3;
n_cols = 4;
for i = 1:nPlots
    subplot(n_rows, n_cols, i);  % Adjust based on your layout (12 plots = 3x4)
    pbaspect([4 3 1]);
    
    % Add time interval as title (replace subplot info)
    title(sprintf('time = [%0.2f, %0.2f] s', cfg.xlim(i), cfg.xlim(i)+time_step), ...
      'FontSize', 8, 'FontWeight', 'normal', 'Color', 'black');
    
    % Remove axis labels (they're redundant for topoplots)
    set(gca, 'XTick', [], 'YTick', []);
    
    % Add colorbar only to the last subplot
    if mod(i,n_cols) == 0
        c = colorbar('eastoutside');
        c.Label.String = 'Amplitude (\muV)';  % FieldTrip standard
        c.Label.FontSize = 8;
        c.Label.Color = 'black';              % Title color

        % Set tick marks and labels to black
        c.Color = 'black';                    % Tick marks color
        c.TickLabels = strsplit(num2str(c.Ticks)); % Ensure tick labels exist
        set(c, 'FontSize', 8, 'Color', 'black');   % Tick label color & font
        pbaspect([2 2 3]); 
    end
end

% Adjust figure margins
set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf, 'topoplot.jpg');

%% ERP Trace
close all;
cfg = [];
cfg.layout = 'acticap-64ch-standard2.mat';
cfg.showcomment = 'no';
cfg.showlabels = 'yes';
cfg.showoutline = 'yes';
cfg.showscale = 'no';
cfg.linewidth = 1.5;
cfg.fontsize = 6;
cfg.figure = figure('Color', 'white');  % Force white background
ft_multiplotER(cfg, grand_avg);
set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf, 'traceplot.jpg');