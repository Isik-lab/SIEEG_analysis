function [data, badtrl_photo] = eeg_alignphoto(data, toilim,...
    photo_threshold, down,...
    onset_sample_number, frames_per_second, ...
    plotting)
%realign trials to photodiode onset
%searches for photodiode down flanks within specified toilim
%assumes channel is labelled photodiode
%uses a z-score threshold of 0 to detect triggers
% D.C. Dima (diana.c.dima@gmail.com) Feb 2020
% Edited by E McMahon (emaliemcmahon@gmail.com) Oct 2023

% Setting variables
low_pass_filter = 50;
time_exclude = .1; %s
sample_exclude = time_exclude * frames_per_second;

%get the data into matrix format
trl = cat(3,data.trial{:});

%look for photodiode onsets within a limited time window
%to avoid finding previous/next trial triggers
time = data.time{1};
t1 = nearest(time,toilim(1));
t2 = nearest(time,toilim(2));
trl = trl(:,t1:t2,:);

ntmp = size(trl,2);
ntrl = size(trl,3);

%get and normalize photodiode data
photoidx = contains(data.label,'photodiode');
photodat = squeeze(trl(photoidx,:,:));
photodat = (photodat - repmat(nanmean(photodat,1),ntmp,1))./repmat(std(photodat,[],1),ntmp,1);

%here we can interactively adjust threshold based on figure - uncomment below
% figure;plot(time(t1:t2),photodat)
% if contains(data.cfg.headerfile)
% t = input(sprintf('Threshold for photodiode triggers is %.1f. Change or press Enter: ', thresh));
% if ~isempty(t) && isnumeric(t), thresh = t; end
% close

% down or up triggers
progressbar
photosmp = nan(ntrl,1);
fprintf(['onset sample number: ', num2str(onset_sample_number), '\n']);
fprintf(['exclude sample number: ', num2str(onset_sample_number+sample_exclude), '\n']);

for itrl = 1:ntrl
    x = photodat(:, itrl);
    x_lpf = lowpass(x, low_pass_filter, frames_per_second);
    x_zscored = (x_lpf-mean(x_lpf))/std(x_lpf);
    if down
        index = find(x_zscored < photo_threshold, 1, 'first');
    else
        index = find(x_zscored > photo_threshold, 1, 'first');
    end

    if ~isempty(index)
        photosmp(itrl) = index;
    end

    if plotting && (photosmp(itrl)<onset_sample_number || photosmp(itrl)>(onset_sample_number+sample_exclude))
        figure(itrl+1); plot([x, x_lpf, x_zscored]);
        title(['Light on sample number: ', num2str(photosmp(itrl))]);
    end
    progressbar(itrl/ntrl); %update progress bar
end

%sometimes (n=3) the photodiode triggers don't work as expected in ~20% of trials
badtrl_photo = logical(isnan(photosmp) + (photosmp<onset_sample_number) + (photosmp>(onset_sample_number+sample_exclude))); %remove bad trigger trials
photosmp(badtrl_photo) = mean(photosmp(~badtrl_photo)); %these will be read in based on average offset and removed later
badtrl_photo = find(badtrl_photo);

%get offset from 0 based on original time axis
zerotime = nearest(time(t1:t2),0);
offsets = -(photosmp-zerotime); %negative offset if trial began befind(x > thresh, 1, 'first')fore new trigger

%realign the trials
cfg = [];
cfg.offset = offsets; %the offsets are added to the current time to create new time axis
cfg.trials = 'all';
cfg.overlap = 1;
data = ft_redefinetrial(cfg,data);

%cut the epochs of interest (non-overlapping and only up to 1 s after onset)
cfg = [];
cfg.toilim = toilim;
data = ft_redefinetrial(cfg,data);

%check that the realignment worked
si = reshape(data.sampleinfo',numel(data.sampleinfo),1);
if ~isempty(find(diff(si)<0,1))
    warning('\n There is some overlap in trials after realignment!')
end
end