#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from src import preprocessing, temporal
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np


class eegPreprocessing:
    def __init__(self, args):
        self.process = 'eegPreprocessing'
        self.data_dir = args.data_dir
        self.sid = f'sub-{str(args.sid).zfill(2)}'
        self.resample_rate = args.resample_rate
        self.n_samples_to_smooth = args.n_samples_to_smooth
        print(vars(self))
        self.out_dir = f'{self.data_dir}/interim/{self.process}'
        Path(f'{self.out_dir}/all_trials').mkdir(parents=True, exist_ok=True)

    @staticmethod
    def average_repetitions(data):
        return data.groupby(['time', 'video_name', 'channel']).mean(numeric_only=True).reset_index()
    
    def load_eeg(self, trials):
        print('loading eeg...')
        file = f'{self.data_dir}/interim/SIdyads_EEG/{self.sid}/{self.sid}_preproc.mat'
        data_dict = loadmat(file)
        return data_dict
    
    def reorganize_and_resample(self, eeg_dict, trials_df): 
        df = []
        iter_top = tqdm(zip(trials_df.groupby('trial'), zip(eeg_dict['time'][0],
                                                          eeg_dict['trial'][0])),
                        total=len(eeg_dict['trial'][0]), desc='Reorganizing EEG')
        for (itrial, trial_row), (times, trial_eeg) in iter_top:
            times = times[0] * 1000 # Change to ms
            for channel, channel_eeg in zip(eeg_dict['label'], trial_eeg):
                channel = channel[0][0]
                resampled_time, resampled_data = temporal.resample(times, channel_eeg,
                                                                   new_sample_rate=self.resample_rate)
                smoothed_data = temporal.smooth(resampled_data, window_size=self.n_samples_to_smooth)
                for (itime, time), signal in zip(enumerate(resampled_time), smoothed_data):
                    df.append({'trial': itrial, 'channel': channel,
                                'time': time, 'time_ind': itime,
                                'signal': signal,
                                'video_name': trial_row.iloc[0]['video_name'],
                                'stimulus_set': trial_row.iloc[0]['stimulus_set'],
                                'condition': trial_row.iloc[0]['condition'],
                                'response': trial_row.iloc[0]['response']})
        return pd.DataFrame(df) 

    def load_trials(self):
        trial_files = f'{self.data_dir}/raw/SIdyads_trials/{self.sid}/timingfiles/*.csv'
        test_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/test.csv')['video_name'].to_list()

        trials = []
        for run, tf in enumerate(sorted(glob(trial_files))):
            t = pd.read_csv(tf)
            t['run'] = run
            t['run_file'] = tf
            trials.append(t)
        trials = pd.concat(trials).reset_index(drop=True)
        trials.reset_index(inplace=True)
        trials.rename(columns={'index': 'trial'}, inplace=True)
        
        # Add information about the training and test split
        trials['stimulus_set'] = 'train'
        trials.loc[trials.video_name.isin(test_videos), 'stimulus_set'] = 'test'
        return trials[['trial', 'video_name', 'condition', 'stimulus_set', 'response']]

    def save(self, df, name):
        print('saving...')
        df.to_csv(f'{self.out_dir}/{name}', compression='gzip', index=False)
        print('Finished!')

    def save_time_df(self, df):
        for time_ind, time_df in df.groupby('time_ind'):
            time_df.to_csv(f'{self.out_dir}/{self.sid}_time-{str(int(time_ind)).zfill(3)}.csv')

    def run(self):
        trials = self.load_trials()
        eeg_dict = self.load_eeg(trials)
        eeg_df = self.reorganize_and_resample(eeg_dict, trials)
        print(eeg_df.head())
        eeg_filtered = preprocessing.label_repetitions(preprocessing.filter_catch_trials(eeg_df))
        self.save(eeg_filtered, f'all_trials/{self.sid}.csv.gz')
        eeg_averaged = self.average_repetitions(eeg_filtered)
        self.save_time_df(eeg_averaged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=int, default=1)
    parser.add_argument('--resample_rate', type=int, default=2.5)
    parser.add_argument('--n_samples_to_smooth', type=int, default=5)
    parser.add_argument('--data_dir', '-d', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data')
    args = parser.parse_args()
    eegPreprocessing(args).run()


if __name__ == '__main__':
    main()