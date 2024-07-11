#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from src import temporal
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np


def track_stimulus_repititions(stimulus_name, stimulus_dict):
    """
    This function takes an stimulus name and a dictionary. If the stimulus name is not
    present in the dictionary, it adds the stimulus with a counter starting at 0.
    If the stimulus name is present, it increments the existing count by 1.

    Args:
    stimulus_name (str): The name of the stimulus to be tracked.
    stimulus_dict (dict): A dictionary containing stimulus names as keys and their counts as values.

    Returns:
    tuple: The updated dictionary and the current count for the stimulus.
    """
    if stimulus_name in stimulus_dict:
        stimulus_dict[stimulus_name] += 1
    else:
        stimulus_dict[stimulus_name] = 0
    
    return stimulus_dict, stimulus_dict[stimulus_name]


def return_value(row, cols):
    """
    This function returns the value from a selected row in a pandas dataframe

    Args:
        row (pd.core.frame.DataFrame): a single rowed pandas dataframe
        cols (str or list): the columns to select

    Returns:
        str or list: the value in the selected column. type returned is the same as the input
    """
    if type(cols) == str:
        return row.iloc[0][cols]
    elif type(cols) == list:
        out = []
        for col in cols:
            out.append(row.iloc[0][col])
        return out


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
        stimulus_dict = {}
        for (itrial, trial_row), (times, trial_eeg) in iter_top:
            condition, response, video_name, stimulus_set = return_value(trial_row, ['condition',
                                                                                     'response',
                                                                                     'video_name', 
                                                                                     'stimulus_set'])
            if bool(condition) and (not bool(response)):
                stimulus_dict, repitition = track_stimulus_repititions(trial_row.iloc[0]['video_name'],
                                                                       stimulus_dict)
                even = (repitition % 2 == 0)
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
                                    'repitition': repitition, 'even': even, 
                                    'video_name': video_name, 'stimulus_set': stimulus_set})
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
        self.save(eeg_df, f'all_trials/{self.sid}.csv.gz')
        eeg_averaged = self.average_repetitions(eeg_df)
        self.save_time_df(eeg_averaged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=int, default=1)
    parser.add_argument('--resample_rate', type=float, default=2.5)
    parser.add_argument('--n_samples_to_smooth', type=int, default=5)
    parser.add_argument('--data_dir', '-d', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data')
    args = parser.parse_args()
    eegPreprocessing(args).run()


if __name__ == '__main__':
    main()