#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
import mne
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eeg import temporal
 


class reorganize_eeg:
    def __init__(self, args):
        self.process = 'ReorganizeEEG'
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
        print('loading eeg from MNE epochs...')
        # Load the MNE epochs file instead of .mat file
        # epochs_file = f'{self.data_dir}/interim/PreprocessRaw/{self.sid}/{self.sid}_preproc-epo.fif'
        epochs_file = f'{self.data_dir}/interim/PreprocessRaw/{self.sid}/filtered_epochs-epo.fif'

        if not os.path.exists(epochs_file):
            raise FileNotFoundError(f"MNE epochs file not found: {epochs_file}")
        
        epochs = mne.read_epochs(epochs_file, verbose=False)
        
        # Resample epochs to reduce memory usage
        print(f"Resampling epochs from {epochs.info['sfreq']} Hz to {self.resample_rate} Hz")
        epochs.resample(self.resample_rate)
        
        # Apply smoothing using MNE
        print(f"Applying smoothing with window size {self.n_samples_to_smooth}")
        kernel = np.ones(self.n_samples_to_smooth) / self.n_samples_to_smooth
        epochs.apply_function(lambda data: convolve1d(data, kernel))
        
        # Get the original trial indices for the kept epochs
        original_trial_indices = epochs.selection
        
        print(f"Original epochs: {len(epochs.drop_log)}, Remaining epochs: {len(epochs)}")
        print(f"Original trial indices for kept epochs: {original_trial_indices[:10]}...")
        
        eeg_dict = {
            'label': np.array(epochs.ch_names),
            'trial': epochs.get_data().astype(np.float32),  # Make it (1, n_epochs) to match [0] indexing
            'time': epochs.times * 1000,    # Make it (1, n_epochs) to match [0] indexing
            'original_trial_indices': original_trial_indices,  # Add the mapping to original trial indices
        }
        
        del epochs 

        print(f"Loaded {len(eeg_dict['trial'])} epochs, {len(eeg_dict['label'])} channels, {len(eeg_dict['time'])} time points")
        return eeg_dict
    
    def reorganize_and_add_trial_info(self, eeg_dict, trials_df): 
        df = []
        # Get the original trial indices for the kept epochs
        original_trial_indices = eeg_dict['original_trial_indices']
        
        # Filter trials_df to only include trials that have corresponding EEG data
        trials_df_ = trials_df.set_index('trial')
        
        print(f"Total trials in dataframe: {len(trials_df)}")
        print(f"Trials with EEG data: {len(trials_df_)}")
        
        iter_top = tqdm(zip(original_trial_indices, eeg_dict['trial']),
                        total=len(eeg_dict['trial']), desc='Reorganizing EEG')
        stimulus_dict = {}
        for original_trial_idx, trial_eeg in iter_top:
            
            # Get condition, response, video_name, stimulus_set
            vals = trials_df_.loc[original_trial_idx][['condition', 'response', 'video_name', 'stimulus_set']]
            cond, resp, name, stim = vals

            if bool(cond) and (not bool(resp)):

                # Keep track of repetitions for each stimulus
                if name in stimulus_dict:
                    stimulus_dict[name] += 1
                else:
                    stimulus_dict[name] = 0
                even = (stimulus_dict[name] % 2 == 0)

                for channel, channel_eeg in zip(eeg_dict['label'], trial_eeg):
                    for itime, (time, signal) in enumerate(zip(eeg_dict['time'], channel_eeg)):
                        df.append({'trial': original_trial_idx, 'channel': channel,
                                    'time': time, 'time_ind': itime,
                                    'signal': signal, 
                                    'repitition': stimulus_dict[name], 'even': even, 
                                    'video_name': name, 'stimulus_set': stim})
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

    def save(self, df):
        print('saving...')
        out_name = os.path.join(self.out_dir, 'all_trials', f'{self.sid}.parquet')
        df.to_parquet(out_name, index=False)
        print('Finished!')
    
    def save_time_df(self, df):
        for time_ind, time_df in df.groupby('time_ind'):
            out_name = os.path.join(self.out_dir, f'{self.sid}_time-{str(int(time_ind)).zfill(3)}.parquet')
            time_df.to_parquet(out_name, index=False)

    def plot_average_timecourse(self, df):
        # Average across channels for each trial
        trial_avg = df.groupby(['trial', 'video_name', 'time']).mean(numeric_only=True).reset_index()
        video_avg = trial_avg.groupby(['video_name', 'time']).mean(numeric_only=True).reset_index()
        video_avg = video_avg.sort_values('time')
        
        # Plot each trial in gray
        for _, video_data in video_avg.groupby('video_name'):
            plt.plot(video_data.time, video_data.signal, color='gray', alpha=0.5, linewidth=0.5)
        
        # Plot the average across trials in red
        overall_avg = video_avg.groupby('time').signal.mean().reset_index().sort_values('time')
        plt.plot(overall_avg.time, overall_avg.signal, color='red', linewidth=2, label='Average')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal (averaged across channels)')
        plt.title(f'Average Timecourse for {self.sid}')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.out_dir, 'all_trials', f'{self.sid}_average_timecourse.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f'Plot saved to {plot_path}')

    def run(self):
        trials = self.load_trials()
        eeg_dict = self.load_eeg(trials)
        eeg_df = self.reorganize_and_add_trial_info(eeg_dict, trials)
        print(eeg_df.head())

        self.save(eeg_df)
        eeg_averaged = self.average_repetitions(eeg_df)
        # self.save_time_df(eeg_averaged)
        self.plot_average_timecourse(eeg_df)
        print('All done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=int, default=2)
    parser.add_argument('--resample_rate', type=float, default=150)
    parser.add_argument('--n_samples_to_smooth', type=int, default=5)
    parser.add_argument('--data_dir', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data')
    args = parser.parse_args()
    reorganize_eeg(args).run()


if __name__ == '__main__':
    main()