#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import argparse
import pandas as pd

from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eeg import loading


class eegReliability:
    def __init__(self, args):
        self.process = 'eegReliability'
        self.data_split = args.data_split
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        print(vars(self))
        self.eeg_file = args.eeg_file
        self.out_dir = args.out_dir
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def load_and_average(self):
        eeg = loading.load_eeg(self.eeg_file)
        print(f'eeg.time.nunique(): {eeg.time.nunique()}')
        eeg_filtered = eeg.loc[eeg.stimulus_set == self.data_split].reset_index(drop=True) #Filter to the right data split
        eeg_average = eeg_filtered.groupby(['time', 'channel', 'video_name', 'even']).mean(numeric_only=True)
        return eeg_average.reset_index()

    def reliability(self, df):
        results = []
        iterator = tqdm(df.groupby('time'), total=df.time.nunique(), desc='Calculating reliability')
        for time, time_df in iterator:
            for ichannel, (channel, channel_df) in enumerate(time_df.groupby('channel')):
                even = channel_df.loc[channel_df['even'], 'signal'].to_numpy()
                odd = channel_df.loc[~channel_df['even'], 'signal'].to_numpy()
                r = pearsonr(even, odd).statistic
            
                time_dict = {'time': time, 'ichannel': ichannel, 'channel': channel, 'r': r}
                results.append(time_dict)
        return pd.DataFrame(results)

    def save(self, df):
        df.to_parquet(f'{self.out_dir}/{self.sid}_set-{self.data_split}.parquet')

    def plot_reliability(self, df):
        # Average reliability across channels
        df_wide = df.pivot(index='time', columns='channel', values='r')
        arr = df_wide.to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_wide.index, arr, linewidth=1, color='lightgray', alpha=0.7)
        plt.plot(df_wide.index, arr.mean(axis=1), color='blue', linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Average Reliability (r)')
        plt.title(f'Average Reliability over Trials for {self.sid}, {self.data_split} set')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f'{self.out_dir}/{self.sid}_set-{self.data_split}_reliability.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f'Plot saved to {plot_path}')

    def run(self):
        df = self.load_and_average()
        print(df.head())
        results = self.reliability(df)
        print(results.head())
        self.save(results)
        self.plot_reliability(results)
        print('finished')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='2')
    parser.add_argument('--data_split', '-d', type=str, default='test')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/eegReliability')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizeEEG/all_trials/sub-02.parquet')
    args = parser.parse_args()
    eegReliability(args).run()


if __name__ == '__main__':
    main()
