#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import loading
from src.stats import bootstrap_gpu, perm_gpu, compute_score
from tqdm import tqdm
import numpy as np


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
                r = compute_score(even, odd).cpu().detach().numpy()
                var =  bootstrap_gpu(even, odd).cpu().detach().numpy()
                null = perm_gpu(even, odd).cpu().detach().numpy()
                time_dict = {'time': time, 'ichannel': ichannel, 'channel': channel, 'r': r}
                time_dict.update({f'var_perm_{i}': val for i, val in enumerate(var)})
                time_dict.update({f'null_perm_{i}': val for i, val in enumerate(null)})
                results.append(time_dict)
                print(results[-1])
            break
        return pd.DataFrame(results)

    def save(self, df):
        df.to_parquet(f'{self.out_dir}/{self.sid}_set-{self.data_split}.parquet')

    def run(self):
        df = self.load_and_average()
        print(df.head())
        results = self.reliability(df)
        print(results.head())
        self.save(results)
        print('finished')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='1')
    parser.add_argument('--data_split', '-d', type=str, default='test')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegReliability')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-01.parquet')
    args = parser.parse_args()
    eegReliability(args).run()


if __name__ == '__main__':
    main()
