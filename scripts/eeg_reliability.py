#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import loading
from src.stats import bootstrap_gpu, compute_score
from tqdm import tqdm
import numpy as np


class eegReliability:
    def __init__(self, args):
        self.process = 'eegReliability'
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
        eeg_filtered = eeg.loc[eeg.stimulus_set == 'test'].reset_index(drop=True) #Filter to the test set
        eeg_average = eeg_filtered.groupby(['time', 'channel', 'video_name', 'even']).mean(numeric_only=True)
        return eeg_average.reset_index()

    def reliability(self, df):
        results = []
        iterator = tqdm(df.groupby('time'), total=df.time.nunique(), desc='Calculating reliability')
        for time, time_df in iterator:
            rs = 0
            vars = np.zeros(5000)
            for ichannel, (__cached__, channel_df) in enumerate(time_df.groupby('channel')):
                even = channel_df.loc[channel_df['even'], 'signal'].to_numpy()
                odd = channel_df.loc[~channel_df['even'], 'signal'].to_numpy()
                rs += compute_score(even, odd).cpu().detach().numpy()
                vars += bootstrap_gpu(even, odd).cpu().detach().numpy()
            time_dict = {'time': time, 'r': rs/(ichannel+1)}
            time_dict.update({f'var_perm_{i}': val for i, val in enumerate(vars/(ichannel+1))})
            results.append(time_dict)
        return pd.DataFrame(results)

    def save(self, df):
        df.to_parquet(f'{self.out_dir}/{self.sid}.parquet')

    def run(self):
        df = self.load_and_average()
        results = self.reliability(df)
        self.save(results)
        print('finished')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='1')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegReliability')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-01.parquet')
    args = parser.parse_args()
    eegReliability(args).run()


if __name__ == '__main__':
    main()
