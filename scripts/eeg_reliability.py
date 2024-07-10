#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import stats, plotting, loading
from tqdm import tqdm


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
        print(f'{len(eeg)=}')
        print(f'{eeg.trial.nunique()=}')
        print(f'{eeg.time.nunique()=}')
        print(f'{eeg.channel.nunique()=}')
        print(f'{(eeg.channel.nunique()*eeg.time.nunique()*eeg.trial.nunique())==len(eeg)=}')
        eeg_filtered = eeg.loc[eeg.stimulus_set == 'test'].reset_index(drop=True) #Filter to the test set
        eeg_average = eeg_filtered.groupby(['time', 'channel', 'video_name', 'even']).mean(numeric_only=True)
        return eeg_average.reset_index()

    def reliability(self, df):
        results = []
        iterator = tqdm(df.groupby('time'), total=df.time.nunique(), desc='Calculating reliability')
        for time, time_df in iterator:
            for channel, channel_df in time_df.groupby('channel'):
                even = channel_df.loc[channel_df['even'], 'signal'].to_numpy()
                odd = channel_df.loc[~channel_df['even'], 'signal'].to_numpy()
                results.append({'time': time, 'channel': channel, 'r': stats.corr(even, odd)})
        return pd.DataFrame(results)

    def save(self, df):
        df.to_csv(f'{self.out_dir}/sub-{self.sid}.csv', index=False)

    def run(self):
        df = self.load_and_average()
        results = self.reliability(df)
        self.save(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='1')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegReliability')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-01.csv.gz')
    args = parser.parse_args()
    eegReliability(args).run()


if __name__ == '__main__':
    main()
