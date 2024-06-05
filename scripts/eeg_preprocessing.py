import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import temporal
import torch
from pathlib import Path


class eegPreprocessing:
    def __init__(self, args):
        self.process = 'eegPreprocessing'
        self.eeg_dir = args.eeg_dir
        self.eeg_sub = f'sub-{str(args.eeg_sub).zfill(2)}'
        self.regress_gaze = args.regress_gaze
        self.smooth_step = args.smooth_step
        self.smooth_window = args.smooth_window
        self.smooth_min_period = args.smooth_min_period
        self.eeg_file = f'{self.eeg_dir}/{self.eeg_sub}_reg-gaze-{self.regress_gaze}.csv.gz'
        self.out_dir = args.out_dir

    @staticmethod
    def average_repetitions(data):
        df_mean = data.groupby(['time', 'video_name']).mean(numeric_only=True)
        cols = [col for col in df_mean.columns if 'channel' in col]
        df_filtered = df_mean[cols].reset_index()
        return df_filtered.sort_values(['time', 'video_name'])

    def smooth_eeg(self, data):
        cols = [col for col in data.columns if 'channel' in col]
        return temporal.smoothing(data, cols, grouping=['video_name'],
                                  window=self.smooth_window,
                                  step=self.smooth_step,
                                  min_periods=self.smooth_min_period)

    def load_eeg(self):
        return pd.read_csv(self.eeg_file)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def save_results(self, df):
        for itime, (_, df_time) in enumerate(df.groupby('time')):
            df_time.sort_values('video_name', inplace=True)
            out_file = f'{self.out_dir}/{self.eeg_sub}_time-{str(itime).zfill(2)}.csv.gz'
            df_time.to_csv(out_file, index=False)

    def run(self):
        df = self.load_eeg()
        df_averaged = self.average_repetitions(df)
        df_smoothed = self.smooth_eeg(df_averaged)
        self.mk_out_dir()
        self.save_results(df_smoothed)


def main():
    parser = argparse.ArgumentParser(description='Predict fMRI responses using the features')
    parser.add_argument('--eeg_dir', '-e', type=str, help='EEG directory')
    parser.add_argument('--eeg_sub', '-s', type=int, help='EEG subject number')
    parser.add_argument('--out_dir', '-o', type=str, help='output drirectory for the smoothing results')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False,
                        help='gaze regressed from the EEG time course')
    parser.add_argument('--smooth_step', type=int, default=1,
                        help='number of time points to step in smoothing')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='number of consecutive time points to smooth')
    parser.add_argument('--smooth_min_period', type=int, default=1,
                        help='number of time points that are required for smoothing kernel')
    args = parser.parse_args()
    eegPreprocessing(args).run()


if __name__ == '__main__':
    main()