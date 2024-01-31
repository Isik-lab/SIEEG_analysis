#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import stats, plotting


class Reliability:
    def __init__(self, args):
        self.process = 'Reliability'
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        self.regress_gaze = args.regress_gaze
        self.stimulus_set = args.stimulus_set
        print(vars(self))
        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_stimulus-set-{self.stimulus_set}_reg-gaze-{self.regress_gaze}_reliability.png'
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_stimulus-set-{self.stimulus_set}_reg-gaze-{self.regress_gaze}_reliability.csv'

    def load(self):
        df = pd.read_csv(f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        channels = [col for col in df.columns if 'channel' in col]
        return df, channels

    def run(self):
        df, channels = self.load()
        df = df.loc[df['stimulus_set'] == self.stimulus_set].reset_index()
        df_split = df.groupby(['time', 'video_name', 'even']).mean().reset_index()
        results = []
        time_groups = df_split.groupby('time')
        for time, time_df in time_groups:
            even = time_df[time_df.even].sort_values('video_name')
            odd = time_df[~time_df.even].sort_values('video_name')
            rs = stats.corr2d(even[channels].to_numpy(), odd[channels].to_numpy())
            results.append([time,] + list(rs))
        results = pd.DataFrame(results, columns=['time',] + channels)
        results = pd.melt(results, id_vars=['time'], value_vars=channels, var_name='channel', value_name='reliability')
        results.to_csv(self.out_file, index=False)
        plotting.plot_splithalf_reliability(results, self.out_figure)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='3')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--stimulus_set', type=str, default='test')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    Reliability(args).run()


if __name__ == '__main__':
    main()
