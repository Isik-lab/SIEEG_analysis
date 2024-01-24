#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from itertools import combinations
from src import rsa, plotting


class PairwiseDecoding:
    def __init__(self, args):
        self.process = 'PairwiseDecoding'
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        self.n_groups = args.n_groups
        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        self.regress_gaze = args.regress_gaze
        self.stimulus_set = args.stimulus_set
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_stimulus-set-{self.stimulus_set}_reg-gaze-{self.regress_gaze}.csv.gz'
        self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_stimulus-set-{self.stimulus_set}_reg-gaze-{self.regress_gaze}.png'
        print(vars(self))

    def run(self):
        print('loading data...')
        df = pd.read_csv(f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        df = df.loc[df['stimulus_set'] == self.stimulus_set].reset_index()
        df.sort_values(['time', 'video_name'], inplace=True)
        videos = df.video_name.unique()
        videos_nCk = list(combinations(videos, 2))
        channels = [col for col in df.columns if 'channel' in col]

        results = rsa.eeg_decoding_distance(df, channels, videos_nCk, self.n_groups)
        results.to_csv(self.out_file, index=False, compression='gzip')
        plotting.plot_pairwise_decoding(results, self.out_figure)
        print('Finished!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--n_groups', type=int, default=5)
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--stimulus_set', type=str, default='test')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    PairwiseDecoding(args).run()


if __name__ == '__main__':
    main()