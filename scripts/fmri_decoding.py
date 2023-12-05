#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import temporal, decoding
import os
from src.mri import Benchmark
import torch

class fMRIDecoding:
    def __init__(self, args):
        self.process = 'fMRIDecoding'
        if 'u' not in args.sid:
            self.sid = f'subj{str(int(args.sid)).zfill(3)}'
        else:
            self.sid = args.sid
        self.regress_gaze = args.regress_gaze
        self.overwrite = args.overwrite
        print(vars(self))
        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}_decoding.png'
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}_decoding.pkl'
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                     'PPA', 'pSTS', 'face-pSTS', 'aSTS']
        print(f'cuda is available {torch.cuda.is_available()}')
        self.channels = None

    def load_eeg(self):
        df_ = pd.read_csv(f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        self.channels = [col for col in df_.columns if 'channel' in col]
        return df_
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata=metadata_,
                        stimulus_data=stimulus_data_,
                        response_data=response_data_)
    
    def assign_stimulus_set(self, df_):
        df_['stimulus_set'] = 'train'
        test_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/test.csv')['video_name'].to_list()
        df_.loc[df_.video_name.isin(test_videos), 'stimulus_set'] = 'test'
        return df_
    
    def preprocess_data(self, df, stimulus_data):
        df_ = df.groupby(['time', 'video_name']).mean(numeric_only=True).reset_index()
        cols_to_drop = set(df_.columns.to_list()) - set(['time', 'video_name'] + self.channels)
        df_.drop(columns=cols_to_drop, inplace=True)
        df_ = df_.loc[df_.video_name.isin(stimulus_data.video_name)]
        df_.sort_values(['time', 'video_name'], inplace=True)
        df_smoothed = temporal.smoothing(df_, self.channels, grouping='video_name')
        return self.assign_stimulus_set(df_smoothed)

    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            results = pd.read_csv(self.out_file)
        else:
            print('loading data...')
            df = self.load_eeg()
            benchmark = self.load_fmri()
            df_avg = self.preprocess_data(df, benchmark.stimulus_data)

            print('beginning decoding...')
            results = decoding.eeg_fmri_decoding(df_avg, benchmark,
                                                  self.channels, self.device)
            results = results.groupby(['time', 'roi_name']).mean().reset_index()
            results.to_pickle(self.out_file)
            print('Finished!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    fMRIDecoding(args).run()


if __name__ == '__main__':
    main()