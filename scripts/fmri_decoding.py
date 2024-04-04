#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import plotting, temporal, decoding
from src.mri import Benchmark
import os
import torch


def check_videos(df_, benchmark):
    """
        Filter the videos in the benchmark class based on those that are present in the EEG data. 
        Data may be missing from a particular video following data cleaning. 
    """
    eeg_videos = df_.video_name.unique()
    benchmark.filter_stimulus(stimulus_set=eeg_videos, col='video_name')


class fMRIDecoding:
    def __init__(self, args):
        self.process = 'fMRIDecoding'
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        self.regress_gaze = args.regress_gaze
        self.overwrite = args.overwrite
        self.shared_variance = args.shared_variance
        self.save_whole_brain = args.save_whole_brain
        self.device = 'cuda:0'
        print(vars(self))
        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file_prefix = f'{self.data_dir}/interim/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}'
        self.channels = None
        self.feature_categories = {'scene_object': ['rating-indoor', 'rating-expanse', 'rating-object'],
                                   'social_primitive': ['rating-agent_distance', 'rating-facingness'],
                                   'social': ['rating-joint_action', 'rating-communication'],
                                   'affective': ['rating-valence', 'rating-arousal']}

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
    
    def preprocess_data(self, df):
        df_ = df.groupby(['time', 'video_name']).mean(numeric_only=True)
        df_ = df_[self.channels].reset_index()
        df_.sort_values(['time', 'video_name'], inplace=True)
        df_smoothed = temporal.smoothing(df_, self.channels, grouping=['video_name'])
        return self.assign_stimulus_set(df_smoothed)

    def run(self):
        print('loading data...')
        df = self.load_eeg()
        benchmark = self.load_fmri()
        df_avg = self.preprocess_data(df)
        print(df_avg.head())
        print(df_avg.columns)

        # Filter so that stimuli in EEG and benchmark class are the same
        # Print sizes when done 
        check_videos(df_avg, benchmark)

        print('beginning decoding...')
        decoding.eeg_fmri_decoding(df_avg, benchmark, self.sid, 
                                    self.channels, self.feature_categories,
                                    self.device, self.out_file_prefix,
                                    save_whole_brain=self.save_whole_brain,
                                    run_shared_variance=self.shared_variance)
        print('Finished!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_whole_brain', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--shared_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    fMRIDecoding(args).run()


if __name__ == '__main__':
    main()
