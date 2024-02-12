#/home/emcmaho7/.conda/envs/eeg/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import temporal, plotting, decoding
import os


class FeatureDecoding:
    def __init__(self, args):
        self.process = 'FeatureDecoding'
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
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
        self.features = ['alexnet', 'moten',
                         'indoor', 'expanse', 'object_directedness', 
                         'agent_distance', 'facingness',
                          'joint_action', 'communication',
                         'valence', 'arousal']
        self.annotated_features = ['indoor', 'expanse', 'object_directedness',
                                   'agent_distance', 'facingness',
                                   'joint_action', 'communication',
                                   'valence', 'arousal']
        self.computed_features = ['alexnet', 'moten']
        self.channels = None

    def load_eeg(self):
        eeg_data_file = f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz'
        print(f'{eeg_data_file=}')
        df_ = pd.read_csv(eeg_data_file)
        self.channels = [col for col in df_.columns if 'channel' in col]
        return df_
    
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
    
    def load_features(self):
        df_ = pd.read_csv(f'{self.data_dir}/interim/FeatureRDMs/feature_annotations.csv')
        features_ = df_.set_index('video_name').columns.to_list()
        return self.assign_stimulus_set(df_), features_
    
    def run(self):
        results = None
        if os.path.exists(self.out_file) and not self.overwrite: 
            results = pd.read_csv(self.out_file)
            cat_type = pd.CategoricalDtype(categories=self.features, ordered=True)
            results['feature'] = results.feature.astype(cat_type)
        else:
            df = self.load_eeg()
            feature_df, predicting_features = self.load_features()
            df_avg = self.preprocess_data(df)

            if 'train' in df_avg['stimulus_set'].unique():
                results = decoding.eeg_feature_decoding(df_avg, feature_df,
                                                        predicting_features, self.channels)
                print(f'{results.head()=}')
                print(f'{results.iloc[0]['r_null'].shape=}')
                print(f'{results.iloc[0]['r_var'].shape=}')
                results.to_pickle(self.out_file)
            else:
                print('encoding not performed, no training set')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    FeatureDecoding(args).run()


if __name__ == '__main__':
    main()