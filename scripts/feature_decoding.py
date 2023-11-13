#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import temporal, plotting, decoding
import os


def preprocess_data(df, channels):
    df_ = df.groupby(['time', 'video_name']).mean(numeric_only=True).reset_index()
    cols_to_drop = set(df.columns.to_list()) - set(['time', 'video_name'] + channels)
    df_.drop(columns=cols_to_drop, inplace=True)
    df_.sort_values(['time', 'video_name'], inplace=True)
    return temporal.smoothing(df_, 'video_name')


class FeatureDecoding:
    def __init__(self, args):
        self.process = 'FeatureDecoding'
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
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}_decoding.csv'
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

    def load_eeg(self):
        df_ = pd.read_csv(f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        all_cols = set(df_.columns.to_list())
        other_cols = set(['trial', 'time', 'offset', 'offset_eyetrack_x', 'video_name',
                    'gaze_x', 'gaze_y', 'pupil_size', 'target_x', 'target_y',
                    'target_distance', 'offset_eyetrack_y', 'repetition', 'even', 'session'])
        channels = list(all_cols - other_cols)
        return df_, channels
    
    def load_features(self):
        df_ = pd.read_csv(f'{self.data_dir}/interim/FeatureRDMs/feature_annotations.csv')
        features_ = df_.set_index('video_name').columns.to_list()
        return df_, features_
    
    def average_results(self, results_):
        temp = results_.pivot(index='time', columns='feature', values='r')
        out = temp[self.annotated_features].reset_index()
        for feature in self.computed_features:
            cols = [col for col in temp.columns if feature in col]
            out[feature] = temp[cols].to_numpy().mean(axis=1)
        out = pd.melt(out, id_vars='time', value_vars=self.features, value_name='r')
        cat_type = pd.CategoricalDtype(categories=self.features, ordered=True)
        out['feature'] = out.feature.astype(cat_type)
        return out
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            results = pd.read_csv(self.out_file)
        else:
            df, channels = self.load_eeg()
            feature_df, predicting_features = self.load_features()
            df_avg = preprocess_data(df, channels)

            results = decoding.eeg_feature_decoding(df_avg, feature_df,
                                                    predicting_features, channels)
            results = self.average_results(results)
            results.to_csv(self.out_file, index=False)
        plotting.plot_eeg_feature_decoding(results, self.features, self.out_figure)


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
    FeatureDecoding(args).run()


if __name__ == '__main__':
    main()