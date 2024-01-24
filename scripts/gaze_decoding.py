#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import decoding
import numpy as np


class GazeDecoding:
    def __init__(self, args):
        self.process = 'GazeDecoding'
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        print(vars(self))
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_decoding.csv'
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

    def load_gaze(self):
        df = pd.read_csv(f'{self.data_dir}/interim/PreprocessData/{self.sid}_reg-gaze-False.csv.gz')
        avg_df = df.groupby(['video_name', 'time']).mean().reset_index()
        gaze_df = avg_df.loc[(avg_df.time > 0) & (avg_df.time <= .5)]
        return np.hstack([gaze_df.pivot(index='video_name', columns='time', values='gaze_x').to_numpy(),
                          gaze_df.pivot(index='video_name', columns='time', values='gaze_y').to_numpy()])
        
    def load_features(self):
        df_ = pd.read_csv(f'{self.data_dir}/interim/FeatureRDMs/feature_annotations.csv')
        features_ = df_.set_index('video_name').columns.to_list()
        return df_, features_
    
    def average_results(self, results_):
        temp = results_.pivot(index='subj', columns='feature', values='r')
        out = temp[self.annotated_features].reset_index()
        for feature in self.computed_features:
            cols = [col for col in temp.columns if feature in col]
            out[feature] = temp[cols].to_numpy().mean(axis=1)
        out = pd.melt(out, id_vars='subj', value_vars=self.features, value_name='r')
        return out
    
    def run(self):
        X = self.load_gaze()
        feature_df, predicting_features = self.load_features()
        results = decoding.gaze_feature_decoding(X, feature_df,
                                                    predicting_features)
        results['subj'] = self.sid
        results = self.average_results(results)
        results.to_csv(self.out_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    args = parser.parse_args()
    GazeDecoding(args).run()


if __name__ == '__main__':
    main()