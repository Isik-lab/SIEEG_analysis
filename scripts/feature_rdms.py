import pandas as pd
from pathlib import Path
import argparse
from src import rsa


class FeatureRDMs:
    def __init__(self, args):
        self.process = 'FeatureRDMs'
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.features_to_exclude = ['dominance', 'intimacy', 'cooperation']
        self.features = ['alexnet', 'moten', 'indoor',
                 'expanse', 'object_directedness', 'agent_distance',
                 'facingness', 'joint_action', 'communication', 
                 'valence', 'arousal']

    def load_annotations(self):
        feature_df = pd.read_csv(f'{self.data_dir}/raw/annotations/annotations.csv')
        feature_df = feature_df.drop(columns=self.features_to_exclude)
        rename = {col: col.replace(' ', '_') for col in feature_df.columns}
        rename['transitivity'] = 'object_directedness'
        feature_df.rename(columns=rename, inplace=True)
        feature_df.sort_index(inplace=True)

        alexnet = pd.read_csv(f'{self.data_dir}/interim/ActivationPCA/alexnet_PCs.csv').drop(columns=['split'])
        feature_df = feature_df.merge(alexnet, on='video_name')

        moten = pd.read_csv(f'{self.data_dir}/interim/ActivationPCA/moten_PCs.csv').drop(columns=['split'])
        feature_df = feature_df.merge(moten, on='video_name')
        return feature_df
    
    def save(self, rdms):
        rdms.to_csv(f'{self.data_dir}/interim/{self.process}/feature_rdms.csv', index=False)

    def run(self):
        df = self.load_annotations()
        rdms = rsa.feature_distance(df, self.features)
        self.save(rdms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    args = parser.parse_args()
    FeatureRDMs(args).run()


if __name__ == '__main__':
    main()