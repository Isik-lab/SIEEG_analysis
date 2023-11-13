#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import rsa, plotting, temporal
import numpy as np
from glob import glob


class EEG_RSA:
    def __init__(self, args):
        self.process = 'EEG_RSA'
        if 'u' not in args.sid:
            self.sid = f'subj{str(int(args.sid)).zfill(3)}'
        else:
            self.sid = args.sid
        self.regress_gaze = args.regress_gaze
        self.eeg_metric = args.eeg_metric
        self.target = args.target
        self.fmri_metric = args.fmri_metric
        print(vars(self))

        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        if self.target == 'features':
            self.target_file = f'{self.data_dir}/interim/FeatureRDMs/feature_rdms.csv'
            self.target_categories = ['alexnet', 'moten', 'indoor',
                                      'expanse', 'object_directedness', 'agent_distance',
                                      'facingness', 'joint_action', 'communication',
                                      'valence', 'arousal']
            self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_feature_EEG-{self.eeg_metric}_reg-gaze-{self.regress_gaze}_rsa.csv'
            self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_feature_EEG-{self.eeg_metric}_reg-gaze-{self.regress_gaze}_rsa.png'
        else:
            self.target_file = f'{self.data_dir}/interim/fMRI_RSA/*{self.fmri_metric}-distance.csv'
            self.target_categories = ['EVC', 'MT', 'EBA',
                                      'LOC', 'FFA', 'PPA',
                                      'pSTS', 'face-pSTS', 'aSTS']
            self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_fMRI-{self.fmri_metric}_EEG-{self.eeg_metric}_reg-gaze-{self.regress_gaze}_rsa.csv'
            self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_fMRI-{self.fmri_metric}_EEG-{self.eeg_metric}_reg-gaze-{self.regress_gaze}_rsa.png'

    def load_eeg(self):
        if self.eeg_metric == 'correlation':
            eeg_rdms = pd.read_csv(f'{self.data_dir}/interim/CorrelationDistance/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        else:
            eeg_rdms = pd.read_csv(f'{self.data_dir}/interim/PairwiseDecoding/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz')
        eeg_rdms = temporal.smoothing(eeg_rdms)
        return eeg_rdms
    
    def load_features(self):
        return pd.read_csv(self.target_file)
    
    def load_fmri(self):
        fmri_rdms = []
        for file in glob(self.target_file):
            fr = pd.read_csv(file)
            fmri_subj_name = file.split('/')[-1].split('_')[0]
            fr['subj'] = fmri_subj_name
            fmri_rdms.append(fr)
        fmri_rdms = pd.concat(fmri_rdms)
        fmri_rdms = fmri_rdms.groupby(['roi', 'video1', 'video2']).mean(numeric_only=True).reset_index()
        fmri_rdms.sort_values(by=['roi', 'video1', 'video2'], inplace=True)
        return fmri_rdms
    
    def filter_df(self):
        eeg_ = self.load_eeg()
        if self.target == 'features':
            target_ = self.load_features()
        else:
            target_ = self.load_fmri()

        target_videos = np.unique(np.concatenate([target_.video1.to_numpy(),
                                                 target_.video2.to_numpy()]))
        eeg_rdms = rsa.filter_pairs(eeg_, target_videos)
        eeg_videos = np.unique(np.concatenate([eeg_rdms.video1.to_numpy(),
                                                eeg_rdms.video2.to_numpy()]))
        target_rdms = rsa.filter_pairs(target_, eeg_videos)
        return eeg_rdms, target_rdms

    def run(self):
        eeg_rdms, target_rdms = self.filter_df()

        if self.target == 'features': 
            results = rsa.compute_eeg_feature_rsa(target_rdms, eeg_rdms, self.target_categories)
            plotting.plot_eeg_feature_rsa(results, self.target_categories, self.out_figure)
        else:
            results = rsa.compute_eeg_fmri_rsa(target_rdms, eeg_rdms, self.target_categories)
            plotting.plot_eeg_fmri_rsa(results, self.target_categories, self.out_figure)
        results.to_csv(self.out_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--eeg_metric', type=str, default='decoding')
    parser.add_argument('--target', type=str, default='features')
    parser.add_argument('--fmri_metric', type=str, default='decoding')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    EEG_RSA(args).run()


if __name__ == '__main__':
    main()
