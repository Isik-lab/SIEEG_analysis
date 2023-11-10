# %%
import argparse
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from src.mri import gen_mask
from src import rsa
from src.plotting import plot_feature_fmri_rsa
import nibabel as nib


class fMRIRDMs:
    def __init__(self, args):
        self.process = 'fMRIRDMs'
        self.sid = f'sub-{str(args.sid).zfill(2)}'
        self.decoding = args.decoding
        self.data_dir = args.data_dir
        self.n_groups = 5 
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.preproc_files = f'{self.data_dir}/raw/fmri_betas/{self.sid}_space-T1w_desc-*-fracridge-all-data.nii.gz'
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        if self.decoding: 
            self.out_figure = f'{self.figure_dir}/{self.sid}_rsa-decoding.png'
            self.out_rdm = f'{self.data_dir}/interim/{self.process}/{self.sid}_decoding-distance.csv'
            self.out_rsa = f'{self.data_dir}/interim/{self.process}/{self.sid}_rsa-decoding.csv'
        else:
            self.out_figure = f'{self.figure_dir}/{self.sid}_rsa-correlation.png'
            self.out_rdm = f'{self.data_dir}/interim/{self.process}/{self.sid}_correlation-distance.csv'
            self.out_rsa = f'{self.data_dir}/interim/{self.process}/{self.sid}_rsa-correlation.csv'
        print(vars(self))
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                      'PPA', 'pSTS', 'face-pSTS', 'aSTS']
        self.features = ['alexnet', 'moten', 'indoor',
                        'expanse', 'object_directedness', 'agent_distance',
                        'facingness', 'joint_action', 'communication', 
                        'valence', 'arousal']

    def load_video_order(self):
        test_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/test.csv')
        train_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/train.csv')
        df = pd.concat([test_videos, train_videos]).reset_index(drop=True).sort_values(by='video_name')
        sort_idx = df.reset_index()['index'].to_numpy()
        videos = df.video_name.to_numpy()
        return sort_idx, videos
    
    def load_fmri(self, sort_idx):
        betas = []
        for file in sorted(glob(self.preproc_files)):
            beta_img = nib.load(file)
            arr = beta_img.get_fdata().reshape((-1, beta_img.shape[-2], beta_img.shape[-1]))
            if arr.shape[-1] > 10:
                arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2] // 2, 2).mean(axis=3)
            betas.append(arr)
        return np.hstack(betas)[:, sort_idx,:]
    
    def load_feature_rdms(self):
        return pd.read_csv(f'{self.data_dir}/interim/FeatureRDMs/feature_rdms.csv')

    def save(self, df, out_file):
        df.to_csv(out_file, index=False)

    def roi_masks(self):
        reliability_mask = np.load(f'{self.data_dir}/raw/reliability_mask/{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.npy')
        masks = dict()
        for roi in self.rois:
            files = glob(f'{self.data_dir}/raw/localizers/{self.sid}/{self.sid}_task-*_space-T1w_roi-{roi}_hemi-*_roi-mask.nii.gz')
            masks[roi] = gen_mask(files, reliability_mask)
        return masks

    def run(self):
        sort_idx, videos = self.load_video_order()
        betas = self.load_fmri(sort_idx)
        masks = self.roi_masks()
        nCk = list(combinations(range(betas.shape[1]), 2))
        if self.decoding:
            fmri_rdms = rsa.mri_decoding_distance(betas, masks, nCk, videos, self.n_groups)
        else:
            fmri_rdms = rsa.fmri_correlation_distance(betas, masks, nCk, videos)
        self.save(fmri_rdms, self.out_rdm)

        #rsa
        feature_rdms = self.load_feature_rdms()
        results = rsa.compute_feature_fmri_rsa(feature_rdms, fmri_rdms, self.features, self.rois)
        self.save(results, self.out_rsa)
        plot_feature_fmri_rsa(results, self.features, self.out_figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--decoding', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    fMRIRDMs(args).run()


if __name__ == '__main__':
    main()