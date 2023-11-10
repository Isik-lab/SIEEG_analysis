# %%
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns
from itertools import combinations
from src.mri import gen_mask
from src.rsa import fit_and_predict
from src.plotting import feature2color
import nibabel as nib
import warnings
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut


class fMRIRDMs:
    def __init__(self, args):
        self.process = 'fMRIRDMs'
        self.sid = f'sub-{str(args.sid).zfill(2)}'
        self.decoding = args.decoding
        self.data_dir = args.data_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.preproc_files = f'{self.data_dir}/raw/fmri_betas/{self.sid}_space-T1w_desc-*-fracridge-all-data.nii.gz'
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                      'PPA', 'pSTS', 'face-pSTS', 'aSTS']
        self.features = ['alexnet', 'moten', 'indoor',
                        'expanse', 'object_directedness', 'agent_distance',
                        'facingness', 'joint_action', 'communication', 
                        'valence', 'arousal']

    def run(self):
        test_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/test.csv')
        train_videos = pd.read_csv(f'{self.data_dir}/raw/annotations/train.csv')
        df = pd.concat([test_videos, train_videos]).reset_index(drop=True).sort_values(by='video_name')
        sort_idx = df.reset_index()['index'].to_numpy()
        videos = df.video_name.to_numpy()

        feature_rdms = pd.read_csv(f'{self.data_dir}/interim/FeatureRDMs/feature_rdms.csv')

        if self.decoding:
            n_groups = 5 
            groups = np.concatenate([np.arange(n_groups), np.arange(n_groups)])
            logo = LeaveOneGroupOut()
            pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression())])
            out_name = f'{self.data_dir}/interim/{self.process}/{self.sid}_pairwise-decoding.csv'
        else:
            out_name = f'{self.data_dir}/interim/{self.process}/{self.sid}_correlation-distance.csv'

        betas = []
        for file in sorted(glob(self.preproc_files)):
            beta_img = nib.load(file)
            arr = beta_img.get_fdata().reshape((-1, beta_img.shape[-2], beta_img.shape[-1]))
            if arr.shape[-1] > 10:
                arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2] // 2, 2).mean(axis=3)
            betas.append(arr)
        betas = np.hstack(betas)[:, sort_idx,:]

        reliability_mask = np.load(f'{self.data_dir}/raw/reliability_mask/{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.npy')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            
            nCk = list(combinations(range(betas.shape[1]), 2))
            results = []
            for roi in tqdm(self.rois, desc='ROIs'):
                files = glob(f'{self.data_dir}/raw/localizers/{self.sid}/{self.sid}_task-*_space-T1w_roi-{roi}_hemi-*_roi-mask.nii.gz')
                mask = gen_mask(files, reliability_mask)
                betas_masked = betas[mask, ...]
                if self.decoding:
                    result_for_t = Parallel(n_jobs=-1)(
                        delayed(fit_and_predict)(betas_masked[:, video1, :].squeeze().T,
                                                betas_masked[:, video2, :].squeeze().T,
                                                n_groups) for video1, video2 in tqdm(nCk, total=len(nCk), desc='Pairwise decoding')
                    )
                    for accuracy, (video1, video2) in zip(result_for_t, nCk):
                        results.append([roi, videos[video1], videos[video2], accuracy])
                else:
                    rdm = pdist(np.nanmean(betas[mask, ...], axis=-1).T, metric='correlation')
                    for i, (video1, video2) in enumerate(nCk):
                        results.append([roi, videos[video1], videos[video2],
                                        rdm[i]])
        results = pd.DataFrame(results, columns=['roi', 'video1', 'video2', 'distance'])
        results.to_csv(out_name, index=False)

        feature_group = feature_rdms.groupby('feature')
        neural_group = results.groupby('roi')
        rsa = []
        for feature, feature_rdm in tqdm(feature_group):
            for time, time_rdm in neural_group:
                rho, _ = spearmanr(feature_rdm.distance, time_rdm.distance)
                rsa.append([feature, time, rho])
        rsa = pd.DataFrame(rsa, columns=['feature', 'roi', 'Spearman rho'])
        cat_type = pd.CategoricalDtype(categories=self.features, ordered=True)
        rsa['feature'] = rsa.feature.astype(cat_type)
        cat_type = pd.CategoricalDtype(categories=self.rois, ordered=True)
        rsa['roi'] = rsa.roi.astype(cat_type)
        if self.decoding: 
            rsa.to_csv(f'{self.data_dir}/interim/{self.process}/{self.sid}_rsa-decoding.csv')
        else:
            rsa.to_csv(f'{self.data_dir}/interim/{self.process}/{self.sid}_rsa-correlation.csv')
        rsa.head()


        feature_group = rsa.groupby('roi')
        _, axes = plt.subplots(3, 3, sharey=True, sharex=True)
        axes = axes.flatten()
        ymin, ymax = rsa['Spearman rho'].min(), rsa['Spearman rho'].max()
        for ax, (roi, feature_df) in zip(axes, feature_group):
            sns.barplot(x='feature', y='Spearman rho',
                        data=feature_df, ax=ax, color='gray')
            if roi in ['EVC', 'LOC', 'pSTS']:
                ax.set_ylabel('Spearman rho')
            else:
                ax.set_ylabel('')
                
            if roi in ['pSTS', 'face-pSTS', 'aSTS']:
                ax.set_xlabel('Feature')
                ax.set_xticklabels(self.features, rotation=90, ha='center')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', length=0)

            for bar, feature in zip(ax.patches, self.features):
                color = feature2color(feature)
                bar.set_color(color)
                
            ax.set_ylim([ymin, ymax])
            ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                colors='gray', linestyles='solid', zorder=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_title(roi)

        plt.tight_layout()
        if self.decoding: 
            plt.savefig(f'{self.figure_path}/{self.sid}_rsa-decoding.png')
        else:
            plt.savefig(f'{self.figure_path}/{self.sid}_rsa-correlation.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=9)
    parser.add_argument('--decoding', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    args = parser.parse_args()
    fMRIRDMs(args).run()


if __name__ == '__main__':
    main()
