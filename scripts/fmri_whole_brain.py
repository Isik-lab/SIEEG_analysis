#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np


class fMRIWholeBrain:
    def __init__(self, args):
        self.process = 'fMRIWholeBrain'
        self.data_dir = f'{args.top_dir}/data'
        self.out_dir = f'{self.data_dir}/interim/{self.process}'
        self.n_eeg_samples = 480
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def load_files(self):
        df = None
        for time_ind in tqdm(range(0, self.n_eeg_samples, 4), desc='Time loop', leave=True):
            files = glob(f'{self.data_dir}/interim/fMRIRegression/sub-*_full-brain_time-{str(time_ind).zfill(3)}.parquet')
            time_df = []
            for file in files:
                eeg_subj = file.split('/')[-1].split('_')[0]
                indiv_df = pd.read_parquet(file).reset_index()
                indiv_df['eeg_subj'] = eeg_subj
                time_df.append(indiv_df)
            time_df = pd.concat(time_df, ignore_index=True).reset_index(drop=True)
            time_df = time_df.groupby('voxel_id').mean(numeric_only=True).reset_index()
            time_df.rename(columns={'value': f'time-{str(time_ind).zfill(3)}'}, inplace=True)
            if df is None:
                df = time_df.copy()
            else:
                df = df.merge(time_df, on='voxel_id')
        return df 

    def save_df(self, df):
        df.to_parquet(f'{self.out_dir}/whole_brain.parquet')

    def load_df(self):
        return pd.read_parquet(f'{self.out_dir}/whole_brain.parquet')

    def add_metadata(self, df):
        meta = pd.read_csv(f'{self.data_dir}/interim/ReorganizefMRI/metadata.csv')
        return df.merge(meta, on='voxel_id')

    def generate_brains_and_save(self, df):
        time_cols = [col for col in df.columns if 'time-' in col]
        for fmri_subj, subj_df in tqdm(df.groupby('subj_id'), desc='fMRI subj loop'):
            fmri_subj_str = str(fmri_subj).zfill(2)
            reliability_img = nib.load(f'{self.data_dir}/raw/reliability_mask/sub-{fmri_subj_str}_space-T1w_desc-test-fracridge_stat-r_statmap.nii.gz')
            indices = None
            for time_col in tqdm(time_cols, desc='Time loop'):
                if indices is None:
                    indices = {}
                    for index_col in ['i_index', 'j_index', 'k_index']: 
                        indices[index_col.split('_')[0]] = subj_df[index_col].to_numpy().astype('int')

                result_img = np.zeros(reliability_img.shape)
                result_img[indices['i'], indices['j'], indices['k']] = subj_df[time_col].to_numpy()
                result_img = nib.Nifti1Image(result_img, affine=reliability_img.affine)
                nib.save(result_img, f'{self.out_dir}/sub-{fmri_subj_str}_{time_col}.nii.gz')

    def run(self):
        if not os.path.exists(f'{self.out_dir}/whole_brain.parquet'):
            df = self.load_files()
            self.save_df(df)
        else:
            df = self.load_df()
        df = self.add_metadata(df)
        self.generate_brains_and_save(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', '-top', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis')
    args = parser.parse_args()
    fMRIWholeBrain(args).run()


if __name__ == '__main__':
    main()
