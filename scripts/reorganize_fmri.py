#/Applications/anaconda3/envs/nibabel/bin/python
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from src.mri import gen_mask
from glob import glob


class ReorganziefMRI:
    def __init__(self, args):
        self.process = 'ReorganziefMRI'
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                     'PPA', 'pSTS', 'face-pSTS', 'aSTS']

    def generate_benchmark(self):
        all_rois = []
        all_betas = []
        for sub in tqdm(range(1,5)):
            sub = str(sub).zfill(2)
            reliability_mask = np.load(f'{self.data_dir}/raw/reliability_mask/sub-{sub}_space-t1w_desc-test-fracridge_reliability-mask.npy').astype('bool')

            # Beta files
            betas_file = f'{self.data_dir}/raw/fmri_betas/sub-{sub}_space-t1w_desc-train-fracridge_data.nii.gz'
            betas_arr = nib.load(betas_file).get_fdata()

            # metadata
            beta_labels = betas_arr[:,:,:,0]
            beta_labels = beta_labels.astype(str)
            reliability_mask = reliability_mask.reshape(beta_labels.shape)

            # Add the roi labels
            for roi in self.rois:
                files = sorted(glob(f'{self.data_dir}/raw/localizers/sub-{sub}/*roi-{roi}*.nii.gz'))
                roi_mask = gen_mask(files, reliability_mask)
                beta_labels[roi_mask] = roi

            # Only save the reliable voxels
            betas_arr = betas_arr[reliability_mask].reshape((-1, betas_arr.shape[-1]))
            beta_labels = beta_labels[reliability_mask].flatten()

            # Add the subject data to list
            all_betas.append(betas_arr)
            all_rois.append([(ind, roi, sub) for (ind, roi) in enumerate(beta_labels)])

        # metadata
        metadata = []
        for i in range(4):
            metadata.append(pd.DataFrame(all_rois[i], columns=['voxel_id', 'roi_name', 'subj_id']))
        metadata = pd.concat(metadata, ignore_index=True)
        metadata = metadata[metadata.roi_name.isin(self.rois)]
        metadata.reset_index(drop=True, inplace=True)

        # response data
        response_data = []
        for i in range(4):
            sub = str(i+1).zfill(2)
            voxels = metadata.loc[metadata['subj_id']==sub, 'voxel_id'].to_numpy()
            df = pd.DataFrame(all_betas[i][voxels,:])
            response_data.append(df)
        response_data = pd.concat(response_data, ignore_index=True)

        # Make the voxel ids unique so that there are no repeats across subjects
        metadata = metadata.drop(columns='voxel_id').reset_index().rename(columns={'index': 'voxel_id'})
        return metadata, response_data
    
    def run(self):
        metadata, response_data = self.generate_benchmark()
        metadata.to_csv(f'{self.data_dir}/interim/{self.process}/metadata.csv', index=False)
        response_data.to_csv(f'{self.data_dir}/interim/{self.process}/response_data.csv.gz', index=False, compression='gzip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    args = parser.parse_args()
    ReorganziefMRI(args).run()


if __name__ == '__main__':
    main()
