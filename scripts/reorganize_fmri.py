#/Applications/anaconda3/envs/nibabel/bin/python
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


class ReorganziefMRI:
    def __init__(self, args):
        self.process = 'ReorganziefMRI'
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                      'PPA', 'pSTS', 'face-pSTS', 'aSTS']

    def generate_benchmark(self):
        rois = ['reliable']
        all_data = []
        all_betas = []
        for sub in tqdm(range(1,5)):
            sub_fill = str(sub).zfill(2)
            sub = str(sub)
            # Beta files
            betas_file = f'{self.data_dir}/raw/fmri_betas/sub-{sub_fill}_space-t1w_desc-train-fracridge_data.nii.gz'
            betas_arr = nib.load(betas_file).get_fdata()

            # metadata
            first_betas = betas_arr[:,:,:,0]
            first_betas = first_betas.astype(str)

            # response data
            all_betas.append(betas_arr.reshape((-1, betas_arr.shape[-1])))

            for roi in self.rois:
                file = f'{self.data_dir}/raw/reliability_mask/sub-{sub_fill}_space-t1w_desc-test-fracridge_reliability-mask.npy'
                roi_mask = np.load(file)
                first_betas[roi_mask] = roi

            sub_data = [(ind, roi, sub) for (ind, roi) in enumerate(first_betas.flatten())]
            all_data.append(sub_data)

        # metadata
        metadata = []
        for i in range(4):
            metadata.append(pd.DataFrame(all_data[i], columns=['voxel_id', 'roi_name', 'subj_id']))
        metadata = pd.concat(metadata, ignore_index=True)
        metadata = metadata[metadata.roi_name.isin(rois)]
        metadata.reset_index(drop=True, inplace=True)

        # response data
        response_data = []
        for i in range(4):
            voxels = metadata.loc[metadata['subj_id']==str(i+1)]['voxel_id'].to_list()
            response_data.append(pd.DataFrame(all_betas[i][voxels,:]).fillna(0))
        response_data = pd.concat(response_data, ignore_index=True)
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
