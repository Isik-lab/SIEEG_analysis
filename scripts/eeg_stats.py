import argparse
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from src.stats import perm, bootstrap, sign_square, corr2d, calculate_p, cluster_correction
from src import loading
import json
from glob import glob
from tqdm import tqdm


class eegStats:
    def __init__(self, args):
        self.process = 'eegStats'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.roi_mean = args.roi_mean
        self.n_perm = args.n_perm
        self.pred_file_pattern = args.pred_file_pattern
        self.out_dir = args.out_dir
        sub = self.pred_file_pattern.split("/")[-1].split("*")[0]
        x_y_names = self.pred_file_pattern.split("/")[-1].split("*_")[-1].split("_yhat")[0]
        print(f'{x_y_names=}')
        self.prefix = f'{self.out_dir}/{sub}_{x_y_names}'
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir

    @staticmethod
    def compute_score(true, pred):
        return np.expand_dims(sign_square(corr2d(true, pred)), axis=0)
    
    def compute_perm_dist(self, true, pred):
        return np.expand_dims(perm(true, pred, n_perm=self.n_perm, square=True), axis=0)
    
    def compute_var_dist(self, true, pred):
        return np.expand_dims(bootstrap(true, pred, n_perm=self.n_perm, square=True), axis=0)

    def load_pred(self, file):
        return pd.read_csv(file).to_numpy()

    def load_true(self):
        true, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        behavior = loading.load_behavior(self.fmri_dir)
        idx = behavior.loc[behavior.stimulus_set == 'test'].index
        return true.iloc[idx].to_numpy()

    def compute_dists(self, true):
        r2, null, var = [], [], []
        files = sorted(glob(self.pred_file_pattern))
        for file in tqdm(files, total=len(files), desc='Computing dists through time'):
            pred = self.load_pred(file)
            r2.append(self.compute_score(true, pred))
            null.append(self.compute_perm_dist(true, pred))
            var.append(self.compute_var_dist(true, pred))
        return np.concatenate(r2, axis=0), np.concatenate(null, axis=0), np.concatenate(var, axis=0)

    def save_array(self, arr, suffix):
        np.save(f'{self.prefix}_{suffix}.npy', arr)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        true = self.load_true()
        r2, null, var = self.compute_dists(true)
        self.mk_out_dir()
        self.save_array(r2, 'r2')
        self.save_array(null, 'null')
        self.save_array(var, 'var')
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--pred_file_pattern', '-p', type=str, help='prediction file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/encodeDecode/eeg/sub-03_time-*_x-eeg_y-fmri_yhat.csv.gz')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegStats')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--n_perm', type=int, default=int(5e3),
                        help='number of permutations/resamples to run')                   
    args = parser.parse_args()
    eegStats(args).run()


if __name__ == '__main__':
    main()