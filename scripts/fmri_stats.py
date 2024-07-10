import argparse
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from src.stats import perm, bootstrap
from src import loading
import json


class fmriStats:
    def __init__(self, args):
        self.process = 'fmriStats'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.roi_mean = args.roi_mean
        print(vars(self)) 
        self.pred_file = args.pred_file
        self.out_dir = args.out_dir
        self.fmri_dir = args.fmri_dir
        self.prefix = f'{self.out_dir}/{self.pred_file.split("/")[-1].split("_yhat")[0]}'

    @staticmethod
    def compute_perm_dist(true, pred):
        return perm(true, pred, n_perm=int(5e3), square=True, verbose=True)
    
    @staticmethod
    def compute_var_dist(true, pred):
        return bootstrap(true, pred, n_perm=int(5e3), square=True, verbose=True)

    def load_pred(self):
        return pd.read_csv(self.pred_file).to_numpy()

    def load_true(self):
        true, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        behavior = loading.load_behavior(self.fmri_dir)
        idx = behavior.loc[behavior.stimulus_set == 'test'].index
        return true.iloc[idx].to_numpy()

    def save_results(self, arr, suffix):
        np.save(f'{self.prefix}_{suffix}.npy', arr)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        true = self.load_true()
        pred = self.load_pred()
        self.mk_out_dir()
        self.save_results(self.compute_perm_dist(true, pred), 'null')
        self.save_results(self.compute_var_dist(true, pred), 'var')
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--pred_file', '-p', type=str, help='prediction file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/encodeDecode/eeg/sub-01_time-000_x-eeg_y-fmri_yhat.csv.gz')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/nullStats')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    args = parser.parse_args()
    fmriStats(args).run()


if __name__ == '__main__':
    main()