import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats, logging
import torch
from pathlib import Path
import numpy as np
import seaborn as sns
import pickle
from src.logging import get_githash
from tqdm import tqdm


class PlotEncoding:
    def __init__(self, args):
        self.process = 'PlotEncoding'
        # logging.neptune_init(self.process)
        self.fmri_dir = args.fmri_dir
        self.fmri_encoding = args.fmri_encoding
        self.out_dir = args.out_dir
        self.roi_mean = args.roi_mean
        # logging.neptune_params(self)
        print(vars(self))

    def load_fmri_encoding(self):
        _, fmri_info = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        out = pd.read_csv(f'{self.fmri_encoding}/scores.csv.gz').rename(columns={'0': 'r'})
        out['targets'] = fmri_info.roi_name.to_list()
        if 'voxel_id' in fmri_info.columns:
            out['voxel_id'] = fmri_info.voxel_id.to_list()
        else:
            out['subj_id'] = fmri_info.subj_id.to_list()
        out = out.groupby('targets').mean(numeric_only=True).reset_index()
        out['r2'] = stats.sign_square(out['r'].to_numpy())
        return out.sort_values(by='targets')

    def viz_results(self, results):
        _, ax = plt.subplots()
        sns.barplot(x='targets', y='r2', data=results, ax=ax)
        plt.savefig(f'{self.out_dir}/roi_encoding.{get_githash()}.pdf')

    def save_results(self, results):
        results.to_csv(f'{self.out_dir}/roi_encoding.csv.gz', index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        data = self.load_fmri_encoding()
        self.mk_out_dir()
        self.viz_results(data)
        self.save_results(data)


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--fmri_encoding', '-e', type=str, help='directory of the decoding results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fmriEncoding')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotEncoding')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predicted roi mean response instead of voxelwise responses')
    args = parser.parse_args()
    PlotEncoding(args).run()


if __name__ == '__main__':
    main()