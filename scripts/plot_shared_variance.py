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


class PlotSharedVariance:
    def __init__(self, args):
        self.process = 'PlotSharedVariance'
        # logging.neptune_init(self.process)
        self.fmri_dir = args.fmri_dir
        self.fmri_encoding = args.fmri_encoding
        self.eeg_decoding_summary = args.eeg_decoding_summary
        self.out_dir = args.out_dir
        # logging.neptune_params(self)
        print(vars(self))

    def load_fmri_encoding(self):
        _, fmri_info = loading.load_fmri(self.fmri_dir)
        out = self.fmri_encoding(f'{fmri_encoding}/scores.csv.gz').rename(columns={'0': 'r'})
        out['voxel_id'] = fmri_info.voxel_id.to_list()
        out['targets'] = y_data.roi_name.to_list()
        out = out.groupby('targets').mean(numeric_only=True).reset_index()
        out['r2'] = stats.sign_square(out['r'].to_numpy())
        return out

    def load_eeg_decoding(self, x_name):
        return pd.read_csv(f'{eeg_decoding_summary}/x-{x_name}_y-fmri.csv.gz')

    def compute_shared_variance(self, r2_a, r2_b, r2_ab):

        r2_a + r2_b

    def viz_results(self, results):
        fig, ax = plt.subplots()
        sns.lineplot(x='time', y='r2', hue='targets', data=df_mean, ax=ax)
        plt.savefig(f'{self.out_dir}/shared_variance.{get_githash()}.pdf')

    def run(self):
        # load data
        r2_behavior = self.load_fmri_encoding()
        r2_eeg_behavior = self.load_eeg_decoding('eeg_behavior')
        r2_eeg = self.load_eeg_decoding('eeg')

        data = self.compute_shared_variance(targets)
        print(data.head())
        
        self.mk_out_dir()
        self.viz_results(data)
        self.save_results(data)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory')
    parser.add_argument('--fmri_encoding', '-f', type=str, help='fMRI encoding results')
    parser.add_argument('--eeg_decoding_summary', '-d', type=str, help='eeg decoding summary results')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs')
    args = parser.parse_args()
    PlotSharedVariance(args).run()


if __name__ == '__main__':
    main()