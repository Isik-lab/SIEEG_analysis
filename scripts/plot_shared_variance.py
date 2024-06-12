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
        out = pd.read_csv(f'{self.fmri_encoding}/scores.csv.gz').rename(columns={'0': 'r'})
        out['voxel_id'] = fmri_info.voxel_id.to_list()
        out['targets'] = fmri_info.roi_name.to_list()
        out = out.groupby('targets').mean(numeric_only=True).reset_index()
        out['r2'] = stats.sign_square(out['r'].to_numpy())
        return out.sort_values(by='targets')

    def load_eeg_decoding(self, x_name):
        out = pd.read_csv(f'{self.eeg_decoding_summary}/x-{x_name}_y-fmri.csv.gz')
        out = out.groupby(['time', 'targets']).mean(numeric_only=True).reset_index()
        if 'r2' not in out.columns:
            out['r2'] = stats.sign_square(out['r'].to_numpy())
        return out.sort_values(by=['time', 'targets'])

    def compute_shared_variance(self, r2_behavior, r2s_eeg, r2s_eeg_behavior):
        r2_behavior.set_index('targets', inplace=True)
        out = []
        time_iter = zip(r2s_eeg.groupby('time'), r2s_eeg_behavior.groupby('time'))
        for (time, r2_eeg), (_, r2_eeg_behavior) in tqdm(time_iter, desc='Computing shared variance through time'):
            r2_eeg.set_index('targets', inplace=True)
            r2_eeg_behavior.set_index('targets', inplace=True)
            r2_shared = (r2_behavior['r2'] + r2_eeg['r2']) - r2_eeg_behavior['r2']
            r2_shared = pd.DataFrame(r2_shared).reset_index()
            r2_shared['time'] = time
            out.append(r2_shared)
        return pd.concat(out, ignore_index=True).reset_index(drop=True)

    def viz_results(self, results):
        fig, ax = plt.subplots()
        sns.lineplot(x='time', y='r2', hue='targets', data=results, ax=ax)
        plt.savefig(f'{self.out_dir}/shared_variance.{get_githash()}.pdf')

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def save_results(self, results):
        results.to_csv(f'{self.out_dir}/shared_variance.csv.gz', index=False)

    def run(self):
        # load data
        r2_behavior = self.load_fmri_encoding()
        r2s_eeg = self.load_eeg_decoding('eeg')
        r2s_eeg_behavior = self.load_eeg_decoding('eeg_behavior')

        data = self.compute_shared_variance(r2_behavior, r2s_eeg, r2s_eeg_behavior)
        print(data.head())

        self.mk_out_dir()
        self.viz_results(data)
        self.save_results(data)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--fmri_encoding', '-e', type=str, help='fMRI encoding results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fmriEncoding')
    parser.add_argument('--eeg_decoding_summary', '-d', type=str, help='eeg decoding summary results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotDecoding')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotSharedVariance')
    args = parser.parse_args()
    PlotSharedVariance(args).run()


if __name__ == '__main__':
    main()