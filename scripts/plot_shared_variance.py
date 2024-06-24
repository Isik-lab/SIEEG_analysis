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
from src.plotting import trim_axs
from tqdm import tqdm


def feature_to_X(feature):
    if feature is None:
        return None
    else: 
        d = {'alexnet': 'moten-behavior',
            'moten': 'alexnet-behavior',
            'behavior': 'alexnet-moten'}
        return d[feature]


def load_eeg(file):
    out = pd.read_csv(file)
    out = out.groupby(['time', 'targets']).mean(numeric_only=True).reset_index()
    if 'r2' not in out.columns:
        out['r2'] = stats.sign_square(out['r'].to_numpy())
    return out.sort_values(by=['time', 'targets'])


def load_fmri(file, fmri_info):
    out = pd.read_csv(file).rename(columns={'0': 'r'})
    out['targets'] = fmri_info.roi_name.to_list()
    if 'voxel_id' in fmri_info.columns:
        out['voxel_id'] = fmri_info.voxel_id.to_list()
    else:
        out['subj_id'] = fmri_info.subj_id.to_list()
    out = out.groupby('targets').mean(numeric_only=True).reset_index()
    out['r2'] = stats.sign_square(out['r'].to_numpy())
    return out.sort_values(by='targets')


class PlotSharedVariance:
    def __init__(self, args):
        self.process = 'PlotSharedVariance'
        # logging.neptune_init(self.process)
        self.fmri_dir = args.fmri_dir
        self.fmri_encoding = args.fmri_encoding
        self.eeg_decoding_summary = args.eeg_decoding_summary
        self.out_dir = args.out_dir
        self.roi_mean = args.roi_mean
        self.unique_feature = args.unique_feature
        self.feature_ids = feature_to_X(self.unique_feature)
        # logging.neptune_params(self)
        print(vars(self))

    def load_fmri_encoding(self):
        _, fmri_info = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        r2_annot_full = load_fmri(f'{self.fmri_encoding}/x-alexnet-moten-behavior_y-fmri_scores.csv.gz', fmri_info)
        if self.unique_feature is None: 
            return r2_annot_full, None
        else:
            r2_annot_drop = load_fmri(f'{self.fmri_encoding}/x-{self.feature_ids}_y-fmri_scores.csv.gz', fmri_info)
            return r2_annot_full, r2_annot_drop

    def load_eeg_decoding(self):
        r2s_annot_eeg_full = load_eeg(f'{self.eeg_decoding_summary}/x-eeg-alexnet-moten-behavior_y-fmri.csv.gz')
        if self.unique_feature is None: 
            r2s_eeg = load_eeg(f'{self.eeg_decoding_summary}/x-eeg_y-fmri.csv.gz')
            return r2s_eeg, r2s_annot_eeg_full
        else:
            r2s_annot_eeg_drop = load_eeg(f'{self.eeg_decoding_summary}/x-eeg-{self.feature_ids}_y-fmri.csv.gz')
            return r2s_annot_eeg_full, r2s_annot_eeg_drop

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

    def compute_unique_shared_variance(self, r2_annot_full, r2_annot_drop,
                                       r2s_annot_eeg_full, r2s_annot_eeg_drop):
        r2_annot_full.set_index('targets', inplace=True)
        r2_annot_drop.set_index('targets', inplace=True)
        out = []
        time_iter = zip(r2s_annot_eeg_full.groupby('time'), r2s_annot_eeg_drop.groupby('time'))
        for (time, r2_annot_eeg_full), (_, r2_annot_eeg_drop) in tqdm(time_iter, desc='Computing shared variance through time'):
            r2_annot_eeg_full.set_index('targets', inplace=True)
            r2_annot_eeg_drop.set_index('targets', inplace=True)
            r2_unique_shared = r2_annot_full['r2'] - r2_annot_eeg_full['r2'] - r2_annot_drop['r2'] + r2_annot_eeg_drop['r2']
            r2_unique_shared = pd.DataFrame(r2_unique_shared).reset_index()
            r2_unique_shared['time'] = time
            out.append(r2_unique_shared)
        return pd.concat(out, ignore_index=True).reset_index(drop=True)

    def viz_results(self, results):
        time_labels = [0, .5]
        n_targets = results.targets.nunique()
        fig, axes = plt.subplots(3, int(np.ceil(n_targets/3)),
                                 sharex=True, sharey=True)
        axes = trim_axs(axes.flatten(), n_targets)
        ymin, ymax = results.r2.min() + 0.01, results.r2.max() + 0.01
        for (target, df), ax in zip(results.groupby('targets'), axes):
            df.sort_values(by='time', inplace=True)
            time = df.time.to_numpy()
            time_inds = np.arange(0, len(time))
            r2 = df.r2.to_numpy()
            # r2_adjusted = r2 - r2[time < 0].mean()
            ax.plot(time_inds, r2, color='black', zorder=1)
            # ax.plot(time_inds, r2_adjusted, color='green', zorder=1)
            ax.set_title(target)
            ax.set_xlim([0, len(time)])
            tick_inds = [(np.abs(time - t)).argmin() for t in time_labels]
            ax.vlines(x=tick_inds, ymin=ymin,
                      ymax=ymax, linestyles='dashed',
                      color='gray', zorder=0, linewidth=1)
            ax.hlines(y=0, xmin=0, xmax=len(time),
                      color='black', zorder=0, linewidth=1)
            ax.set_xticks(tick_inds, time_labels)
            ax.set_ylim([ymin, ymax])
            ax.set_xticklabels(time_labels)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Shared variance ($r^2$)', fontsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        plt.tight_layout()
        if self.unique_feature is None:
            plt.savefig(f'{self.out_dir}/shared_variance.{get_githash()}.pdf')
        else:
            plt.savefig(f'{self.out_dir}/unique-shared-{self.unique_feature}.{get_githash()}.pdf')

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def save_results(self, results):
        if self.unique_feature is None:
            results.to_csv(f'{self.out_dir}/shared_variance.csv.gz', index=False)
        else:
            results.to_csv(f'{self.out_dir}/unique-shared-{self.unique_feature}.csv.gz', index=False)

    def run(self):
        # load data
        if self.unique_feature is None:
            r2_behavior, _ = self.load_fmri_encoding()
            r2s_eeg, r2s_eeg_behavior = self.load_eeg_decoding()
            data = self.compute_shared_variance(r2_behavior, r2s_eeg, r2s_eeg_behavior)
        else:
            r2_annot_full, r2_annot_drop = self.load_fmri_encoding()
            r2s_annot_eeg_full, r2s_annot_eeg_drop = self.load_eeg_decoding()
            data = self.compute_unique_shared_variance(r2_annot_full, r2_annot_drop,
                                                       r2s_annot_eeg_full, r2s_annot_eeg_drop)
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
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predicted roi mean response instead of voxelwise responses')
    parser.add_argument('--unique_feature', '-u', type=str, default=None,
                        help='The feature category that uniquely shares variance with EEG\n Default is none and will plot the shared variance between all features and EEG.')                    
    args = parser.parse_args()
    PlotSharedVariance(args).run()


if __name__ == '__main__':
    main()