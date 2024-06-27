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
from glob import glob
import matplotlib.ticker as mticker


def find_missing(original, new):
    missing = set(original) - set(new)
    return missing.pop() if missing else 'all'


def extract_elements(filename):
    file = filename.split('/')[-1]
    prefix = file.split('_')[0]
    elements = prefix.split('-')[1:]  # [1:] skips the initial 'x'
    return elements


def cat2color(key=None, light_gray=False):
    d = dict()
    if light_gray:
        d['alexnet'] = np.array([0.7, 0.7, 0.7, 0.8])
        d['moten'] = np.array([0.6953125, 0.7421875, 1., 0.8])
    else:
        d['alexnet'] = np.array([0.1953125, 0.1953125, 0.1953125, 0.8])
        d['moten'] = np.array([0.1953125, 0.2421875, 0.5, 0.8])
    d['scene'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['primitive'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['social'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['affective'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    if key is not None:
        return d[key]
    else:
        return d


class PlotfMRIVariance:
    def __init__(self, args):
        self.process = 'PlotfMRIVariance'
        # logging.neptune_init(self.process)
        self.fmri_dir = args.fmri_dir
        self.fmri_encoding = args.fmri_encoding
        self.out_dir = args.out_dir
        self.roi_mean = args.roi_mean
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS', 'face-pSTS', 'aSTS']
        self.categories = ['alexnet', 'moten', 'scene', 'primitive', 'social', 'affective']
        # logging.neptune_params(self)
        print(vars(self))

    def load_fmri_encoding(self):
        _, fmri_info = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        files = glob(f'{self.fmri_encoding}/x-*_scores.csv.gz')
        data = []
        for file in files: 
            out = pd.read_csv(file).rename(columns={'0': 'r'})
            out['targets'] = fmri_info.roi_name.to_list()
            if 'voxel_id' in fmri_info.columns:
                out['voxel_id'] = fmri_info.voxel_id.to_list()
            else:
                out['subj_id'] = fmri_info.subj_id.to_list()
            out['r2'] = stats.sign_square(out['r'].to_numpy())
            out['condition'] = find_missing(self.categories, extract_elements(file))
            data.append(out.reset_index(drop=True))
        df = pd.concat(data, ignore_index=True).set_index(['condition', 'targets', 'subj_id'])
        all_r2 = df.xs('all', level='condition').reset_index()[['targets', 'subj_id', 'r2']]
        all_r2.columns = ['targets', 'subj_id', 'r2_all']
        df_reset = df.reset_index()
        df_merged = pd.merge(df_reset, all_r2, on=['targets', 'subj_id'], how='left')
        df_merged['r2_diff'] = df_merged['r2_all'] - df_merged['r2']
        df_final = df_merged.set_index(['condition', 'targets', 'subj_id']).drop(columns=['r', 'r2_all']).reset_index()
        df_final = df_final[df_final.condition != 'all']
        df_final = df_final[~df_final.targets.isin(['PPA','FFA'])]
        df_mean = df_final.groupby(['condition', 'targets']).mean(numeric_only=True).reset_index()
        df_mean['targets'] = pd.Categorical(df_mean['targets'], categories=self.rois, ordered=True)
        df_mean['condition'] = pd.Categorical(df_mean['condition'], categories=self.categories, ordered=True)
        return df_mean

    def viz_results(self, results, font=6):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='paper', style='white', rc=custom_params)
        fig, axes = plt.subplots(2, 4, figsize=(6.5, 3),
                                sharey=False, sharex=False)
        axes = axes.flatten()

        for i, (ax, (roi, df)) in enumerate(zip(axes, results.groupby('targets'))):
            sns.barplot(x='condition', y='r2_diff', palette='gray', saturation=0.8,
                        data=df,
                        ax=ax)
            ax.set_title(roi, fontsize=font+2)
            ax.set_xlabel('')
            # ax.set_ylim([0, self.y_max])

            # Change the ytick font size
            label_format = '{:,.2f}'
            y_ticklocs = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
            ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=font)

            # Change the xaxis font size and colors
            if i > 3:
                ax.set_xticklabels(self.categories,
                                fontsize=font,
                                rotation=45, ha='right')
            else:
                ax.set_xticklabels([])

            if i != 0 or i != 4:
                ax.set_ylabel('')
            else:
                ax.set_ylabel(f'Unique variance ($r^2$)', fontsize=font)

            # Manipulate the color and add error bars
            for bar, category in zip(ax.patches, self.categories):
                bar.set_color(cat2color(category))
            ax.legend([], [], frameon=False)
        fig.delaxes(axes[-1])
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/roi-encoding.{get_githash()}.pdf')

    def save_results(self, results):
        results.to_csv(f'{self.out_dir}/roi-encoding.csv.gz', index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        data = self.load_fmri_encoding()
        print(data.head())
        self.mk_out_dir()
        self.viz_results(data)
        self.save_results(data)


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--fmri_encoding', '-e', type=str, help='directory of the decoding results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/encodeDecode/fmri')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotfMRIVariance')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predicted roi mean response instead of voxelwise responses')
    args = parser.parse_args()
    PlotfMRIVariance(args).run()


if __name__ == '__main__':
    main()