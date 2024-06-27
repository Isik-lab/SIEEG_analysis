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
import json


class PlotTimeCourse:
    def __init__(self, args):
        self.process = 'PlotTimeCourse'
        # logging.neptune_init(self.process)
        self.decoding_dir = args.decoding_dir
        self.fmri_dir = args.fmri_dir
        self.eeg_dir = args.eeg_dir
        self.y_names = json.loads(args.y_names)
        self.x_names = json.loads(args.x_names)
        valid_names = ['fmri', 'eeg', 'alexnet', 'moten', 'scene', 'primitive', 'social', 'affective']
        valid_err_msg = f"One or more x_names are not valid. Valid options are {valid_names}"
        assert all(name in valid_names for name in self.x_names), valid_err_msg
        assert all(name in valid_names for name in self.y_names), valid_err_msg
        self.out_dir = args.out_dir
        self.roi_mean = args.roi_mean
        self.behavior_categories = {'scene': ['rating-indoor', 'rating-expanse', 'rating-object'],
                                    'primitive': ['rating-agent_distance', 'rating-facingness'],
                                    'social': ['rating-joint_action', 'rating-communication'],
                                    'affective': ['rating-valence', 'rating-arousal']}
        # logging.neptune_params(self)
        print(vars(self))

    @staticmethod
    def map_ind_to_time(data, time_map):
        out = data.copy()
        out['time'] = out['time_id'].map(time_map)
        return out

    @staticmethod
    def average_over_subjs(results):
        return results.groupby(['time', 'targets']).mean(numeric_only=True).reset_index()
    
    @staticmethod
    def compute_explained_variance(results):
        out = results.copy()
        out['r2'] = stats.sign_square(out['r'].to_numpy())
        return out

    def get_targets(self):
        if 'fmri' not in self.y_names:
            y_data = loading.load_behavior(self.fmri_dir)
            out = {'targets': [col for col in y_data.columns if 'rating' in col]}
        else:
            _, y_data = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
            if self.roi_mean:
                out = {'subj_id': y_data.subj_id.to_list(),
                       'targets': y_data.roi_name.to_list()}
            else:
                out = {'voxel_id': y_data.voxel_id.to_list(),
                       'targets': y_data.roi_name.to_list()}
        return out

    def get_time_map(self):
        with open(f'{self.eeg_dir}/map_time.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict

    def load_results(self, targets):
        name_pattern = f'*x-{'-'.join(self.x_names)}_y-{'-'.join(self.y_names)}_scores.csv.gz'
        return loading.load_decoding_files(self.decoding_dir, name_pattern, targets)

    def viz_results(self, results):
        fig, ax = plt.subplots()
        sns.lineplot(x='time', y='r2', hue='targets', data=results, ax=ax)
        plt.savefig(f'{self.out_dir}/x-{'-'.join(self.x_names)}_y-{'-'.join(self.y_names)}.{get_githash()}.pdf')

    def save_results(self, results):
        results.to_csv(f'{self.out_dir}/x-{'-'.join(self.x_names)}_y-{'-'.join(self.y_names)}.csv.gz', index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        targets = self.get_targets()
        data = self.load_results(targets)
        time_map = self.get_time_map()
        data = self.map_ind_to_time(data, time_map)
        data = self.average_over_subjs(data)
        data = self.compute_explained_variance(data)

        print(data.head())
        self.mk_out_dir()
        self.viz_results(data)
        self.save_results(data)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_dir', '-e', type=str, help='EEG preprocessing directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/')
    parser.add_argument('--decoding_dir', '-d', type=str, help='directory of the decoding results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/encodeDecode/eeg')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotTimeCourse')
    parser.add_argument('--y_names', '-y', type=str, default='["fmri"]',
                        help='a list of data names to be used as regression target')
    parser.add_argument('--x_names', '-x', type=str, default='["eeg"]',
                        help='a list of data names for regression fitting')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predicted roi mean response instead of voxelwise responses')
    args = parser.parse_args()
    PlotTimeCourse(args).run()


if __name__ == '__main__':
    main()