import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats, logging
import torch
from pathlib import Path
import numpy as np
import seaborn as sns
import pickle


class PlotDecoding:
    def __init__(self, args):
        self.process = 'PlotDecoding'
        # logging.neptune_init(self.process)
        self.decoding_dir = args.decoding_dir
        self.fmri_dir = args.fmri_dir
        self.eeg_dir = args.eeg_dir
        self.y_name = args.y_name
        self.x_name = args.x_name
        assert self.x_name == 'eeg' or self.x_name == 'eeg_behavior', 'x input must be eeg or eeg_behavior'
        assert self.y_name == 'behavior' or self.y_name == 'fmri', 'y input must be behavior or fmri'
        self.out_dir = args.out_dir
        # logging.neptune_params(self)
        print(vars(self))

    @staticmethod
    def map_ind_to_time(data, time_map):
        out = data.copy()
        out['time'] = out['time_id'].map(time_map)
        return out

    def get_targets(self):
        if self.y_name == 'behavior':
            y_data = loading.load_behavior(self.fmri_dir)
            targets = [col for col in y_data.columns if 'rating' in col]
        else: #self.y_name == 'fmri'
            _, y_data = loading.load_fmri(self.fmri_dir)
            targets = y_data.voxel_id.to_list()
        return targets

    def get_time_map(self):
        with open(f'{self.eeg_dir}/map_time.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict

    def load_results(self, targets):
        name_pattern = f'*x-{self.x_name}_y-{self.y_name}_scores.csv.gz'
        return loading.load_decoding_files(self.decoding_dir, name_pattern, targets)

    def viz_results(self, results):
        data.groupby('time').mean(numeric_only=True).reset_index()
        sns.plot(x='time', y='')

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        targets = self.get_targets()
        data = self.load_results(targets)
        time_map = self.get_time_map()
        data = self.map_ind_to_time(data, time_map)
        print(data.head())
        self.mk_out_dir()
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Combine and plot the EEG decoding results')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory')
    parser.add_argument('--eeg_dir', '-e', type=str, help='EEG preprocessing directory')
    parser.add_argument('--decoding_dir', '-d', type=str, help='directory of the decoding results')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for plot outputs')
    parser.add_argument('--y_name', '-y', type=str, help='name of the data to be used as regression target')
    parser.add_argument('--x_name', '-x', type=str, help='name of the data for regression fitting')
    args = parser.parse_args()
    PlotDecoding(args).run()


if __name__ == '__main__':
    main()