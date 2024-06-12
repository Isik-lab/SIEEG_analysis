import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import corr2d_gpu
from src.regression import regression_model


class eegDecoding:
    def __init__(self, args):
        self.process = 'eegDecoding'
        self.fmri_dir = args.fmri_dir
        self.eeg_file = args.eeg_file
        self.y_name = args.y_name
        self.x_name = args.x_name
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.rotate_x = args.rotate_x
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eeg_base = self.eeg_file.split('/')[-1].split('.')[0]
        self.out_dir = args.out_dir
        self.regression_method = args.regression_method
        self.roi_mean = args.roi_mean
        print(vars(self))
        assert self.x_name == 'eeg' or self.x_name == 'eeg_behavior', 'x input must be eeg or eeg_behavior'
        assert self.y_name == 'behavior' or self.y_name == 'fmri', 'y input must be behavior or fmri'
        

    def load_and_validate(self):
        behavior_raw = loading.load_behavior(self.fmri_dir)
        fmri_raw, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_filtered, behavior, fmri = loading.check_videos(eeg_raw, behavior_raw, fmri_raw)
        eeg = loading.strip_eeg(eeg_filtered)
        return {'behavior': behavior, 'eeg': eeg, 'fmri': fmri}

    def split_data(self, data):
        splits = regression.split_data(data['behavior'], {'eeg': data['eeg'], 'fmri': data['fmri']})
        if self.y_name == 'behavior' and self.x_name == 'eeg':
            X_train, X_test = splits['eeg_train'], splits['eeg_test']
            y_train, y_test = splits['behavior_train'], splits['behavior_test']
        elif self.y_name == 'fmri' and self.x_name == 'eeg':
            X_train, X_test = splits['eeg_train'], splits['eeg_test']
            y_train, y_test = splits['fmri_train'], splits['fmri_test']
        elif self.y_name == 'fmri' and self.x_name == 'eeg_behavior':
            X_train = np.hstack((splits['eeg_train'], splits['behavior_train']))
            X_test = np.hstack((splits['eeg_test'], splits['behavior_test']))
            y_train, y_test = splits['fmri_train'], splits['fmri_test']
        else:
            raise Exception(f'Sorry regression not implemented for combinations of x={self.x_name} and y={self.y_name}')
            
        return X_train, X_test, y_train, y_test

    def save_results(self, results):
        for key, val in results.items():
            out_file = f'{self.out_dir}/{self.eeg_base}_x-{self.x_name}_y-{self.y_name}_{key}.csv.gz'
            pd.DataFrame(tools.to_numpy(val)).to_csv(out_file, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def get_kwargs(self):
        return vars(self).copy()

    def run(self):
        data = self.load_and_validate()
        X_train, X_test, y_train, y_test = self.split_data(data)
        [X_train, X_test, y_train, y_test] = tools.to_torch([X_train, X_test, y_train, y_test],
                                                            device=self.device)
        regression.preprocess(X_train, X_test, y_train, y_test) #inplace
        print(f'{X_train.size()=}')
        print(f'{X_test.size()=}')
        print(f'{y_train.size()=}')
        print(f'{y_test.size()=}')

        kwargs = self.get_kwargs()
        results = regression_model(self.regression_method,
                                   X_train, y_train, X_test,
                                   **kwargs)
        results['scores'] = corr2d_gpu(results['yhat'], y_test)

        self.mk_out_dir()
        self.save_results(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/sub-01_time-00.csv.gz')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegDecoding')
    parser.add_argument('--y_name', '-y', type=str, default='behavior',
                        help='name of the data to be used as regression target')
    parser.add_argument('--x_name', '-x', type=str, default='eeg',
                        help='name of the data for regression fitting')
    parser.add_argument('--rotate_x', action=argparse.BooleanOptionalAction, default=True,
                        help='rotate the values of X by performing PCA before regression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--regression_method', '-r', type=str, default='ridge',
                        help='whether to perform ridge or ols regression')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=10,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    eegDecoding(args).run()


if __name__ == '__main__':
    main()