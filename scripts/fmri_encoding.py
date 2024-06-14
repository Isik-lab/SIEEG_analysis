import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import logging, loading, regression, tools, stats
import torch
from pathlib import Path
from src.regression import T_torch
from src.stats import corr2d_gpu
from src.regression import regression_model


class fmriEncodings:
    def __init__(self, args):
        self.process = 'fmriEncodings'
        self.fmri_dir = args.fmri_dir
        self.out_dir = args.out_dir
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.rotate_x = args.rotate_x
        self.roi_mean = args.roi_mean
        self.regression_method = args.regression_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self):
        fmri, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        return {'behavior': loading.load_behavior(self.fmri_dir),
                 'fmri': fmri}

    def split_data(self, data):
        splits = regression.split_data(data['behavior'], {'fmri': data['fmri']})
        X_train, X_test = splits['behavior_train'], splits['behavior_test']
        y_train, y_test = splits['fmri_train'], splits['fmri_test']
        return [X_train], [X_test], y_train, y_test

    def save_results(self, results):
        for key, val in results.items():
            out_file = f'{self.out_dir}/{key}.csv.gz'
            pd.DataFrame(tools.to_numpy(val)).to_csv(out_file, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def get_kwargs(self):
        kwargs = vars(self).copy()
        return kwargs

    def run(self):
        data = self.load()
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


def main():
    parser = argparse.ArgumentParser(description='Predict fMRI responses using the features')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory for the regression results',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fmriEncoding')
    parser.add_argument('--regression_method', '-r', type=str, default='banded_ridge',
                        help='whether to perform OLS, ridge, or banded ridge regression')
    parser.add_argument('--rotate_x', action=argparse.BooleanOptionalAction, default=True,
                        help='rotate the values of X by performing PCA before regression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    fmriEncodings(args).run()


if __name__ == '__main__':
    main()