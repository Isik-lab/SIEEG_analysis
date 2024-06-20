import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import corr2d_gpu
from src.regression import regression_model, feature_scaler


class eegDecoding:
    def __init__(self, args):
        self.process = 'eegDecoding'
        self.fmri_dir = args.fmri_dir
        self.eeg_file = args.eeg_file
        self.motion_energy = args.motion_energy
        self.alexnet = args.alexnet
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
        eeg_raw = loading.load_eeg(self.eeg_file)

        behavior = loading.load_behavior(self.fmri_dir)
        fmri, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        moten = loading.load_model_activations(self.motion_energy)
        alexnet = loading.load_model_activations(self.alexnet)
        
        eeg_filtered, behavior, [fmri, alexnet, moten] = loading.check_videos(eeg_raw, behavior, [fmri, alexnet, moten])
        eeg = loading.strip_eeg(eeg_filtered)
        return behavior, {'eeg': eeg, 'fmri': fmri, 'alexnet': alexnet, 'moten': moten}

    def split_and_norm(self, behavior, data):
        train, test = regression.train_test_split(behavior, data, device=self.device)
        for key in train.keys():
            train[key], test[key] = feature_scaler(train[key], test[key], device=self.device)

        if self.x_name == 'eeg_behavior' and self.y_name == 'fmri':
            y_train, y_test = train['fmri'], test['fmri']
            X_train = torch.hstack((train['eeg'],
                                    train['behavior'],
                                    train['alexnet'],
                                    train['moten']))
            X_test = torch.hstack((test['eeg'],
                                   test['behavior'],
                                   test['alexnet'],
                                   test['moten']))
            groups = torch.hstack((torch.ones(train['eeg'].size()[1])*0,
                                   torch.ones(train['behavior'].size()[1])*1,
                                   torch.ones(train['alexnet'].size()[1])*2,
                                   torch.ones(train['moten'].size()[1])*3))
        elif self.x_name == 'eeg' and self.y_name == 'fmri': 
            y_train, y_test = train['fmri'], test['fmri']
            X_train, X_test = train['eeg'], test['eeg']
            groups = torch.ones(train['eeg'].size()[1])*0
        elif self.x_name == 'eeg' and self.y_name == 'behavior':
            y_train, y_test = train['behavior'], test['behavior']
            X_train, X_test = train['eeg'], test['eeg']
            groups = torch.ones(train['eeg'].size()[1])*0
        else: 
            raise Exception(f'Sorry regression not implemented for combinations of x={self.x_name} and y={self.y_name}')

        return X_train, X_test, y_train, y_test, groups

    def save_results(self, results):
        for key, val in results.items():
            out_file = f'{self.out_dir}/{self.eeg_base}_x-{self.x_name}_y-{self.y_name}_{key}.csv.gz'
            pd.DataFrame(tools.to_numpy(val)).to_csv(out_file, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def get_kwargs(self, groups):
        out = vars(self).copy()
        out['groups'] = groups
        return out

    def run(self):
        behavior, other_data = self.load_and_validate()
        X_train, X_test, y_train, y_test, groups = self.split_and_norm(behavior, other_data)
        print(f'X_train mean check: {np.isclose(torch.mean(X_train, dim=0)[0], 0)}')
        print(f'X_train std check: {np.isclose(torch.std(X_train, dim=0)[0], 1)}')
        print(f'y_train mean check: {np.isclose(torch.mean(y_train, dim=0)[0], 0)}')
        print(f'y_train std check: {np.isclose(torch.std(y_train, dim=0)[0], 1)}')

        kwargs = self.get_kwargs(groups)
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
    parser.add_argument('--alexnet', '-a', type=str, help='AlexNet activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/AlexNetActivations/alexnet_conv2.npy')
    parser.add_argument('--motion_energy', '-m', type=str, help='Motion energy activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/MotionEnergyActivations/motion_energy.npy')
    parser.add_argument('--y_name', '-y', type=str, default='behavior',
                        help='name of the data to be used as regression target')
    parser.add_argument('--x_name', '-x', type=str, default='eeg',
                        help='name of the data for regression fitting')
    parser.add_argument('--rotate_x', action=argparse.BooleanOptionalAction, default=True,
                        help='rotate the values of X by performing PCA before regression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--regression_method', '-r', type=str, default='banded_ridge',
                        help='whether to perform ridge or ols regression')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    eegDecoding(args).run()


if __name__ == '__main__':
    main()