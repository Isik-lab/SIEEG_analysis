import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import logging, loading, regression, tools, stats
import torch
from pathlib import Path
from src.regression import T_torch
from src.stats import corr2d_gpu
from src.regression import regression_model, feature_scaler
import numpy as np
import json


def dict_to_tensor(train_dict, test_dict, keys):
    def list_to_tensor(l):
        return torch.hstack(tuple(l))

    def group_vec(tensor, i):
        return torch.ones(tensor.size()[1])*i_group

    train_out, test_out, groups = [], [], []
    for i_group, key in enumerate(keys):
        train_out.append(train_dict[key])
        test_out.append(test_dict[key])
        groups.append(group_vec(train_dict[key], i_group))
    return list_to_tensor(train_out), list_to_tensor(test_out), list_to_tensor(groups)


def are_all_elements_present(list1, list2):
    return all(elem in list2 for elem in list1)


class fmriEncoding:
    def __init__(self, args):
        self.process = 'fmriEncoding'
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.rotate_x = args.rotate_x
        self.roi_mean = args.roi_mean
        self.regression_method = args.regression_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_names = json.loads(args.y_names)
        self.x_names = json.loads(args.x_names)
        valid_names = ['fmri', 'eeg', 'alexnet', 'moten', 'scene', 'primitive', 'social', 'affective']
        valid_err_msg = f"One or more x_names are not valid. Valid options are {valid_names}"
        assert all(name in valid_names for name in self.x_names), valid_err_msg
        assert all(name in valid_names for name in self.y_names), valid_err_msg
        print(vars(self))
        self.fmri_dir = args.fmri_dir
        self.motion_energy = args.motion_energy
        self.alexnet = args.alexnet
        self.out_dir = args.out_dir
        self.behavior_categories = {'scene': ['rating-indoor', 'rating-expanse', 'rating-object'],
                                    'primitive': ['rating-agent_distance', 'rating-facingness'],
                                    'social': ['rating-joint_action', 'rating-communication'],
                                    'affective': ['rating-valence', 'rating-arousal']}

    def load(self):
        fmri, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        moten = loading.load_model_activations(self.motion_energy)
        alexnet = loading.load_model_activations(self.alexnet)
        return loading.load_behavior(self.fmri_dir), {'fmri': fmri,
                                                     'alexnet': alexnet,
                                                     'moten': moten}

    def split_and_norm(self, behavior, data):
        train, test = regression.train_test_split(behavior, data, behavior_categories=self.behavior_categories)
        train_normed, test_normed = {}, {}
        for key in train.keys():
            train_normed[key], test_normed[key] = feature_scaler(train[key], test[key], device=self.device)

        X_train, X_test, groups = dict_to_tensor(train_normed, test_normed, self.x_names)
        print(np.all(np.isclose(torch.mean(X_train, dim=0).cpu().detach().numpy(), 0)))
        y_train, y_test, _ = dict_to_tensor(train_normed, test_normed, self.y_names)
        return X_train, X_test, y_train, y_test, groups

    def save_results(self, results):
        for key, val in results.items():
            out_file = f'{self.out_dir}/x-{'-'.join(self.x_names)}_y-{'-'.join(self.y_names)}_{key}.csv.gz'
            pd.DataFrame(tools.to_numpy(val)).to_csv(out_file, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def get_kwargs(self, groups):
        out = vars(self).copy()
        out['groups'] = groups
        return out

    def run(self):
        behavior, other_data = self.load()
        X_train, X_test, y_train, y_test, groups = self.split_and_norm(behavior, other_data)
        print(f'X_train mean check: {np.all(np.isclose(torch.mean(X_train, dim=0).cpu().detach().numpy(), 0, atol=1e-5))}')
        print(f'X_train std check: {np.all(np.isclose(torch.std(X_train, dim=0).cpu().detach().numpy(), 1, atol=1e-5))}')
        print(f'y_train mean check: {np.all(np.isclose(torch.mean(y_train, dim=0).cpu().detach().numpy(), 0, atol=1e-5))}')
        print(f'y_train std check: {np.all(np.isclose(torch.std(y_train, dim=0).cpu().detach().numpy(), 1, atol=1e-5))}')

        kwargs = self.get_kwargs(groups)
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
    parser.add_argument('--alexnet', '-a', type=str, help='AlexNet activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/AlexNetActivations/alexnet_conv2.npy')
    parser.add_argument('--motion_energy', '-m', type=str, help='Motion energy activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/MotionEnergyActivations/motion_energy.npy')
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
    parser.add_argument('--y_names', '-y', type=str, default='["fmri"]',
                        help='a list of data names to be used as regression target')
    parser.add_argument('--x_names', '-x', type=str, default='["alexnet", "moten", "scene", "primitive", "social", "affective"]',
                        help='a list of data names for regression fitting')
    args = parser.parse_args()
    fmriEncoding(args).run()


if __name__ == '__main__':
    main()