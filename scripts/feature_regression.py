import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import corr2d_gpu, perm_gpu, bootstrap_gpu
from src.regression import ridge, feature_scaler, ols
import json
from tqdm import tqdm
from sklearn.model_selection import KFold


def dict_to_tensor(train_dict, test_dict, keys):
    def list_to_tensor(l):
        return torch.hstack(tuple(l))

    train_out, test_out, groups = [], [], []
    for i_group, key in enumerate(keys):
        if train_dict[key].ndim > 1: 
            train_out.append(train_dict[key])
            test_out.append(test_dict[key])
            group_vec = torch.ones(test_dict[key].size()[1])*i_group
        else: 
            train_out.append(torch.unsqueeze(train_dict[key], 1))
            test_out.append(torch.unsqueeze(test_dict[key], 1))
            group_vec = torch.tensor([i_group])
        groups.append(group_vec)
    return list_to_tensor(train_out), list_to_tensor(test_out), list_to_tensor(groups)


def are_all_elements_present(list1, list2):
    return all(elem in list2 for elem in list1)


class FeatureRegression:
    def __init__(self, args):
        self.process = 'FeatureRegression'
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.out_name = f'{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_features.parquet'
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior = loading.check_videos(eeg_raw, behavior)
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')
        eeg = {}
        iterator = tqdm(eeg_filtered.groupby('time_ind'), total=eeg_filtered.time_ind.nunique(), desc='EEG to numpy')
        time_map = {}
        for time_ind, time_df in iterator:
            eeg[time_ind] = loading.strip_eeg(time_df)
            time_map[time_ind] = time_df.time.unique()[0]
        return behavior, {'eeg': eeg}, time_map

    def split_and_norm(self, behavior, data):
        def apply_feature_scaler(train_dict, test_dict, device):
            def recursive_apply(train_subdict, test_subdict):
                for key in train_subdict.keys():
                    if isinstance(train_subdict[key], dict):
                        # Recursively apply to sub-dictionaries
                        recursive_apply(train_subdict[key], test_subdict[key])
                    else:
                        # Apply feature_scaler to the current value
                        train_subdict[key], test_subdict[key] = feature_scaler(train_subdict[key], test_subdict[key], device=device)
            recursive_apply(train_dict, test_dict)
        
        train, test = regression.train_test_split(behavior, data, behavior_categories=self.behavior_categories)
        apply_feature_scaler(train, test, device=self.device)
        return train, test

    def reorganize_results(self, scores, time_map, scores_null, scores_var):
        results = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(results.columns))]
        results.columns = temp_cols
        results = results.rename(index=time_map).reset_index()
        results = pd.melt(results, id_vars='index')
        cols = list(self.behavior_categories.keys())
        results['feature'] = results.variable.replace({temp_col: feature for feature, temp_col in zip(cols, temp_cols)})
        results = results.rename(columns={'index': 'time'}).drop(columns='variable')

        scores_null_df = pd.DataFrame(scores_null.reshape(self.n_perm, -1).transpose(),
                                columns=[f'null_perm_{i}' for i in range(self.n_perm)])
        scores_var_df = pd.DataFrame(scores_var.reshape(self.n_perm, -1).transpose(),
                                columns=[f'var_perm_{i}' for i in range(self.n_perm)])
        scores_null_df[['feature', 'time']] = results[['feature', 'time']]
        scores_var_df[['feature', 'time']] = results[['feature', 'time']]
        scores_null_df.set_index(['feature', 'time'], inplace=True)
        scores_var_df.set_index(['feature', 'time'], inplace=True)

        results = results.set_index(['feature', 'time']).join(scores_null_df).join(scores_var_df).reset_index()
        return results
    
    def standard_regression(self, train, test):
        #Define y
        y_train, y_test, group2 = dict_to_tensor(train, test, list(self.behavior_categories.keys()))

        scores, scores_null, scores_var = {}, [], []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict features from EEG', leave=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X_train, X_test = train['eeg'][time_ind], test['eeg'][time_ind]

            # Determine the optimal number of components for the regression 
            pc_scores = []
            for n_components in range(1, X_train.shape[-1]+1):
                comp_scores = []
                for _, (train_index, test_index) in enumerate(kf.split(X_train)):
                    yhat = ridge(X_train[train_index], y_train[train_index],
                                X_train[test_index],
                                alpha_start=self.alpha_start,
                                alpha_stop=self.alpha_stop,
                                device=self.device,
                                rotate_x=True, 
                                n_components=n_components)['yhat']
                    comp_scores.append(corr2d_gpu(yhat, y_train[test_index]))
                pc_scores.append(torch.vstack(comp_scores).mean(dim=0))
            pc_scores = torch.vstack(pc_scores).mean(dim=1)
            n_components = torch.argmax(pc_scores) + 1

            # Next predict yhat by the features
            yhat = ridge(X_train, y_train, X_test,
                         alpha_start=self.alpha_start,
                         alpha_stop=self.alpha_stop,
                         device=self.device,
                         rotate_x=True,
                         n_components=n_components)['yhat']

            # Evaluate against y and compute stats
            scores[time_ind] = corr2d_gpu(yhat, y_test)
            scores_null.append(torch.unsqueeze(perm_gpu(yhat, y_test, n_perm=self.n_perm), 2))
            scores_var.append(torch.unsqueeze(bootstrap_gpu(yhat, y_test, n_perm=self.n_perm), 2))
        return scores, torch.cat(scores_null, 2).cpu().detach().numpy(), torch.cat(scores_var, 2).cpu().detach().numpy()

    def save_df(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, time_map = self.load_and_validate()
        print(behavior.head())
        train, test = self.split_and_norm(behavior, other_data)
        scores, scores_null, scores_var = self.standard_regression(train, test)
        results = self.reorganize_results(scores, time_map, scores_null, scores_var)
        print(results.head())
        self.mk_out_dir()
        self.save_df(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/FeatureRegression')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')         
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    args = parser.parse_args()
    FeatureRegression(args).run()


if __name__ == '__main__':
    main()