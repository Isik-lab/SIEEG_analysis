import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import numpy as np

import json
from tqdm import tqdm

import sys

from eeg.stats import perm_gpu, bootstrap_gpu, compute_score
from eeg.regression import ridge, feature_scaler, ols
from eeg import loading, regression
from eeg.tools import dict_to_tensor


class FeatureRegression:
    def __init__(self, args):
        self.process = 'FeatureRegression'
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.run_stats = args.run_stats
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.out_name = f"{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_features.parquet"
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication_500ms',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        
        # Check EEG trials 
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior = loading.check_videos(eeg_raw, behavior)
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')

        # Transform EEG to dict 
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

        if scores_null is not None and scores_var is not None:
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
        y_train, y_true, group2 = dict_to_tensor(train, test, list(self.behavior_categories.keys()))

        scores = {}
        scores_null = [] if self.run_stats else None
        scores_var = [] if self.run_stats else None
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict features from EEG', leave=True)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X_train, X_test = train['eeg'][time_ind], test['eeg'][time_ind]

            y_pred = ridge(X_train, y_train, X_test,
                         alpha_start=self.alpha_start,
                         alpha_stop=self.alpha_stop,
                         device=self.device,
                         rotate_x=True)['yhat']

            # Evaluate against y
            score = compute_score(y_true, y_pred, score_type=self.scoring,
                                  adjusted=X_train.size()[1])
            if isinstance(score, torch.Tensor):
                score = score.detach().cpu().numpy()
            scores[time_ind] = score

            if self.run_stats:
                # Compute stats 
                perm = perm_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                                adjusted=X_train.size()[1])
                var = bootstrap_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                                    adjusted=X_train.size()[1])
                scores_null.append(torch.unsqueeze(perm, 2))
                scores_var.append(torch.unsqueeze(var, 2))
        if self.run_stats:
            scores_null = torch.cat(scores_null, 2).cpu().detach().numpy()
            scores_var = torch.cat(scores_var, 2).cpu().detach().numpy()
        return scores, scores_null, scores_var

    def save_df(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, time_map = self.load_and_validate()
        print(behavior.head())
        train, test = self.split_and_norm(behavior, other_data)
        scores, scores_null, scores_var = self.standard_regression(train, test)
        if self.run_stats:
            print(f'{scores_null.shape=}')
        results = self.reorganize_results(scores, time_map, scores_null, scores_var)
        print(results.head())
        self.mk_out_dir()
        self.save_df(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/FeatureRegression')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    parser.add_argument('--run_stats', action='store_true',
                        help='run permutation and bootstrap statistics (default: off)')
    args = parser.parse_args()
    FeatureRegression(args).run()


if __name__ == '__main__':
    main()