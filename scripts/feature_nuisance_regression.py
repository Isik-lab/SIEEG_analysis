import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import perm_gpu, bootstrap_gpu
from src.regression import ridge, feature_scaler, ols
import json
from tqdm import tqdm
from src.stats import compute_score
from src.tools import dict_to_tensor


class FeatureNuisanceRegression:
    def __init__(self, args):
        self.process = 'FeatureNuisanceRegression'
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.feature_to_predict = args.feature_to_predict
        self.out_name = f'{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_feature-{self.feature_to_predict}.parquet'
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}
        self.x_nuisance = ['alexnet', 'moten'] + list(self.behavior_categories)
        self.x_nuisance.remove(self.feature_to_predict)

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        moten = loading.load_model_activations(self.motion_energy)
        alexnet = loading.load_model_activations(self.alexnet)
        
        # Check EEG trials 
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior = loading.check_videos(eeg_raw, behavior, [alexnet, moten])
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')

        # Transform EEG to dict 
        eeg = {}
        iterator = tqdm(eeg_filtered.groupby('time_ind'), total=eeg_filtered.time_ind.nunique(), desc='EEG to numpy')
        time_map = {}
        for time_ind, time_df in iterator:
            eeg[time_ind] = loading.strip_eeg(time_df)
            time_map[time_ind] = time_df.time.unique()[0]
        return behavior, {'eeg': eeg, 'alexnet': alexnet, 'moten': moten}, time_map

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
        
        #Define x2
        # Normalize the behavior features
        for feature in self.behavior_categories.keys(): 
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
        # Normalize AlexNet and Moten and reduce the dimensionality
        for feature in ['alexnet', 'moten']:
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
            train[feature], test[feature], _ = regression.pca_rotation(train[feature], test[feature])
            print(f'{feature} PCs: {train[feature].size()[1]}')
        y_train, y_test, _ = dict_to_tensor(train, test, self.feature_to_predict)
        X_nuisance_train, _, _ = dict_to_tensor(train, test, self.x_nuisance)

        scores, scores_null, scores_var = {}, [], []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict features from EEG', leave=True)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X_eeg_train, X_eeg_test = train['eeg'][time_ind], test['eeg'][time_ind]
            X_train = torch.hstack((X_train, ))

            y_pred = ridge(X_train, y_train, X_eeg_test,
                         alpha_start=self.alpha_start,
                         alpha_stop=self.alpha_stop,
                         device=self.device,
                         rotate_x=True)['yhat']

            # Evaluate against y
            scores[time_ind] = compute_score(y_true, y_pred, score_type=self.scoring,
                                             adjusted=X_train.size()[1])

            # Compute states 
            perm = perm_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                            adjusted=X_train.size()[1])
            var = bootstrap_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                                adjusted=X_train.size()[1])
            scores_null.append(torch.unsqueeze(perm, 2))
            scores_var.append(torch.unsqueeze(var, 2))
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
        print(f'{scores_null.shape=}')
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
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    parser.add_argument('--feature_to_predict', '-y', type=str, default='expanse',
                        help='the feature that you want to predict')
    args = parser.parse_args()
    FeatureNuisanceRegression(args).run()


if __name__ == '__main__':
    main()