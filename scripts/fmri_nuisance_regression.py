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


class fMRINuisanceRegression:
    def __init__(self, args):
        self.process = 'fMRINuisanceRegression'
        self.roi_mean = args.roi_mean
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.smoothing = args.smoothing
        self.roi_to_predict = args.roi_to_predict
        self.sub = args.sub
        if self.roi_mean:
            self.out_name = f'{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_fsub-{str(self.sub).zfill(2)}_roi-{self.roi_to_predict}.parquet'
        elif not self.roi_mean:
            self.out_name = f'{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_full-brain'
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        fmri, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean, smoothing=self.smoothing)
        print(fmri.columns)
        x_nuisance_cols = [col for col in fmri.columns if f'sub-{self.sub}' in col and 'face' not in col and self.roi_to_predict not in col]
        y_cols = [col for col in fmri.columns if f'sub-{self.sub}' in col and 'face' not in col  and self.roi_to_predict in col]
        print('nuisance cols:')
        print(x_nuisance_cols)
        print('y cols:')
        print(y_cols)
        print()
        fmri_X = fmri[x_nuisance_cols]
        fmri_y = fmri[y_cols]
        
        # Check EEG trials 
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior, [fmri_X, fmri_y] = loading.check_videos(eeg_raw, behavior, [fmri_X, fmri_y])
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')
        
        # Transform EEG to dict 
        eeg = {}
        iterator = tqdm(eeg_filtered.groupby('time_ind'), total=eeg_filtered.time_ind.nunique(), desc='EEG to numpy')
        time_map = {}
        for time_ind, time_df in iterator:
            eeg[time_ind] = loading.strip_eeg(time_df)
            time_map[time_ind] = time_df.time.unique()[0]
        return behavior, {'eeg': eeg, 'fmri_X': fmri_X, 'fmri_y': fmri_y}, time_map

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
        results = pd.DataFrame({'score': scores}, index=time_map.values())
        results['roi_name'] = self.roi_to_predict
        results['fmri_subj_id'] = self.sub
        results['score'] = results['score'].astype('float32')
        print(results.head())

        scores_null_df = pd.DataFrame(scores_null.transpose(), index=time_map.values(),
                                columns=[f'null_perm_{i}' for i in range(self.n_perm)])
        scores_var_df = pd.DataFrame(scores_var.transpose(), index=time_map.values(),
                                columns=[f'var_perm_{i}' for i in range(self.n_perm)])

        results = results.join(scores_null_df).join(scores_var_df).reset_index()
        return results.rename(columns={'index': 'time'})
    
    def standard_regression(self, train, test):
        #Define y
        train['fmri_X'], train['fmri_X'] = feature_scaler(train['fmri_X'], train['fmri_X'], device=self.device)
        train['fmri_y'], train['fmri_y'] = feature_scaler(train['fmri_y'], train['fmri_y'], device=self.device)
        X_nuisance_train, _, _ = dict_to_tensor(train, test, ['fmri_X'])
        y_train, y_true, _ = dict_to_tensor(train, test, ['fmri_y'])

        scores, scores_null, scores_var = [], [], []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict fMRI from EEG', leave=True)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X_eeg_train, X_eeg_test = train['eeg'][time_ind], test['eeg'][time_ind]
            X_train = torch.hstack((X_eeg_train, X_nuisance_train))
            groups = torch.hstack((torch.zeros(X_eeg_train.size(1)),
                                   torch.ones(X_nuisance_train.size(1))))

            eeg_betas = []
            for i_split in range(100): 
                torch.manual_seed(i_split)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(i_split)
                perm = torch.randperm(X_train.size(0))
                idx = perm[:int(X_train.size(0)/2)] 
                betas = ridge(X_train, y_train,
                            groups=groups,
                            alpha_start=self.alpha_start,
                            alpha_stop=self.alpha_stop,
                            device=self.device,
                            return_betas=True,
                            rotate_x=True)['betas']
                eeg_betas.append(torch.unsqueeze(betas[:X_eeg_train.shape[-1]], dim=2))
            # Get the mean of the betas over 100 random samples of the training set
            eeg_betas = torch.mean(torch.cat(eeg_betas, dim=2), dim=2)
            # Fit the prediction
            y_pred = torch.matmul(X_eeg_test, eeg_betas)

            # Evaluate against y
            # Evaluate against y
            scores.append(compute_score(y_true, y_pred, score_type=self.scoring,
                            adjusted=X_train.size()[1]).cpu().detach().numpy())

            # Compute states 
            perm = perm_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                            adjusted=X_train.size()[1])
            var = bootstrap_gpu(y_true, y_pred, n_perm=self.n_perm, score_type=self.scoring,
                                adjusted=X_train.size()[1])
            scores_null.append(perm)
            scores_var.append(var)
        scores_null = torch.cat(scores_null, 1).cpu().detach().numpy()
        scores_var = torch.cat(scores_var, 1).cpu().detach().numpy()
        return scores, scores_null, scores_var

    def save_df(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def save_indiv_times(self, scores, fmri_meta):
        for time_ind, score in scores.items():
            results = pd.DataFrame(score.cpu().detach().numpy(),
                                   index=fmri_meta['voxel_id'],
                                   columns=['value'])
            if time_ind == 0:
                print(results.head())
            results.to_parquet(f'{self.out_name}_time-{str(time_ind).zfill(3)}.parquet')

    def run(self):
        behavior, other_data, time_map = self.load_and_validate()
        train, test = self.split_and_norm(behavior, other_data)
        self.mk_out_dir()
        scores, scores_null, scores_var = self.standard_regression(train, test)
        results = self.reorganize_results(scores, time_map, scores_null, scores_var)
        print(results.head())
        self.save_df(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fMRINuisanceRegression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--smoothing', action=argparse.BooleanOptionalAction, default=False,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    parser.add_argument('--roi_to_predict', '-y', type=str, default='pSTS',
                        help='the ROI that you want to predict')
    parser.add_argument('--sub', '-s', type=int, default=4,
                        help='the fMRI subject to predict')
    args = parser.parse_args()
    fMRINuisanceRegression(args).run()


if __name__ == '__main__':
    main()