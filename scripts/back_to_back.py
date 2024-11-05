import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import perm_unique_variance_gpu as perm_uv
from src.stats import bootstrap_unique_variance_gpu as boot_uv
from src.regression import ridge, feature_scaler, ols
import json
from tqdm import tqdm
from sklearn.model_selection import KFold
from src.stats import compute_score
from src.tools import dict_to_tensor


class Back2Back:
    def __init__(self, args):
        self.process = 'Back2Back'
        self.roi_mean = args.roi_mean
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.feature_ablation = args.feature_to_ablate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.out_base = f'{str(Path(self.eeg_file).stem)}_feature-{self.feature_ablation}.parquet'
        self.out_name = str(Path(self.out_dir).joinpath(self.out_base))
        self.fmri_dir = args.fmri_dir
        self.motion_energy = args.motion_energy
        self.alexnet = args.alexnet        
        self.random_state = args.random_state
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}
        self.X2_names_all = ['alexnet', 'moten'] + list(self.behavior_categories)
        self.X2_name_ablate = self.X2_names_all.copy()
        self.X2_name_ablate.remove(self.feature_ablation)
        print(vars(self)) 

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        fmri, fmri_meta = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        moten = loading.load_model_activations(self.motion_energy)
        alexnet = loading.load_model_activations(self.alexnet)
        
        # Check EEG trials 
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior, [fmri, alexnet, moten] = loading.check_videos(eeg_raw, behavior, [fmri, alexnet, moten])
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')
        eeg_filtered = eeg_filtered.loc[eeg_filtered.time > 950].reset_index()

        # Transform EEG to dict 
        eeg = {}
        iterator = tqdm(eeg_filtered.groupby('time_ind'), total=eeg_filtered.time_ind.nunique(), desc='EEG to numpy')
        time_map = {}
        for time_ind, time_df in iterator:
            eeg[time_ind] = loading.strip_eeg(time_df)
            time_map[time_ind] = time_df.time.unique()[0]
        return behavior, {'eeg': eeg, 'fmri': fmri, 'alexnet': alexnet, 'moten': moten}, fmri_meta, time_map

    def split(self, behavior, data):
        train, test = regression.train_test_split(behavior, data, behavior_categories=self.behavior_categories)
        return train, test

    def reorg_stats(self, stats, results, col_name):
        stats_df = pd.DataFrame(stats.reshape(self.n_perm, -1).transpose(),
                                columns=[f'{col_name}_perm_{i}' for i in range(self.n_perm)])
        stats_df[['fmri_subj_id', 'roi_name', 'time']] = results[['fmri_subj_id', 'roi_name', 'time']]
        return stats_df.set_index(['fmri_subj_id', 'roi_name', 'time'])

    def reorg_scores(self, scores, fmri_meta, time_map, col_name):
        scores_df = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(scores_df.columns))]
        scores_df.columns = temp_cols
        scores_df = scores_df.rename(index=time_map).reset_index()
        scores_df = pd.melt(scores_df, id_vars='index')
        scores_df['fmri_subj_id'] = scores_df.variable.replace({temp_col: subj_id for subj_id, temp_col in zip(fmri_meta.subj_id, temp_cols)})
        scores_df['roi_name'] = scores_df.variable.replace({temp_col: roi_name for roi_name, temp_col in zip(fmri_meta.roi_name, temp_cols)})
        scores_df = scores_df.rename(columns={'index': 'time'}).drop(columns='variable')
        return scores_df.rename(columns={'value': col_name}).set_index(['fmri_subj_id', 'roi_name', 'time'])

    def reorganize_results(self, cv_scores, scores, scores_null, scores_var, fmri_meta, time_map): 
        results = self.reorg_scores(scores, fmri_meta, time_map, 'eeg_score')
        cv_scores = self.reorg_scores(cv_scores, fmri_meta, time_map, 'score')
        results = results.join(cv_scores)
        if type(scores_null) != list: 
            scores_null = self.reorg_stats(scores_null, results, 'null')
            results = results.join(scores_null)
        
        if type(scores_var) != list:
            scores_var = self.reorg_stats(scores_var, results, 'var')
            results = results.join(scores_var)
        return results.reset_index()
    
    def b2b_regression(self, train, test):
        #Define y
        _, y_true = feature_scaler(train['fmri'], test['fmri'], device=self.device)

        #Define x2
        # Normalize the behavior features
        for feature in self.behavior_categories.keys(): 
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
        # Normalize AlexNet and Moten and reduce the dimensionality
        for feature in ['alexnet', 'moten']:
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
            train[feature], test[feature], _ = regression.pca_rotation(train[feature], test[feature])
            print(f'{feature} PCs: {train[feature].size()[1]}')
        X2_train_all, X2_test_all, _ = dict_to_tensor(train, test, self.X2_names_all)
        X2_train_ablate, X2_test_ablate, _ = dict_to_tensor(train, test, self.X2_name_ablate)

        reg1_scores, reg2_scores = {}, {}
        reg2_scores_null, reg2_scores_var = [], []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc='Back to back regression', leave=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X1 = train['eeg'][time_ind]
            yhat_train, scores_cv = [], []
            for train_index, test_index in kf.split(X1):
                X1_train, X1_test = feature_scaler(X1[train_index], X1[test_index])
                y_train_cv, y_true_cv = feature_scaler(train['fmri'][train_index], train['fmri'][test_index])
                y_pred_cv = ridge(X1_train, y_train_cv, X1_test,  
                                 alpha_start=self.alpha_start,
                                 alpha_stop=self.alpha_stop,
                                 device=self.device,
                                 rotate_x=True)['yhat']
                scores_cv.append(compute_score(y_true_cv, y_pred_cv, score_type=self.scoring))
                yhat_train.append(y_pred_cv)
            yhat_train = torch.cat(yhat_train)
            reg1_scores[time_ind] = torch.nanmean(torch.stack(scores_cv), dim=0)

            # Fit a second regression 
            yhat_all = ridge(X2_train_all, yhat_train, X2_test_all,
                        alpha_start=self.alpha_start,
                        alpha_stop=self.alpha_stop,
                        device=self.device,
                        rotate_x=False)['yhat']
            yhat_ablate = ridge(X2_train_ablate, yhat_train, X2_test_ablate,
                        alpha_start=self.alpha_start,
                        alpha_stop=self.alpha_stop,
                        device=self.device,
                        rotate_x=False)['yhat']

            # Evaluate against y and compute stats
            r2_all = compute_score(y_true, yhat_all, score_type=self.scoring) ** 2
            r2_ablate = compute_score(y_true, yhat_ablate, score_type=self.scoring) ** 2
            reg2_scores[time_ind] = r2_all - r2_ablate
                                               
            perm = perm_uv(yhat_all, yhat_ablate, y_test, n_perm=self.n_perm, score_func=self.score_func)
            var = boot_uv(yhat_all, yhat_ablate, y_test, n_perm=self.n_perm, score_func=self.score_func)
            reg2_scores_null.append(torch.unsqueeze(perm, 2))
            reg2_scores_var.append(torch.unsqueeze(var, 2))
        return reg1_scores, reg2_scores, \
        torch.cat(reg2_scores_null, 2).cpu().detach().numpy(), torch.cat(reg2_scores_var, 2).cpu().detach().numpy()

    def save_results(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, fmri_meta, time_map = self.load_and_validate()
        train, test = self.split(behavior, other_data)
        cv_scores, scores, scores_null, scores_var = self.b2b_regression(train, test)
        results = self.reorganize_results(cv_scores, scores, scores_null, scores_var, fmri_meta, time_map)
        print(results.head())
        self.mk_out_dir()
        self.save_results(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/Back2Back')
    parser.add_argument('--alexnet', '-a', type=str, help='AlexNet activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/AlexNetActivations/alexnet_conv2.npy')
    parser.add_argument('--motion_energy', '-m', type=str, help='Motion energy activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/MotionEnergyActivations/motion_energy.npy')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, or explained_variance')   
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    parser.add_argument('--random_state', type=int, default=0,
                        help='the random state for CV splitting in regression 1')
    parser.add_argument('--feature_to_ablate', '-x', type=str, default='expanse',
                        help='the feature that you want to calculate the unique variance for')
    args = parser.parse_args()
    Back2Back(args).run()


if __name__ == '__main__':
    main()