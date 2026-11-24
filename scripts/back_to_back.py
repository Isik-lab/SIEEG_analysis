import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import loading, regression, tools, stats
import torch
from pathlib import Path
import numpy as np
from src.stats import compute_score, perm_gpu, bootstrap_gpu
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


class Back2Back:
    def __init__(self, args):
        self.process = 'Back2Back'
        self.y_names = json.loads(args.y_names)
        self.feature = json.loads(args.feature)
        self.roi_mean = args.roi_mean
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.n_perm = args.n_perm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.out_name = f'{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_feature-{'-'.join(self.feature)}.parquet'
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.motion_energy = args.motion_energy
        self.alexnet = args.alexnet
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        fmri, fmri_meta = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        moten = loading.load_model_activations(self.motion_energy)
        alexnet = loading.load_model_activations(self.alexnet)
        
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior, [fmri, alexnet, moten] = loading.check_videos(eeg_raw, behavior, [fmri, alexnet, moten])
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')

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
    
    def reorganize_scores(self, scores, fmri_meta, time_map, out_col):
        df = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(df.columns))]
        df.columns = temp_cols
        df = df.rename(index=time_map).reset_index()
        df = pd.melt(df, id_vars='index')
        df['fmri_subj_id'] = df.variable.replace({temp_col: subj_id for subj_id, temp_col in zip(fmri_meta.subj_id, temp_cols)})
        df['roi_name'] = df.variable.replace({temp_col: roi_name for roi_name, temp_col in zip(fmri_meta.roi_name, temp_cols)})
        df = df.rename(columns={'index': 'time', 'value': out_col}).drop(columns='variable')
        return df.set_index(['fmri_subj_id', 'roi_name', 'time'])
    
    def reorganize_dist(self, dist, dist_name, results):
        dist_df = pd.DataFrame(dist.reshape(self.n_perm, -1).transpose(),
                                columns=[f'{dist_name}_perm_{i}' for i in range(self.n_perm)])
        dist_df[['fmri_subj_id', 'roi_name', 'time']] = results.reset_index()[['fmri_subj_id', 'roi_name', 'time']]
        return dist_df.set_index(['fmri_subj_id', 'roi_name', 'time'])

    def reorganize_results(self, scores, cv_scores, scores_null, scores_var, fmri_meta, time_map):
        results = self.reorganize_scores(scores, fmri_meta,
                                         time_map, 'score')
        cv_results = self.reorganize_scores(cv_scores, fmri_meta,
                                            time_map, 'cv_score')
        results = results.join(cv_results)

        null_df = self.reorganize_dist(scores_null, 'null', results)
        var_df = self.reorganize_dist(scores_var, 'var', results)
        return results.join(null_df).join(var_df).reset_index()
    
    def b2b_regression(self, train, test):
        #Define y
        _, y_test = feature_scaler(train['fmri'], test['fmri'], device=self.device)

        #Define x2
        # Normalize the behavior features
        for feature in self.behavior_categories.keys(): 
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
        # Normalize AlexNet and Moten and reduce the dimensionality
        for feature in ['alexnet', 'moten']:
            train[feature], test[feature] = feature_scaler(train[feature], test[feature], device=self.device)
            train[feature], test[feature], _ = regression.pca_rotation(train[feature], test[feature])
            print(f'{feature} PCs: {train[feature].size()[1]}')
        X2_train, X2_test, _ = dict_to_tensor(train, test, self.feature)

        reg1_scores, reg2_scores = {}, {}
        reg2_scores_null, reg2_scores_var = [], []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc='Back to back regression', leave=True)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X1 = train['eeg'][time_ind]
            yhat_train, scores_cv = [], []
            for train_index, test_index in KFold(n_splits=5).split(X1):
                X1_train, X1_test = feature_scaler(X1[train_index], X1[test_index])
                y_train_cv, y_test_cv = feature_scaler(train['fmri'][train_index], train['fmri'][test_index])
                yhat_cv = ridge(X1_train, y_train_cv, X1_test,  
                                 alpha_start=self.alpha_start,
                                 alpha_stop=self.alpha_stop,
                                 device=self.device,
                                 rotate_x=True)['yhat']
                scores_cv.append(compute_score(y_test_cv, yhat_cv))
                yhat_train.append(yhat_cv)
            yhat_train = torch.cat(yhat_train)
            reg1_scores[time_ind] = torch.nanmean(torch.stack(scores_cv), dim=0)

            # Next predict yhat by the features
            if X2_train.size()[-1] > 1:
                yhat = ridge(X2_train, yhat_train, X2_test,
                            alpha_start=self.alpha_start,
                            alpha_stop=self.alpha_stop,
                            device=self.device,
                            rotate_x=False)['yhat']
            else:
                yhat = ols(X2_train, yhat_train, X2_test,
                           rotate_x=False)['yhat']

            # Evaluate against y and compute stats
            reg2_scores[time_ind] = compute_score(y_test, yhat)
            reg2_scores_null.append(torch.unsqueeze(perm_gpu(y_test, yhat, n_perm=self.n_perm), 2))
            reg2_scores_var.append(torch.unsqueeze(bootstrap_gpu(y_test, yhat, n_perm=self.n_perm), 2))
        return reg1_scores, reg2_scores, \
               torch.cat(reg2_scores_null, 2).cpu().detach().numpy(), \
               torch.cat(reg2_scores_var, 2).cpu().detach().numpy()

    def save_results(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, fmri_meta, time_map = self.load_and_validate()
        train, test = self.split(behavior, other_data)
        cv_scores, scores, scores_null, scores_var = self.b2b_regression(train, test)
        results = self.reorganize_results(scores, cv_scores, 
                                          scores_null, scores_var,
                                          fmri_meta, time_map)
        print(results.head())
        self.mk_out_dir()
        self.save_results(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/Back2Back')
    parser.add_argument('--alexnet', '-a', type=str, help='AlexNet activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/AlexNetActivations/alexnet_conv2.npy')
    parser.add_argument('--motion_energy', '-m', type=str, help='Motion energy activation file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/MotionEnergyActivations/motion_energy.npy')
    parser.add_argument('--y_names', '-y', type=str, default='["fmri"]',
                        help='a list of data names to be used as regression target')
    parser.add_argument('--feature', '-x', type=str, default='["alexnet"]',
                        help='a list of data names for fitting the second regression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=30,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    parser.add_argument('--n_perm', type=int, default=5000,
                        help='the number of permutations for stats')
    args = parser.parse_args()
    Back2Back(args).run()


if __name__ == '__main__':
    main()