import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eeg import loading
from eeg.regression import banded_ridge, train_test_split
from eeg.tools import dict_to_tensor


class JointRegression:
    def __init__(self, args):
        self.process = 'JointRegression'
        self.roi_mean = args.roi_mean
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        if self.roi_mean:
            self.out_name = f"{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_rois.parquet"
        elif not self.roi_mean:
            self.out_name = f"{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_full-brain"
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'scene_object': ['rating-expanse', 'rating-object'],
                                    'social_primitive': ['rating-agent_distance', 'rating-facingness'],
                                    'social_interaction': ['rating-joint_action', 'rating-communication'],
                                    'affective': ['rating-valence', 'rating-arousal']}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        fmri, fmri_meta = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        
        # Check EEG trials 
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_raw = eeg_raw.groupby(['channel', 'time', 'video_name']).mean(numeric_only=True)
        eeg_raw = eeg_raw.reset_index().drop(columns=['trial', 'repitition', 'even'])
        eeg_filtered, behavior, [fmri] = loading.check_videos(eeg_raw, behavior, [fmri])
        eeg_filtered['time_ind'] = eeg_filtered['time_ind'].astype('int')
        
        # Transform EEG to dict 
        eeg = {}
        iterator = tqdm(eeg_filtered.groupby('time_ind'), total=eeg_filtered.time_ind.nunique(), desc='EEG to numpy')
        time_map = {}
        for time_ind, time_df in iterator:
            eeg[time_ind] = loading.strip_eeg(time_df)
            time_map[time_ind] = time_df.time.unique()[0]
        return behavior, {'eeg': eeg, 'fmri': fmri}, fmri_meta, time_map

    def reorganize_results(self, scores, fmri_meta, time_map, scores_null=None, scores_var=None):
        results = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(results.columns))]
        results.columns = temp_cols
        results = results.rename(index=time_map).reset_index()
        results = pd.melt(results, id_vars='index')
        results['fmri_subj_id'] = results.variable.replace({temp_col: subj_id for subj_id, temp_col in zip(fmri_meta.subj_id, temp_cols)})
        results['roi_name'] = results.variable.replace({temp_col: roi_name for roi_name, temp_col in zip(fmri_meta.roi_name, temp_cols)})
        results = results.rename(columns={'index': 'time'}).drop(columns='variable')

        if scores_null is not None and scores_var is not None:
            scores_null_df = pd.DataFrame(scores_null.reshape(self.n_perm, -1).transpose(),
                                    columns=[f'null_perm_{i}' for i in range(self.n_perm)])
            scores_var_df = pd.DataFrame(scores_var.reshape(self.n_perm, -1).transpose(),
                                    columns=[f'var_perm_{i}' for i in range(self.n_perm)])
            scores_null_df[['fmri_subj_id', 'roi_name', 'time']] = results[['fmri_subj_id', 'roi_name', 'time']]
            scores_var_df[['fmri_subj_id', 'roi_name', 'time']] = results[['fmri_subj_id', 'roi_name', 'time']]
            scores_null_df.set_index(['fmri_subj_id', 'roi_name', 'time'], inplace=True)
            scores_var_df.set_index(['fmri_subj_id', 'roi_name', 'time'], inplace=True)
            results = results.set_index(['fmri_subj_id', 'roi_name', 'time']).join(scores_null_df).join(scores_var_df).reset_index()

        return results
    
    def regression(self, train, test, fmri_meta, time_map):
        #Define y
        y_train, y_test, _ = dict_to_tensor(train, test, ['fmri'], device=self.device)

        def to_scalar(value):
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
                return value.squeeze().item()
            return value

        scores = []
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict fMRI from EEG', leave=True)
        for time_ind in outer_iterator:
            behavior_train, behavior_test, groups = dict_to_tensor(train, test, list(self.behavior_categories.keys()),
                                                 device=self.device)
            train_eeg, test_eeg = train['eeg'][time_ind], test['eeg'][time_ind]
            if not torch.is_tensor(train_eeg):
                train_eeg = torch.from_numpy(train_eeg)
            if not torch.is_tensor(test_eeg):
                test_eeg = torch.from_numpy(test_eeg)
            train_eeg = train_eeg.to(self.device)
            test_eeg = test_eeg.to(self.device)
            if torch.is_tensor(behavior_train):
                behavior_train = behavior_train.to(self.device)
            if torch.is_tensor(behavior_test):
                behavior_test = behavior_test.to(self.device)
            if torch.is_tensor(groups):
                groups = groups.to(self.device)
            n_groups = len(self.behavior_categories)
            if torch.is_tensor(train_eeg):
                n_eeg_features = train_eeg.size(1)
            else:
                n_eeg_features = train_eeg.shape[1]
            groups = torch.concat([groups, torch.ones(n_eeg_features, dtype=torch.int32, device=self.device) * n_groups], dim=0)
            X_train = torch.concat([train_eeg, behavior_train], dim=1)
            X_test = torch.concat([test_eeg, behavior_test], dim=1)

            output = banded_ridge(X_train, y_train, X_test,
                                  y_test, groups,   
                                  alpha_start=self.alpha_start,
                                  alpha_stop=self.alpha_stop,
                                  device=self.device,
                                  rotate_x=False)

            for i_roi, mri_df in enumerate(fmri_meta.itertuples()):
                subj_id = mri_df.subj_id
                roi_name = mri_df.roi_name
                scores.append({'time': time_map[time_ind],
                                   'fmri_subj_id': subj_id,
                                   'roi_name': roi_name,
                                   'feature': 'total',
                                   'score': to_scalar(output['r2'][i_roi])})
                group_labels = list(self.behavior_categories.keys()) + ['eeg']
                for i_group, group in enumerate(group_labels):
                    r2_split = output['r2_split']
                    score_val = to_scalar(r2_split[i_group, i_roi])
                    scores.append({'time': time_map[time_ind],
                                   'fmri_subj_id': subj_id,
                                   'roi_name': roi_name,
                                   'feature': group,
                                   'score': score_val})
        return pd.DataFrame(scores)

    def save_df(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, fmri_meta, time_map = self.load_and_validate()
        train, test = train_test_split(behavior, other_data,
                                       behavior_categories=self.behavior_categories)
        self.mk_out_dir()
        scores = self.regression(train, test, fmri_meta, time_map)
        self.save_df(pd.DataFrame(scores))
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-06.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/JointRegression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=5,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    args = parser.parse_args()
    JointRegression(args).run()


if __name__ == '__main__':
    main()