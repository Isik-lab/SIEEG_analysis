import argparse
import pandas as pd
import torch
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eeg.regression import ridge, train_test_split
from eeg.tools import dict_to_tensor, to_torch
from eeg import loading


class FeatureRegression:
    def __init__(self, args):
        self.process = 'FeatureRegression'
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
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

    def reorganize_results(self, scores, time_map):
        results = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(results.columns))]
        results.columns = temp_cols
        results = results.rename(index=time_map).reset_index()
        results = pd.melt(results, id_vars='index')
        cols = list(self.behavior_categories.keys())
        results['feature'] = results.variable.replace({temp_col: feature for feature, temp_col in zip(cols, temp_cols)})
        results = results.rename(columns={'index': 'time'}).drop(columns='variable')
        return results
    
    def standard_regression(self, train, test):
        y_train, y_test, _ = dict_to_tensor(train, test, list(self.behavior_categories.keys()), 
                                            dtype=torch.float32, device=self.device)

        scores = {}
        alpha_values = {}
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict features from EEG', leave=True)
        for time_ind in outer_iterator:
            # First predict the variance in the fMRI by the EEG and predict the result
            X_train, X_test = train['eeg'][time_ind], test['eeg'][time_ind]
            X_train, X_test = to_torch(X_train, self.device), to_torch(X_test, self.device)

            results = ridge(X_train, y_train, 
                            X_test, y_test,
                            alpha_start=self.alpha_start,
                            alpha_stop=self.alpha_stop,
                            scoring=self.scoring,
                            device=self.device,
                            rotate_x=True)
            score = results['score']

            if isinstance(score, torch.Tensor):
                score = score.detach().cpu().numpy()
            scores[time_ind] = score
        return scores

    def save_df(self, results):
        results.to_parquet(self.out_name, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        behavior, other_data, time_map = self.load_and_validate()
        print(behavior.head())
        train, test = train_test_split(behavior, other_data, behavior_categories=self.behavior_categories)
        scores = self.standard_regression(train, test)
        results = self.reorganize_results(scores, time_map)
        print(results.head())
        self.mk_out_dir()
        self.save_df(results)
        print('finished')

def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-03.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/FeatureRegression')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=5,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    args = parser.parse_args()
    FeatureRegression(args).run()


if __name__ == '__main__':
    main()