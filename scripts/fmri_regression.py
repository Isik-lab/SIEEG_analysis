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



class fMRIRegression:
    def __init__(self, args):
        self.process = 'fMRIRegression'
        self.roi_mean = args.roi_mean
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = args.out_dir
        self.eeg_file = args.eeg_file
        self.smoothing = args.smoothing
        if self.roi_mean:
            self.out_name = f"{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_rois.parquet"
        elif not self.roi_mean:
            self.out_name = f"{self.out_dir}/{self.eeg_file.split('/')[-1].split('.parquet')[0]}_full-brain.parquet"
        print(vars(self)) 
        self.fmri_dir = args.fmri_dir
        self.behavior_categories = {'expanse': 'rating-expanse', 'object': 'rating-object',
                                    'agent_distance': 'rating-agent_distance', 'facingness': 'rating-facingness',
                                    'joint_action': 'rating-joint_action', 'communication': 'rating-communication',
                                    'valence': 'rating-valence', 'arousal': 'rating-arousal'}

    def load_and_validate(self):
        behavior = loading.load_behavior(self.fmri_dir)
        fmri, fmri_meta = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean, smoothing=self.smoothing)
        
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

    def reorganize_results(self, scores, fmri_meta, time_map):
        results = pd.DataFrame(scores).transpose()
        temp_cols = [f'col{i}' for i in range(len(results.columns))]
        results.columns = temp_cols
        results = results.rename(index=time_map).reset_index()
        results = pd.melt(results, id_vars='index')
        results['fmri_subj_id'] = results.variable.replace({temp_col: subj_id for subj_id, temp_col in zip(fmri_meta.subj_id, temp_cols)})
        results['roi_name'] = results.variable.replace({temp_col: roi_name for roi_name, temp_col in zip(fmri_meta.roi_name, temp_cols)})
        results = results.rename(columns={'index': 'time'}).drop(columns='variable')
        return results
    
    def standard_regression(self, train, test):
        #Define y
        y_train, y_test, _ = dict_to_tensor(train, test, ['fmri'], 
                                            dtype=torch.float32, device=self.device)
        scores = {}
        outer_iterator = tqdm(train['eeg'].keys(), total=len(train['eeg']),
                              desc=f'Predict fMRI from EEG', leave=True)
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

    def save_indiv_times(self, scores, fmri_meta):
        for time_ind, score in scores.items():
            results = pd.DataFrame(score.cpu().detach().numpy(),
                                   index=fmri_meta['voxel_id'],
                                   columns=['value'])
            if time_ind == 0:
                print(results.head())
            results.to_parquet(f'{self.out_name}_time-{str(time_ind).zfill(3)}.parquet')

    def run(self):
        behavior, other_data, fmri_meta, time_map = self.load_and_validate()
        train, test = train_test_split(behavior, other_data, behavior_categories=self.behavior_categories)
        scores = self.standard_regression(train, test)
        self.mk_out_dir()
        if self.roi_mean:
            results = self.reorganize_results(scores, fmri_meta, time_map)
            print(results.head())
            self.save_df(results)
        else:
            results = self.save_indiv_times(scores, fmri_meta)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizeEEG/all_trials/sub-01.parquet')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/fMRIRegression')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--smoothing', action=argparse.BooleanOptionalAction, default=False,
                        help='smooth the voxelwise response')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=5,
                        help='stopping value in log space for the ridge alpha penalty')      
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. Options are pearsonr, r2_score, r2_adj, or explained_variance')     
    args = parser.parse_args()
    fMRIRegression(args).run()


if __name__ == '__main__':
    main()