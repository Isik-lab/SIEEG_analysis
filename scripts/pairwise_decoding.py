#/Applications/anaconda3/envs/nibabel/bin/python
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from itertools import combinations


def divide_into_groups(arr, n_groups=5):
    n = len(arr)
    
    # Calculate the size of each group
    group_size = n // n_groups
    remainder = n % n_groups
    
    groups = []
    start_idx = 0
    
    for i in range(n_groups):
        end_idx = start_idx + group_size + (i < remainder)  # Add 1 if this group takes an extra element
        group = arr[start_idx:end_idx]
        groups.append(group)
        start_idx = end_idx  # Update the starting index for the next group
        
    return groups


def generate_pseudo(arr, n_groups=5):
    inds = np.arange(len(arr))
    np.random.shuffle(inds)
    groups = divide_into_groups(inds, n_groups)
    pseudo_arr = []
    for group in groups:
        pseudo_arr.append(arr[group].mean(axis=0))
    return np.array(pseudo_arr)


def fit_and_predict(video1_array, video2_array, n_groups):
    #regenerate to ensure that it is thread safe
    logo = LeaveOneGroupOut()
    pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression())])

    #define X, y, and groups to loop through
    X = np.vstack([generate_pseudo(video1_array, n_groups),
                    generate_pseudo(video2_array, n_groups)])
    y = np.hstack([np.zeros((n_groups)), np.ones((n_groups))]).astype('int')
    groups = np.concatenate([np.arange(n_groups), np.arange(n_groups)])

    #fit and predict 
    y_pred = []
    y_true = []
    for train_index, test_index in logo.split(X, y, groups=groups):
        pipe.fit(X[train_index], y[train_index])
        y_pred.append(pipe.predict(X[test_index]))
        y_true.append(y[test_index])
        
    #Return the mean prediction acurracy over all groups
    return np.mean(np.array(y_pred) == np.array(y_true))


class PairwiseDecoding:
    def __init__(self, args):
        self.process = 'PairwiseDecoding'
        self.data_dir = args.data_dir
        self.sid = f'subj{str(args.sid).zfill(3)}'
        self.n_groups = 5
        Path(f'{self.data_dir}/{self.process}/{self.sid}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def run(self):
        print('loading data...')
        df_filtered = pd.read_csv(f'{self.data_dir}/EEG_RSA/{self.sid}_raw-filtered.csv')
        df_filtered.sort_values(['time', 'video_name'], inplace=True)
        videos = df_filtered.video_name.unique()
        videos_nCk = list(combinations(videos, 2))
        channels = df_filtered.drop(columns=['time', 'trial', 'video_name']).columns
        print(channels)

        results = []
        time_groups = df_filtered.groupby('time')
        for time, time_df in tqdm(time_groups, total=len(time_groups)):
            result_for_t = Parallel(n_jobs=-1)(
                delayed(fit_and_predict)(time_df.loc[time_df.video_name == video1, channels].to_numpy(),
                                        time_df.loc[time_df.video_name == video2, channels].to_numpy(),
                                        self.n_groups) for video1, video2 in videos_nCk
            )
            for accuracy, (video1, video2) in zip(result_for_t, videos_nCk):
                results.append([time, video1, video2, accuracy])
        results = pd.DataFrame(results, columns=['time', 'video1', 'video2', 'accuracy'])
        results.to_csv(f'{self.data_dir}/EEG_RSA/{self.sid}_pairwise-decoding.csv', index=False)
        print('Finished!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--perm', type=int, default=0)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data/interim')
    args = parser.parse_args()
    PairwiseDecoding(args).run()


if __name__ == '__main__':
    main()