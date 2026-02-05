#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import pickle
import argparse

from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import numpy as np


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eeg import loading


def video_arr(df, video):
    video_df = df.loc[df['video_name'] == video]
    return video_df.pivot(index='repitition', columns='channel', values='signal').to_numpy()

def shuffle_along_axis(arr, axis):
    idx = np.arange(arr.shape[axis])
    np.random.shuffle(idx)
    return arr.take(idx, axis=axis)

def pseudotrials(arr, n_splits=4):
    """Create pseudotrials by averaging over splits along axis 0."""
    shuffled_arr = shuffle_along_axis(arr, axis=0)
    split_arrays = np.array_split(shuffled_arr, n_splits, axis=0)
    pseudotrial_arr = np.array([np.mean(split, axis=0) for split in split_arrays])
    return pseudotrial_arr

def decode_distance(arr1, arr2, n_splits=10,
                    n_pseudo_splits=4):
    """Compute the decoding distance between two arrays."""
    pipeline = make_pipeline(
            StandardScaler(),
            LinearSVC()
    )
    kf = KFold(n_splits=2, shuffle=True)
    accuracies = []

    for split in range(n_splits):
        pseudo_arr1 = pseudotrials(arr1, n_splits=n_pseudo_splits)
        pseudo_arr2 = pseudotrials(arr2, n_splits=n_pseudo_splits)

        for train_index, test_index in kf.split(pseudo_arr1):
            X1_train, X1_test = pseudo_arr1[train_index], pseudo_arr1[test_index]
            X2_train, X2_test = pseudo_arr2[train_index], pseudo_arr2[test_index]
            X_train = np.vstack([X1_train, X2_train])
            X_test = np.vstack([X1_test, X2_test])
            y_train = np.array([0]*len(X1_train) + [1]*len(X2_train))
            y_test = np.array([0]*len(X1_test) + [1]*len(X2_test))

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            accuracies.append(accuracy)

    return 1 - np.mean(accuracies)

class eegRDMs:
    def __init__(self, args):
        self.process = 'eegRDMs'
        self.data_split = args.data_split
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        print(vars(self))
        self.eeg_file = args.eeg_file
        self.out_dir = args.out_dir
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)


    def neural_distance(self):
        eeg = loading.load_eeg(self.eeg_file)
        eeg = eeg[eeg.stimulus_set == self.data_split].reset_index(drop=True)
        videos = sorted(eeg.video_name.unique().tolist())
        results = dict()
        for time, time_df in tqdm(eeg.groupby('time'), total=eeg.time.nunique(), desc='Calculating RDMs'):
            rdm = []
            for v1, v2 in list(combinations(videos, 2)):
                arr1 = video_arr(time_df, v1)
                arr2 = video_arr(time_df, v2)
                rdm.append(decode_distance(arr1, arr2))
            results[time] = rdm
        return results

    def save(self, rdms):
        out_file = f'{self.out_dir}/{self.sid}_set-{self.data_split}.pkl'
        with open(out_file, 'wb') as f: # Open in write binary mode ('wb')
            pickle.dump(rdms, f, pickle.HIGHEST_PROTOCOL)

    def run(self):
        rdms = self.neural_distance()
        self.save(rdms)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='2')
    parser.add_argument('--data_split', '-d', type=str, default='test')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/eegReliability')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/ReorganizeEEG/all_trials/sub-02.parquet')
    args = parser.parse_args()
    eegRDMs(args).run()


if __name__ == '__main__':
    main()
