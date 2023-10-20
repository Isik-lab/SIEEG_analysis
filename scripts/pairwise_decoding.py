#/Applications/anaconda3/envs/nibabel/bin/python
import scipy
from sklearn.discriminant_analysis import _cov
from sklearn.svm import LinearSVC
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import time
from pathlib import Path
import argparse
import pandas as pd


def fit_and_predict(X_train, X_test, y_train, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return np.mean(predictions == y_test)


class PairwiseDecoding:
    def __init__(self, args):
        self.process = 'PairwiseDecoding'
        self.data_dir = args.data_dir
        self.sid = f'subj{str(args.sid).zfill(3)}'
        self.permutation_number = str(args.perm).zfill(2)
        Path(f'{self.data_dir}/{self.process}/{self.sid}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def run(self):
        print('loading data...')
        data = np.load(f'{self.data_dir}/CleanedEEG/{self.sid}/data4rdms_perm-{self.permutation_number}.npz', allow_pickle=True)

        # Take some data out of the dict for faster access
        X = data['X']
        labels_pseudo_train = data['labels_pseudo_train']
        labels_pseudo_test = data['labels_pseudo_test']
        ind_pseudo_train = data['ind_pseudo_train']
        ind_pseudo_test = data['ind_pseudo_test']
        conditions_nCk = data['conditions_nCk']
        videos_nCk = data['videos_nCk']

        # 1. Compute pseudo-trials for training and test
        print('computing pseudo-trials...')
        start = time.time()
        Xpseudo_train = np.full((len(data['train_indices']), data['n_sensors'], data['n_time']), np.nan)
        Xpseudo_test = np.full((len(data['test_indices']), data['n_sensors'], data['n_time']), np.nan)
        for i, ind in tqdm(enumerate(data['train_indices']), total=len(data['train_indices'])):
            Xpseudo_train[i, :, :] = np.mean(X[ind.astype('int'), :, :], axis=0)
        for i, ind in tqdm(enumerate(data['test_indices']), total=len(data['test_indices'])):
            Xpseudo_test[i, :, :] = np.mean(X[ind.astype('int'), :, :], axis=0)
        end = time.time()
        print(f'computing pseudo-trials took {end-start:0f} s.')

        # 2. Whitening using the Epoch method
        print('beginning whitening...')
        start = time.time()
        sigma_conditions = data['labels_pseudo_train'][0, :, data['n_pseudo']-1:].flatten()
        sigma_ = np.empty((data['n_conditions'], data['n_sensors'], data['n_sensors']))
        for c in data['conditions']:
            # compute sigma for each time point, then average across time
            sigma_[c] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, :, t], shrinkage='auto')
                                    for t in range(data['n_time'])], axis=0)
        sigma = sigma_.mean(axis=0)  # average across conditions
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
        Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
        Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
        end = time.time()
        print(f'whitening took {end-start:0f} s.')

        print('computing pairwise distances...')
        results = []
        for t_i, t in tqdm(enumerate(data['time']), total=data['n_time']):
            result_for_t = Parallel(n_jobs=-1)(
                delayed(fit_and_predict)(Xpseudo_train[ind_pseudo_train[c1, c2], :, t_i],
                                         Xpseudo_test[ind_pseudo_test[c1, c2], :, t_i],
                                         labels_pseudo_train[c1, c2],
                                           labels_pseudo_test[c1, c2]) for c1, c2 in conditions_nCk
            )
            for accuracy, (v1, v2) in zip(result_for_t, videos_nCk):
                results.append({'v1': v1, 'v2': v2, 'time': t, 'accuracy': accuracy})

        print('saving...')
        results = pd.DataFrame(results)
        results['perm'] = self.permutation_number
        results.to_csv(f'{self.data_dir}/{self.process}/{self.sid}/rdm_perm-{self.permutation_number}.csv.gz',
                       index=False, compression='gzip')
        print('Finished!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--perm', type=int, default=0)
    parser.add_argument('--data_dir', '-data', type=str, default='../data/interim')
    args = parser.parse_args()
    PairwiseDecoding(args).run()


if __name__ == '__main__':
    main()