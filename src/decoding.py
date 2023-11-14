from tqdm import tqdm 
import numpy as np
import pandas as pd
from src import stats

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import torch 


def correlation_scorer(y_true, y_pred):
    return stats.corr(y_true, y_pred)


class Benchmark:
    def __init__(self, metadata, stimulus_data, response_data):
        self.metadata = metadata
        self.stimulus_data = stimulus_data
        self.response_data = response_data

    def filter_rois(self, rois):
        self.metadata = self.metadata.loc[self.metadata.roi_name.isin(rois)].reset_index()
        voxel_id = self.metadata['voxel_id'].to_numpy()
        self.response_data = self.response_data.iloc[voxel_id]
    
    def filter_subjids(self, subj_ids):
        self.metadata = self.metadata.loc[self.metadata.subj_id.isin(subj_ids)].reset_index()
        voxel_id = self.metadata['voxel_id'].to_numpy()
        self.response_data = self.response_data.iloc[voxel_id]



def eeg_feature_decoding(neural_df, feature_df,
                          features, channels, verbose=True):
    # initialize pipe and kfold splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scorer = make_scorer(correlation_scorer, greater_is_better=True)
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('rcv', RidgeCV(
            fit_intercept=False,
            alphas=[10.**power for power in np.arange(-5, 2)],
            alpha_per_target=True,
            scoring=scorer
        ))
    ])

    results = []
    time_groups = neural_df.groupby('time')

    if verbose:
        iterator = tqdm(time_groups, desc='Time')
    else:
        iterator = time_groups

    for time, time_df in iterator:
        X = time_df[channels].to_numpy()
        y = feature_df[features].to_numpy()

        y_pred = []
        y_true = []
        for train_index, test_index in cv.split(X):
            pipe.fit(X[train_index], y[train_index])
            y_pred.append(pipe.predict(X[test_index]))
            y_true.append(y[test_index])
        rs = stats.corr2d(np.concatenate(y_pred), np.concatenate(y_true))
        for feature, r in zip(features, rs): 
            results.append([time, feature, r])

    results = pd.DataFrame(results, columns=['time', 'feature', 'r'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results


def eeg_fmri_decoding(feature_map, benchmark, channels, device,
                      verbose=True, n_splits=4):
    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    alphas = [10.**power for power in np.arange(-5, 2)]
    score_func = make_scorer(correlation_scorer)
    if 'cuda' in device.type:
        from deepjuice.alignment import TorchRidgeCV, get_scorer
        score_func = get_scorer('pearsonr')
        pipe = TorchRidgeCV(alphas=alphas, device=device, scale_X=True,)
    else:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('rcv', RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                alpha_per_target=True,
                scoring=score_func
            ))
        ])

    results = []
    time_groups = feature_map.groupby('time')

    if verbose:
        time_iterator = tqdm(time_groups, desc='Time')
    else:
        time_iterator = time_groups

    results = []
    for time, time_df in time_iterator:
        X = time_df[channels].to_numpy()
        y = benchmark.response_data.to_numpy().T
        if 'cuda' in device.type:
            X = X.to(torch.float32).to(device)
            y = y.T.to(torch.float32).to(device)

        y_pred = []
        y_true = []

        if verbose:
            cv_iterator = tqdm(cv.split(X), desc='CV fold', total=n_splits)
        else:
            cv_iterator = cv.split(X)

        for train_index, test_index in cv_iterator:
            pipe.fit(X[train_index], y[train_index])
            y_pred.append(pipe.predict(X[test_index]).cpu().detach().numpy())
            y_true.append(y[test_index].cpu().detach().numpy())
        
        if 'cuda' in device.type:
            scores = score_func(np.concatenate(y_pred), np.concatenate(y_true))
        else:
            scores = correlation_scorer(np.concatenate(y_pred), np.concatenate(y_true))

        for region in benchmark.metadata.roi_name.unique():
            for subj_id in benchmark.metadata.subj_id.unique():
                voxel_id = benchmark.metadata.loc[(benchmark.metadata.subj_id == subj_id) &
                                                   (benchmark.metadata.roi_name == region), 'voxel_id'].to_numpy()
                results.append({'time': time,
                                'roi_name': region,
                                'subj_id': subj_id,
                                'score': np.mean(scores[voxel_id])})
    return pd.DataFrame(results)


def gaze_feature_decoding(X, feature_df, features):
    # initialize pipe and kfold splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scorer = make_scorer(correlation_scorer, greater_is_better=True)
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('rcv', RidgeCV(
            fit_intercept=False,
            alphas=[10.**power for power in np.arange(-5, 2)],
            alpha_per_target=True,
            scoring=scorer
        ))
    ])
    
    y = feature_df[features].to_numpy()

    y_pred = []
    y_true = []
    for train_index, test_index in cv.split(X):
        pipe.fit(X[train_index], y[train_index])
        y_pred.append(pipe.predict(X[test_index]))
        y_true.append(y[test_index])
    rs = stats.corr2d(np.concatenate(y_pred), np.concatenate(y_true))

    results = []
    for feature, r in zip(features, rs): 
        results.append([feature, r])
    return pd.DataFrame(results, columns=['feature', 'r'])
