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
from deepjuice.alignment import TorchRidgeGCV


def correlation_scorer(y_true, y_pred):
    return stats.corr(y_true, y_pred)


def eeg_feature_decoding(neural_df, feature_df,
                          features, channels):
    # initialize pipe and kfold splitter
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

    iterator = tqdm(time_groups, desc='Time')
    for time, time_df in iterator:
        X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
             'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}
        y = {'train': feature_df.loc[feature_df.stimulus_set == 'train', features].to_numpy(),
             'test': feature_df.loc[feature_df.stimulus_set == 'test', features].to_numpy()}

        pipe.fit(X['train'], y['train'])
        rs, ps, _ = stats.perm(pipe.predict(X['test']), y['test'])
        for feature, (r, p) in zip(features, zip(rs, ps)): 
            results.append([time, feature, r, p])

    results = pd.DataFrame(results, columns=['time', 'feature', 'r', 'p'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results


def eeg_fmri_decoding(feature_map, benchmark,
                       channels, device,
                      verbose=True):
    # initialize pipe and kfold splitter
    alphas = [10.**power for power in np.arange(-5, 2)]
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

    results = []
    time_groups = feature_map.groupby('time')

    if verbose:
        time_iterator = tqdm(time_groups, desc='Time')
    else:
        time_iterator = time_groups

    results = []
    for time, time_df in time_iterator:
        X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
             'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}
        train_idx = benchmark.stimulus_data.index[benchmark.stimulus_data['stimulus_set'] == 'train'].to_list()
        test_idx = benchmark.stimulus_data.index[benchmark.stimulus_data['stimulus_set'] == 'test'].to_list()
        y = {'train': benchmark.response_data.to_numpy().T[train_idx],
             'test': benchmark.response_data.to_numpy().T[test_idx]}
        
        X = {key: torch.from_numpy(val).to(torch.float32).to(device) for key, val in X.items()}
        y = {key: torch.from_numpy(val).to(torch.float32).to(device) for key, val in y.items()}

        pipe.fit(X['train'], y['train'])
        scores, null_scores = stats.perm_gpu(pipe.predict(X['test']), y['test'], verbose=True)

        for region in benchmark.metadata.roi_name.unique():
            voxel_id = benchmark.metadata.loc[(benchmark.metadata.roi_name == region), 'voxel_id'].to_numpy()
            results.append({'time': time, 
                            'roi_name': region,
                            'score': torch.mean(scores[voxel_id]).cpu().detach().numpy(),
                            'score_null': torch.mean(null_scores[:, voxel_id], dim=1).cpu().detach().numpy(),
                            'method': 'ridge'})
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
