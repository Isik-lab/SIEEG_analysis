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


def eeg_feature_decoding(neural_df, feature_df,
                          features, channels):
    """
        inputs:
            neural_df: pd.DataFrame of the EEG data in long format
            feature_df: pd.DataFrame of videos by the annotated features
            features: list of the features to predict in desired categorical order
            channels: a list of channels present in the EEG data.
                This can vary between participants depending on preprocessing. 
        output:
            results: pd.DataFrame containing the Ridge results for each feature at each time point

    """
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
        # Split the data into train and test sets
        X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
             'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}
        y = {'train': feature_df.loc[feature_df.stimulus_set == 'train', features].to_numpy(),
             'test': feature_df.loc[feature_df.stimulus_set == 'test', features].to_numpy()}

        # Perform the regression
        pipe.fit(X['train'], y['train'])
        y_hat = pipe.predict(X['test'])

        # Compute significance and variance
        rs, ps, rs_null = stats.perm(y_hat, y['test'], verbose=False)
        rs_var = stats.bootstrap(y_hat, y['test'], verbose=False)

        # Append to the results
        for feature, r, p, r_null, r_var in zip(features, rs, ps, rs_null.T, rs_var.T): 
            results.append([time, feature, r, p, r_null, r_var])

    # Turn list into dataframe with feature data as categorical
    results = pd.DataFrame(results, columns=['time', 'feature', 'r', 'p', 'r_null', 'r_var'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results

def eeg_fmri_decoding(feature_map, benchmark,
                       channels, device,
                      verbose=True):
    from deepjuice.alignment import TorchRidgeGCV

    # initialize pipe and kfold splitter
    alphas = [10.**power for power in np.arange(-5, 2)]
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)
    
    # get fmri in right format and then send to the gpu
    train_response_data, _ = benchmark.filter_stimulus(stimulus_set='train')
    test_response_data, _ = benchmark.filter_stimulus(stimulus_set='test')
    y = {'train': train_response_data.to_numpy().T,
         'test': test_response_data.to_numpy().T}
    y = {key: torch.from_numpy(val).to(torch.float32).to(device) for key, val in y.items()}

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
        X = {key: torch.from_numpy(val).to(torch.float32).to(device) for key, val in X.items()}

        pipe.fit(X['train'], y['train'])
        y_hat = pipe.predict(X['test'])
        rs, rs_null = stats.perm_gpu(y_hat, y['test'])
        rs_var = stats.bootstrap_gpu(y_hat, y['test'])

        rs, rs_null = rs.cpu().detach().numpy(), rs_null.cpu().detach().numpy()
        rs_var = rs_var.cpu().detach().numpy()
        for (i, row), (r, r_null, r_var) in zip(benchmark.metadata.iterrows(), zip(rs, rs_null.T, rs_var.T)): 
            row['r'] = r[i]
            row['r_null'] = r_null[i]
            row['r_var'] = r_var[i]
            row['time'] = time
            results.append(row)

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
