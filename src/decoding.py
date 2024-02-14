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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

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
        ('select', SelectKBest(score_func=f_regression)),
        ('ridge', RidgeCV(
            fit_intercept=False,
            alphas=[10.**power for power in np.arange(-5, 2)],
            scoring=scorer
        ))
    ])
    # Set up the grid search with cross-validation
    param_grid = {'select__k': np.arange(1, len(channels) + 1)}
    grid_search = GridSearchCV(pipe, param_grid, scoring=scorer, cv=4, n_jobs=-1)

    results = []
    time_groups = neural_df.groupby('time')
    for time, time_df in tqdm(time_groups, desc='Time'):
        # Split the data into train and test sets
        X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
             'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}

        for feature in features: 
            y = {'train': feature_df.loc[feature_df.stimulus_set == 'train', feature].to_numpy(),
                'test': feature_df.loc[feature_df.stimulus_set == 'test', feature].to_numpy()}

            # Fit the grid search to the data
            grid_search.fit(X['train'], y['train'])

            # Evaluate the best model found on the test set
            best_model = grid_search.best_estimator_
            y_hat = best_model.predict(X['test'])

            # save the best channels and number of channels
            channel_indices = np.where(best_model.named_steps['select'].get_support())[0]
            best_channels = np.array(channels)[channel_indices]
            best_k = grid_search.best_params_['select__k']

            # Compute significance and variance
            r, p, r_null = stats.perm(y_hat, y['test'], verbose=False, square=False)
            r_var = stats.bootstrap(y_hat, y['test'], verbose=False, square=False)

            # append to results
            results.append([time, feature, r, p, 
                            r_null.squeeze(), r_var.squeeze(), 
                            best_k, best_channels])

    # Turn list into dataframe with feature data as categorical
    results = pd.DataFrame(results, columns=['time', 'feature', 'r', 'p', 'r_null', 'r_var', 'best_k', 'best_channels'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results


def eeg_fmri_decoding(feature_map, benchmark,
                       channels, device, out_file_prefix,
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

    time_groups = feature_map.groupby('time')

    if verbose:
        time_iterator = tqdm(time_groups, desc='Time')
    else:
        time_iterator = time_groups

    for time, time_df in time_iterator:
        X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
             'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}
        X = {key: torch.from_numpy(val).to(torch.float32).to(device) for key, val in X.items()}

        pipe.fit(X['train'], y['train'])
        y_hat = pipe.predict(X['test'])
        rs, rs_null = stats.perm_gpu(y_hat, y['test'])
        rs_var = stats.bootstrap_gpu(y_hat, y['test'])

        # move from torch to numpy
        rs_cpu = rs.cpu().detach().numpy()
        rs_null_cpu = rs_null.cpu().detach().numpy().T
        rs_var_cpu = rs_var.cpu().detach().numpy().T

        # free up gpu memory
        del rs, rs_null, rs_var, X
        torch.cuda.empty_cache()

        # put the results in a data frame
        results = []
        for (i, row), (r, r_null, r_var) in zip(benchmark.metadata.iterrows(),
                                                zip(rs_cpu, rs_null_cpu, rs_var_cpu)): 
            print(f'{r_null.shape}')
            row['r'] = r
            row['r_null'] = r_null
            row['r_var'] = r_var
            row['time'] = time
            results.append(row)
        results = pd.DataFrame(results)
        results.to_pickle(f'{out_file_prefix}_time-{time}_decoding.pkl.gz')


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
