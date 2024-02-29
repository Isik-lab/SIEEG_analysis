from tqdm import tqdm 
import numpy as np
import pandas as pd
from src import stats
import gc
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import torch 
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from deepjuice.alignment import TorchRidgeGCV


def correlation_scorer(y_true, y_pred):
    return stats.corr(y_true, y_pred)


def feature_scaler(train, test):
    mean_ = torch.mean(train)
    std_ = torch.std(train)
    return (train-mean_)/std_, (test-mean_)/std_


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


def eeg_channel_selection_feature_decoding(neural_df, feature_df,
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
            r, p, r_null = stats.perm(y_hat, y['test'], verbose=False)
            r_var = stats.bootstrap(y_hat, y['test'], verbose=False)

            # append to results
            results.append([time, feature, r, p, 
                            r_null.squeeze(), r_var.squeeze(), 
                            best_k, best_channels])

    # Turn list into dataframe with feature data as categorical
    results = pd.DataFrame(results, columns=['time', 'feature', 'r', 'p', 'r_null', 'r_var', 'best_k', 'best_channels'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results


def eeg_fmri_decoding(neural_df, benchmark, sid, 
                      channels, device, out_file_prefix,
                      verbose=True, scale_y=True,
                      save_whole_brain=False,
                      alphas=[10.**power for power in np.arange(-5, 2)]):
    
    # get fmri in right format and then send to the gpu
    train_inds = benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'train'].reset_index()['index']
    test_inds = benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'test'].reset_index()['index']
    video_list = {'train': benchmark.stimulus_data.loc[train_inds, 'video_name'].tolist(),
                  'test': benchmark.stimulus_data.loc[test_inds, 'video_name'].tolist()}
    y = {'train': torch.from_numpy(benchmark.response_data.to_numpy().T[train_inds]).to(torch.float32).to(device),
         'test': torch.from_numpy(benchmark.response_data.to_numpy().T[test_inds]).to(torch.float32).to(device)}
    if scale_y: 
        y['train'], y['test'] = feature_scaler(y['train'], y['test'])

    # loop through time points
    roi_results = []
    time_groups = neural_df.groupby('time')
    if verbose:
        time_iterator = tqdm(time_groups, desc='Time')
    else:
        time_iterator = time_groups
    for i, (time, time_df) in enumerate(time_iterator):
        # initialize pipe and kfold splitter
        pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device)
        time_ind = str(i).zfill(3)

        # Devide EEG time point into train and test, organize by fMRI, 
        # and send to the GPU
        X = {}
        for stim_set in ['train', 'test']: 
            set_df = time_df.loc[time_df.stimulus_set == stim_set, channels+['video_name']]
            set_df['video_name'] = pd.Categorical(set_df['video_name'], categories=video_list[stim_set], ordered=True)
            set_df = set_df.sort_values(by='video_name').set_index('video_name')
            X[stim_set] = torch.from_numpy(set_df.to_numpy()).to(torch.float32).to(device)
        X['train'], X['test'] = feature_scaler(X['train'], X['test'])

        pipe.fit(X['train'], y['train'])
        y_hat = pipe.predict(X['test'])
        rs = stats.corr2d_gpu(y_hat, y['test']).cpu().detach().numpy()

        if save_whole_brain: 
            # Put the results in each voxel in a dataframe
            whole_brain_results = benchmark.metadata.copy()
            whole_brain_results['time'] = time
            whole_brain_results['eeg_sid'] = sid
            whole_brain_results['r'] = rs
            pd.DataFrame(whole_brain_results).to_csv(f'{out_file_prefix}_time-{time_ind}_whole-brain-decoding.csv.gz', index=False)

        # Get the roi names
        rois = benchmark.metadata.roi_name.unique()
        rois = list(rois[rois != 'none'])

        # filter the data to only the voxels in the rois
        metadata_filtered = benchmark.metadata.loc[benchmark.metadata.roi_name.isin(rois)].reset_index()
        voxel_ids = metadata_filtered['index'].tolist()
        metadata_filtered.drop(columns='index', inplace=True)
        y_hat_filtered = y_hat[:, voxel_ids]
        y_test_filtered = y['test'][:, voxel_ids]
        rs_filtered = rs[voxel_ids]

        # compute the null and bootstrapped distributions in the filtered data
        rs_null = stats.perm_gpu(y_hat_filtered, y_test_filtered).cpu().detach().numpy().T
        rs_var = stats.bootstrap_gpu(y_hat_filtered, y_test_filtered).cpu().detach().numpy().T
        for fmri_sid in metadata_filtered.subj_id.unique(): 
            for roi in rois: 
                inds = metadata_filtered.loc[(metadata_filtered.roi_name == roi) & (metadata_filtered.subj_id == fmri_sid)].reset_index()['index'].tolist()
                roi_results.append({'roi': roi, 'r_null': rs_null[inds].mean(axis=0),
                                    'r_var': rs_var[inds].mean(axis=0), 'r': rs_filtered[inds].mean(),
                                    'eeg_sid': sid, 'time': time, 'fmri_sid': fmri_sid})

        # free up gpu memory
        del rs, rs_null, rs_var, X
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(roi_results).to_pickle(f'{out_file_prefix}_roi-decoding.pkl.gz')


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
