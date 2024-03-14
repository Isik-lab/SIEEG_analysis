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
                      channels, feature_categories,
                      device, out_file_prefix,
                      verbose=True, scale_y=True,
                      save_whole_brain=False,
                      alphas=[10.**power for power in np.arange(-5, 2)]):
    
    # get fmri in right format and then send to the gpu
    inds = {'train': benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'train'].reset_index()['index'], 
            'test': benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'test'].reset_index()['index']}
    video_list = {'train': benchmark.stimulus_data.loc[inds['train'], 'video_name'].tolist(),
                  'test': benchmark.stimulus_data.loc[inds['test'], 'video_name'].tolist()}
    y = {'train': torch.from_numpy(benchmark.response_data.to_numpy().T[inds['train']]).to(torch.float32).to(device),
         'test': torch.from_numpy(benchmark.response_data.to_numpy().T[inds['test']]).to(torch.float32).to(device)}
    if scale_y: 
        y['train'], y['test'] = feature_scaler(y['train'], y['test'])

    # get the voxel indices just belonging to ROIs
    rois = benchmark.metadata.roi_name.unique() # Get the roi names
    rois = list(rois[rois != 'none']) # Remove 'none' from the ROI list
    metadata_filtered = benchmark.metadata.loc[benchmark.metadata.roi_name.isin(rois)].reset_index()
    voxel_ids = metadata_filtered['index'].tolist()
    metadata_filtered.drop(columns='index', inplace=True)
    y_test_filtered = y['test'][:, voxel_ids] # filter y_test just to the reliable voxels

    # loop through feature categories and time
    out = []
    time_groups = neural_df.groupby('time')
    if verbose:
        feature_iterator = tqdm(enumerate(feature_categories.items()), total=len(feature_categories), desc='Categories', leave=True)
        time_iterator = tqdm(time_groups, desc='Time')
    else:
        feature_iterator = feature_categories
        time_iterator = time_groups

    # Feature category loop
    for i_cat, (category, feature_list) in feature_iterator:
        X = {'features': {}}
        for stim_set in ['train', 'test']: 
            arr = benchmark.stimulus_data.iloc[inds[stim_set]][feature_list].to_numpy()
            X['features'][stim_set] = torch.from_numpy(arr).to(torch.float32).to(device)
        X['features']['train'], X['features']['test'] = feature_scaler(X['features']['train'], X['features']['test'])

        # Compute the regression for the features
        pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                    device=device, scale_X=False)
        pipe.fit(X['features']['train'], y['train'])

        # Make the prediction and store in a dict 
        y_hat = {'features': pipe.predict(X['features']['test'])}

        # Score the prediction and store in a dict
        statistics = {'r2': {}, 'r2_filtered': {}, 'r2_null': {}, 'r2_var': {}}
        statistics['r2']['features'] = stats.corr2d_gpu(y_hat['features'], y['test']).cpu().detach().numpy() ** 2
        statistics['r2_filtered']['features'] = statistics['r2']['features'][voxel_ids]
        
        # Time loop
        for i_time, (time, time_df) in enumerate(time_iterator):
            # initialize pipe and kfold splitter
            time_ind = str(i_time).zfill(3)

            # Devide EEG time point into train and test, organize by fMRI, and send to the GPU
            X['eeg'] = {}
            for stim_set in ['train', 'test']: 
                set_df = time_df.loc[time_df.stimulus_set == stim_set, channels+['video_name']]
                set_df['video_name'] = pd.Categorical(set_df['video_name'], categories=video_list[stim_set], ordered=True)
                set_df = set_df.sort_values(by='video_name').set_index('video_name')
                X['eeg'][stim_set] = torch.from_numpy(set_df.to_numpy()).to(torch.float32).to(device)
            X['eeg']['train'], X['eeg']['test'] = feature_scaler(X['eeg']['train'], X['eeg']['test'])

            # Create also the values for the combination of the EEG and feature data
            X['full'] = {}
            for stim_set in ['train', 'test']: 
                X['full'][stim_set] = torch.cat([X['eeg'][stim_set], X['features'][stim_set]], dim=1)

            # Fit the regression for EEG and full model
            for regression_type in ['eeg', 'full']: 
                pipe.fit(X[regression_type]['train'], y['train']) 
                y_hat[regression_type] = pipe.predict(X[regression_type]['test'])
                r2 = stats.corr2d_gpu(y_hat[regression_type], y['test']) ** 2
                statistics['r2'][regression_type] = r2.cpu().detach().numpy()

            # Calculate the shared variance between EEG, fMRI, and the features
            statistics['r2']['shared'] = statistics['r2']['eeg'] + statistics['r2']['features'] - statistics['r2']['full']

            if save_whole_brain and (time < .25) and (time > -.1):
                # Put the results in each voxel in a dataframe
                whole_brain_results = benchmark.metadata.copy()
                whole_brain_results['time'] = time
                whole_brain_results['eeg_sid'] = sid
                whole_brain_results['feature_category'] = category
                if i_cat == 0: 
                    whole_brain_results['r2'] = statistics['r2']['eeg']
                    pd.DataFrame(whole_brain_results).to_csv(f'{out_file_prefix}_time-{time_ind}_whole-brain-decoding.csv.gz', index=False)

                whole_brain_results['r2'] = statistics['r2']['shared']
                pd.DataFrame(whole_brain_results).to_csv(f'{out_file_prefix}_time-{time_ind}_category-{category}_whole-brain-decoding.csv.gz', index=False)

            # perform the significance testing in the eeg data only on the first category loop
            if i_cat == 0: 
                statistics['r2_filtered']['eeg'] = statistics['r2']['eeg'][voxel_ids]
                statistics['r2_null']['eeg'] = stats.perm_gpu(y_hat['eeg'][:, voxel_ids], y_test_filtered).cpu().detach().numpy().T
                statistics['r2_var']['eeg'] = stats.bootstrap_gpu(y_hat['eeg'][:, voxel_ids], y_test_filtered).cpu().detach().numpy().T
                for fmri_sid in metadata_filtered.subj_id.unique(): 
                    for roi in rois: 
                        bool_idx = (metadata_filtered.roi_name == roi) & (metadata_filtered.subj_id == fmri_sid)
                        roi_inds = metadata_filtered.loc[bool_idx].reset_index()['index'].tolist()
                        out.append({'roi': roi, 'regression_type': 'eeg', 'category': 'none',
                                    'r2_null': statistics['r2_null']['eeg'][roi_inds].mean(axis=0),
                                    'r2_var': statistics['r2_var']['eeg'][roi_inds].mean(axis=0),
                                    'r2': statistics['r2_filtered']['eeg'][roi_inds].mean(),
                                    'fmri_sid': fmri_sid, 'eeg_sid': sid, 'time': time})

            # perform the significance testing for the shared variance
            statistics['r2_filtered']['shared'] = statistics['r2']['shared'][voxel_ids]
            statistics['r2_null']['shared'] = stats.perm_shared_variance_gpu(y_hat['eeg'][:, voxel_ids],
                                                                             y_hat['features'][:, voxel_ids],
                                                                             y_hat['full'][:, voxel_ids],
                                                                             y_test_filtered).cpu().detach().numpy().T
            statistics['r2_var']['shared'] = stats.bootstrap_shared_variance_gpu(y_hat['eeg'][:, voxel_ids],
                                                                                 y_hat['features'][:, voxel_ids],
                                                                                 y_hat['full'][:, voxel_ids],
                                                                                 y_test_filtered).cpu().detach().numpy().T
            for fmri_sid in metadata_filtered.subj_id.unique(): 
                for roi in rois: 
                    bool_idx = (metadata_filtered.roi_name == roi) & (metadata_filtered.subj_id == fmri_sid)
                    roi_inds = metadata_filtered.loc[bool_idx].reset_index()['index'].tolist()
                    out.append({'roi': roi, 'regression_type': 'shared', 'category': category,
                                'r2_null': statistics['r2_null']['shared'][roi_inds].mean(axis=0),
                                'r2_var': statistics['r2_var']['shared'][roi_inds].mean(axis=0),
                                'r2': statistics['r2_filtered']['shared'][roi_inds].mean(),
                                'fmri_sid': fmri_sid, 'eeg_sid': sid, 'time': time})

            # free up gpu memory
            del statistics['r2']['eeg'], statistics['r2']['full'], statistics['r2']['shared']
            del statistics['r2_filtered']['shared'], statistics['r2_null']['shared'], statistics['r2_var']['shared']
            del X['eeg'], X['full']
            if i_cat == 0:
                del statistics['r2_filtered']['eeg'], statistics['r2_null']['eeg'], statistics['r2_var']['eeg']
            gc.collect()
            torch.cuda.empty_cache()

        # free up gpu memory
        del statistics, X, pipe
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(out).to_pickle(f'{out_file_prefix}_roi-decoding.pkl.gz')


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
