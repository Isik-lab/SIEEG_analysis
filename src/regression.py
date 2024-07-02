import inspect
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
from src.pca import PCA
from himalaya.ridge import GroupRidgeCV
from himalaya.backend import set_backend
from src.tools import to_torch
from kneed import KneeLocator


def T_torch(tensor):
    return torch.transpose(tensor, dim0=0, dim1=1)


def correlation_scorer(y_true, y_pred):
    """correlation_scorer

    Args:
        y_true (numpy.ndarray): 1d np vector from the actual data
        y_pred (numpy.ndarray): 1d np vector of predicted values

    Returns:
        numpy.ndarray: Pearson correlation between y_true and y_pred
    """
    return stats.corr(y_true, y_pred)


def train_test_split(behavior, neurals, behavior_categories=None):
    """split the data into train and test sets based on the simulus data

    Args:
        behavior (pandas.core.frame.DataFrame): ratings for each of the features with info about train/test split
        neurals (dictionary of pandas.core.frame.DataFrame or numpy.ndarray): ratings for each of the features with info about train/test split
        behavior_categories (dictionary, optional): dictionary that maps behavioral categories to columns in behavior. Default is None

    Returns:
        out (dictionary): returns a dictionary of the data that has been separated
    """
    train_idx = behavior.loc[behavior['stimulus_set'] == 'train'].index
    test_idx = behavior.loc[behavior['stimulus_set'] == 'test'].index

    train = {}
    test = {}
    for key, neural in neurals.items():
        if type(neural) is pd.core.frame.DataFrame: 
            neural_train = neural.iloc[train_idx].to_numpy()
            neural_test = neural.iloc[test_idx].to_numpy()
        else:
            neural_train = neural[train_idx]
            neural_test = neural[test_idx]
        train[key], test[key] = neural_train, neural_test

    if behavior_categories is None:
        cols = [col for col in behavior.columns if 'rating-' in col]
        train['behavior'] = behavior.iloc[train_idx][cols].to_numpy()
        test['behavior'] = behavior.iloc[test_idx][cols].to_numpy()
    else:
        for key, cols in behavior_categories.items(): 
            train[key] = behavior.iloc[train_idx][cols].to_numpy()
            test[key] = behavior.iloc[test_idx][cols].to_numpy()
    return train, test


def feature_scaler(train, test, dim=0, device='cpu'):
    """Calculate the mean and std of the train set. 
    Normalize the train and test by this mean and std. 
    If the input is a numpy array, it is converted to a torch tensor to perform normalization. 

    Args:
        train (torch.Tensor or np.ndarray): 2D tensor of samples x features (or reversed).
                                            If np.ndarray, output will be a torch tensor. 
        test (torch.Tensor or np.ndarray): 2D tensor of samples x features (or reversed).
                                            If np.ndarray, output will be a torch tensor. 
        dim (int, optional): The dimension over which to calculate the mean and std. Defaults to 0.
        device (str, optional): whether on cpu or cuda. Default is cpu.

    Returns:
        train_scored (torch.Tensor): 2D tensor of samples x features normalized by train mean and std
        test_scored (torch.Tensor): 2D tensor of samples x features normalized by train mean and std
    """
    if type(train) == np.ndarray:
        # If the input is a numpy array, first convert to torch tensor
        [train, test] = to_torch([train, test], device=device)

    # First make sure that there are no features that are the same for all videos.
    # If there are, remove those features from the train and test data. 
    idx = torch.squeeze(torch.std(train, dim=dim).nonzero())
    train_ = train[:, idx]
    test_ = test[:, idx]

    # Calculate the mean and std of the modified train data
    train_mean = torch.mean(train_, dim=dim, keepdim=True)
    train_std = torch.std(train_, dim=dim, keepdim=True)

    train_normed = (train_-train_mean)/train_std
    test_normed = (test_-train_mean)/train_std
    return train_normed, test_normed


def preprocess(X_train, X_test, y_train, y_test):
    """scales X_train, X_test, y_train, y_test inplace 

    Args:
        X_train (numpy.ndarray): 2D np array of samples x features (or reversed)
        X_test (numpy.ndarray): 2D np array of samples x features (or reversed)
        y_train (numpy.ndarray): 2D np array of samples x features (or reversed)
        y_test (numpy.ndarray): 2D np array of samples x features (or reversed)

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test = feature_scaler(X_train, X_test)   
    y_train, y_test = feature_scaler(y_train, y_test)   
    return X_train, X_test, y_train, y_test


def pca_rotation(X_train, X_test, groups=None):
    """_summary_

    Args:
        X_train (torch.Tensor): X training data
        X_test (torch.Tensor): X test data
        groups (numpy.array, optional ): if not None a 1D np.array with length equal to dim 1 of 
                                         X_train and X_test that defines the groups of the features
                                         Default is None

    Returns:
        X_train_rotated (torch.Tensor): rotated X train within group
        X_test_rotated (torch.Tensor): rotated X test based on components from X train within group
    """
    if groups is not None:
        X_train_, X_test_, groups_ = [], [], []
        for group in torch.unique(groups): 
            idx = groups == group #Get # of features 

            # In the regression, the # of features needs to differ from the # of samples
            # So this identifies the elbow if necessary with first PCA
            if torch.sum(idx) > X_train.size()[0]: 
                pca = PCA()
                pca.fit(X_train[:, idx])
                kneedle = KneeLocator(np.arange(X_train.size()[0])+1,
                                      pca.explained_variance_ratio_,
                                      S=1.0, curve="convex", direction="decreasing")
                n_components = kneedle.knee
            else:
                n_components = int(torch.sum(idx))

            print(f'{n_components=}')
            pca = PCA(n_components=n_components)
            X_train_.append(pca.fit_transform(X_train[:, idx]))
            X_test_.append(pca.transform(X_test[:, idx]))
            groups_.append(torch.ones(n_components)*group)
        return torch.cat(X_train_, dim=1), torch.cat(X_test_, dim=1), torch.cat(groups_)
    else:
        pca = PCA(n_components=X_train.size()[1])
        return pca.fit_transform(X_train), pca.transform(X_test)


def regression_model(method_name, X_train, y_train, X_test, **kwargs):
    method_dict = {
        'ols': ols,
        'ridge': ridge,
        'banded_ridge': banded_ridge
    }
    
    if method_name not in method_dict:
        raise ValueError(f"Unknown method name: {method_name}")
    regression_function = method_dict[method_name]

    # Filter kwargs to include only those that the function accepts
    params = inspect.signature(regression_function).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
    
    return regression_function(X_train, y_train, X_test, **filtered_kwargs)


def banded_ridge(X_train, y_train, X_test, groups,
                 alpha_start=-2, alpha_stop=5,
                 device='cuda', rotate_x=True,
                 return_alpha=False, return_betas=False):
    """Use himalaya GroupRidgeCV to perform the regression 
    and predict the response in the held out data

    Args:
        X_train (torch.Tensor): training X data
        y_train (torch.Tensor): training y data
        X_test (torch.Tensor): testing X data
        alpha_start (int, optional): smallest power for alpha. Defaults to -2.
        alpha_stop (int, optional): largest power for alpha. Defaults to 5.
        device (str, optional): device location of tensors. Defaults to 'cuda'.
        rotate_x (bool, optional): rotate X using PCA prior to fitting regression
        return_alpha (bool, optional): return fitted alphas. Defaults to False.
        return_betas (bool, optional): return beta coefficients. Defaults to False.
    
    Returns: 
        y_hat (torch.Tensor): predicted y values
    """
    if device == 'cpu':
        backend = set_backend("torch")
    else:
        backend = set_backend("torch_cuda")

    alphas = np.logspace(alpha_start, alpha_stop, num=(alpha_stop-alpha_start)+1)

    if rotate_x:
        X_train, X_test, groups = pca_rotation(X_train, X_test, groups)
        print(f'X Size after rotation = {X_train.size()}')

    pipe = GroupRidgeCV(groups=groups,
                        solver_params={'alphas': alphas},
                        fit_intercept=False)
    pipe.fit(X_train, y_train)
    out = {'yhat': pipe.predict(X_test)}

    if return_alpha:
        out['alpha'] = pipe.best_alphas_
    if return_betas:
        out['betas'] = pipe.coef_
    return out


def ridge(X_train, y_train, X_test, groups, 
          alpha_start=-2, alpha_stop=5,
          scoring='pearsonr', device='cuda',
          rotate_x=True,
          return_alpha=False, return_betas=False):
    """Use deepjuice TorchRidgeGCV to perform the regression 
    and predict the response in the held out data

    Args:
        X_train (torch.Tensor): training X data
        y_train (torch.Tensor): training y data
        X_test (torch.Tensor): testing X data
        alpha_start (int, optional): smallest power for alpha. Defaults to -2.
        alpha_stop (int, optional): largest power for alpha. Defaults to 5.
        scoring (str, optional): type of scoring function. Defaults to 'pearsonr'.
        device (str, optional): device location of tensors. Defaults to 'cuda'.
        rotate_x (bool, optional): rotate X using PCA prior to fitting regression
        return_alpha (bool, optional): return fitted alphas. Defaults to False.
        return_betas (bool, optional): return beta coefficients. Defaults to False.
    
    Returns: 
        y_hat (torch.Tensor): predicted y values
    """
    if rotate_x:
        X_train, X_test, _ = pca_rotation(X_train, X_test, groups)

    pipe = TorchRidgeGCV(alphas=np.logspace(alpha_start, alpha_stop),
                        alpha_per_target=True,
                        device=device,
                        scale_X=False,
                        fit_intercept=False,
                        scoring=scoring)

    pipe.fit(X_train, y_train)
    out = {'yhat': pipe.predict(X_test)}

    if return_alpha:
        out['alpha'] = pipe.alpha_
    if return_betas:
        out['betas'] = pipe.coef_
    return out


def ols(X_train, y_train, X_test, rotate_x=True):
    """
    Fits an ordinary least squares regression model using torch.linalg.lstsq
    without an intercept and generates test predictions.
    
    Parameters:
    X_train (torch.Tensor): Training data features of shape (n_samples, n_features)
    y_train (torch.Tensor): Training data targets of shape (n_samples, n_targets)
    X_test (torch.Tensor): Test data features of shape (m_samples, n_features)
    rotate_x (bool, optional): rotate X using PCA prior to fitting regression
    
    Returns:
    torch.Tensor: Predictions for the test data of shape (m_samples, n_targets)
    """
    # Concatenate lists of tensors if needed
    if isinstance(X_train, list):
        X_train = torch.cat(X_train, dim=0)
    if isinstance(X_test, list):
        X_test = torch.cat(X_test, dim=0)

    if rotate_x:
        X_train, X_test = pca_rotation(X_train, X_test)
    
    coeffs = torch.linalg.lstsq(X_train, y_train).solution
    y_pred = X_test @ coeffs
    return {'yhat': y_pred.squeeze()}


# def eeg_feature_decoding(neural_df, feature_df,
#                           features, channels):
#     """
#         inputs:
#             neural_df: pd.DataFrame of the EEG data in long format
#             feature_df: pd.DataFrame of videos by the annotated features
#             features: list of the features to predict in desired categorical order
#             channels: a list of channels present in the EEG data.
#                 This can vary between participants depending on preprocessing. 
#         output:
#             results: pd.DataFrame containing the Ridge results for each feature at each time point
#     """
#     # initialize pipe and kfold splitter
#     scorer = make_scorer(correlation_scorer, greater_is_better=True)
#     pipe = Pipeline([
#         ('scale', StandardScaler()),
#         ('rcv', RidgeCV(
#             fit_intercept=False,
#             alphas=[10.**power for power in np.arange(-5, 2)],
#             alpha_per_target=True,
#             scoring=scorer
#         ))
#     ])
#     results = []
#     time_groups = neural_df.groupby('time')
#     iterator = tqdm(time_groups, desc='Time')
#     for time, time_df in iterator:
#         # Split the data into train and test sets
#         X = {'train': time_df.loc[time_df.stimulus_set == 'train', channels].to_numpy(),
#              'test': time_df.loc[time_df.stimulus_set == 'test', channels].to_numpy()}
#         y = {'train': feature_df.loc[feature_df.stimulus_set == 'train', features].to_numpy(),
#              'test': feature_df.loc[feature_df.stimulus_set == 'test', features].to_numpy()}
#         # Perform the regression
#         pipe.fit(X['train'], y['train'])
#         y_hat = pipe.predict(X['test'])
#         # Compute significance and variance
#         r2s = stats.sign_square(stats.corr2d(y_hat, y['test']))
#         r2s_null = stats.perm(y_hat, y['test'], verbose=False)
#         r2s_var = stats.bootstrap(y_hat, y['test'], verbose=False)
#         # Append to the results
#         for feature, r2, r2_null, r2_var in zip(features, r2s, r2s_null.T, r2s_var.T): 
#             results.append([time, feature, r2, r2_null, r2_var])
#     # Turn list into dataframe with feature data as categorical
#     results = pd.DataFrame(results, columns=['time', 'feature', 'r2', 'r2_null', 'r2_var'])
#     cat_type = pd.CategoricalDtype(categories=features, ordered=True)
#     results['feature'] = results.feature.astype(cat_type)
#     return results

# def eeg_fmri_decoding(neural_df, benchmark, sid, 
#                       channels, feature_categories,
#                       device, out_file_prefix,
#                       verbose=True, scale_y=True,
#                       save_whole_brain=False,
#                       perform_sig_testing=True, 
#                       run_shared_variance=False,
#                       alphas=[10.**power for power in np.arange(-5, 2)]):
    
#     # get fmri in right format and then send to the gpu
#     inds = {'train': benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'train'].reset_index()['index'], 
#             'test': benchmark.stimulus_data.loc[benchmark.stimulus_data.stimulus_set == 'test'].reset_index()['index']}
#     video_list = {'train': benchmark.stimulus_data.loc[inds['train'], 'video_name'].tolist(),
#                   'test': benchmark.stimulus_data.loc[inds['test'], 'video_name'].tolist()}
#     y = {'train': torch.from_numpy(benchmark.response_data.to_numpy().T[inds['train']]).to(torch.float32).to(device),
#          'test': torch.from_numpy(benchmark.response_data.to_numpy().T[inds['test']]).to(torch.float32).to(device)}
#     if scale_y: 
#         y['train'], y['test'] = feature_scaler(y['train'], y['test'])
#     rois = benchmark.metadata.roi_name.unique().tolist()
#     rois.remove('none')
#     print(rois)
#     roi_bool_idx = benchmark.metadata.roi_name != 'none'

#     # Define pipe 
#     pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
#                     device=device, scale_X=False)

#     # Time iterator                
#     if verbose:
#         time_iterator = tqdm(neural_df.groupby('time'), desc='Time')
#     else:
#         time_iterator = neural_df.groupby('time')
#     # Time loop
#     category = 'none'
#     out = []
#     for i_time, (time, time_df) in enumerate(time_iterator):
#         time_df = time_df[channels+['video_name']].set_index('video_name')
#         # initialize pipe and kfold splitter
#         time_ind = str(i_time).zfill(3)
#         # Devide EEG time point into train and test, organize by fMRI, and send to the GPU
#         X = {}
#         statistics = {}
#         for stim_set in ['train', 'test']: 
#             set_arr = time_df.loc[video_list[stim_set]].to_numpy()
#             X[stim_set] = torch.from_numpy(set_arr).to(torch.float32).to(device)
#         X['train'], X['test'] = feature_scaler(X['train'], X['test'])

#         pipe.fit(X['train'], y['train'])
#         y_hat = pipe.predict(X['test'])
#         statistics['r2'] = stats.sign_square(stats.corr2d_gpu(y_hat, y['test'])).cpu().detach().numpy()

#         if save_whole_brain and (-.1 < time) and (time < .25):
#             # Put the results in each voxel in a dataframe
#             whole_brain_results = benchmark.metadata.copy()
#             whole_brain_results['time'] = time
#             whole_brain_results['eeg_sid'] = sid
#             whole_brain_results['feature_category'] = category
#             whole_brain_results['r2'] = statistics['r2']
#             pd.DataFrame(whole_brain_results).to_csv(f'{out_file_prefix}_time-{time_ind}_category-{category}_whole-brain-decoding.csv.gz', index=False)

#         # perform the significance testing in the eeg data only on the first category loop
#         if perform_sig_testing: 
#             statistics['r2_null'] = np.ones((len(roi_bool_idx), 5000))*np.nan
#             statistics['r2_var'] = np.ones((len(roi_bool_idx), 5000))*np.nan
#             statistics['r2_null'][roi_bool_idx] = stats.perm_gpu(y_hat[:, roi_bool_idx],
#                                                                  y['test'][:, roi_bool_idx],
#                                                                  verbose=False).cpu().detach().numpy().T
#             statistics['r2_var'][roi_bool_idx] = stats.bootstrap_gpu(y_hat[:, roi_bool_idx],
#                                                                      y['test'][:, roi_bool_idx],
#                                                                      verbose=False).cpu().detach().numpy().T

#         for fmri_sid in benchmark.metadata.subj_id.unique(): 
#             for roi in rois:
#                 bool_idx = (benchmark.metadata.roi_name == roi) & (benchmark.metadata.subj_id == fmri_sid)
#                 cur = {'fmri_sid': fmri_sid, 'time': time, 
#                         'roi': roi, 'regression_type': 'eeg',
#                         'category': category,
#                         'r2': statistics['r2'][bool_idx].mean()}
#                 if perform_sig_testing:
#                     cur['r2_null'] = statistics['r2_null'][bool_idx].mean(axis=0)
#                     cur['r2_var'] = statistics['r2_var'][bool_idx].mean(axis=0),
#                 out.append(cur)

#     if run_shared_variance: 
#         if verbose:
#             feature_iterator = tqdm(enumerate(feature_categories.items()),
#                                             total=len(feature_categories),
#                                             desc='Categories', leave=True)
#         else:
#             feature_iterator = feature_categories
#         # Feature category loop
#         for i_cat, (category, feature_list) in feature_iterator:
#             X = {'features': {}}
#             for stim_set in ['train', 'test']: 
#                 arr = benchmark.stimulus_data.iloc[inds[stim_set]][feature_list].to_numpy()
#                 X['features'][stim_set] = torch.from_numpy(arr).to(torch.float32).to(device)
#             X['features']['train'], X['features']['test'] = feature_scaler(X['features']['train'], X['features']['test'])

#             # Compute the regression for the features
#             pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
#                                 device=device, scale_X=False)
#             pipe.fit(X['features']['train'], y['train'])

#             # Make the prediction and store in a dict 
#             y_hat = {'features': pipe.predict(X['features']['test'])}

#             # Score the prediction and store in a dict
#             statistics = {'r2': {}, 'r2_null': {}, 'r2_var': {}}
#             statistics['r2']['features'] = stats.sign_square(stats.corr2d_gpu(y_hat['features'],
#                                                                               y['test'])).cpu().detach().numpy()

#             if verbose:
#                 time_iterator = tqdm(neural_df.groupby('time'), desc='Time')
#             else:
#                 time_iterator = neural_df.groupby('time')
#             # Time loop
#             for i_time, (time, time_df) in enumerate(time_iterator):
#                 time_df = time_df[channels+['video_name']].set_index('video_name')
#                 # initialize pipe and kfold splitter
#                 time_ind = str(i_time).zfill(3)

#                 # Devide EEG time point into train and test, organize by fMRI, and send to the GPU
#                 X['eeg'] = {}
#                 for stim_set in ['train', 'test']: 
#                     set_arr = time_df.loc[video_list[stim_set]].to_numpy()
#                     X['eeg'][stim_set] = torch.from_numpy(set_arr).to(torch.float32).to(device)
#                 X['eeg']['train'], X['eeg']['test'] = feature_scaler(X['eeg']['train'], X['eeg']['test'])

#                 # Create also the values for the combination of the EEG and feature data
#                 X['full'] = {}
#                 for stim_set in ['train', 'test']: 
#                     X['full'][stim_set] = torch.cat([X['eeg'][stim_set], X['features'][stim_set]], dim=1)

#                 # Fit and score the regression for EEG and full model
#                 for regression_type in ['eeg', 'full']: 
#                     pipe.fit(X[regression_type]['train'], y['train']) 
#                     y_hat[regression_type] = pipe.predict(X[regression_type]['test'])
#                     r2 = stats.sign_square(stats.corr2d_gpu(y_hat[regression_type], y['test']))
#                     statistics['r2'][regression_type] = r2.cpu().detach().numpy()

#                 # Calculate the shared variance between EEG, fMRI, and the features
#                 statistics['r2']['shared'] = statistics['r2']['eeg'] + statistics['r2']['features'] - statistics['r2']['full']

#                 if save_whole_brain and (-.1 < time) and (time < .25):
#                     # Put the results in each voxel in a dataframe
#                     whole_brain_results = benchmark.metadata.copy()
#                     whole_brain_results['time'] = time
#                     whole_brain_results['eeg_sid'] = sid
#                     whole_brain_results['feature_category'] = category
#                     whole_brain_results['r2'] = statistics['r2']['shared']
#                     pd.DataFrame(whole_brain_results).to_csv(f'{out_file_prefix}_time-{time_ind}_category-{category}_whole-brain-decoding.csv.gz', index=False)

#                 # perform the significance testing in the eeg data only on the first category loop
#                 if perform_sig_testing: 
#                     statistics['r2_null'] = {'shared': np.ones((len(roi_bool_idx), 5000))*np.nan}
#                     statistics['r2_var'] = {'shared': np.ones((len(roi_bool_idx), 5000))*np.nan}
#                     statistics['r2_null']['shared'][roi_bool_idx] = stats.perm_shared_variance_gpu(y_hat_a=y_hat['eeg'][:, roi_bool_idx],
#                                                                                                     y_hat_b=y_hat['features'][:, roi_bool_idx],
#                                                                                                     y_hat_ab=y_hat['full'][:, roi_bool_idx],
#                                                                                                     y_true=y['test'][:, roi_bool_idx],
#                                                                                                     verbose=False).cpu().detach().numpy().T
#                     statistics['r2_var']['shared'][roi_bool_idx] = stats.bootstrap_shared_variance_gpu(y_hat_a=y_hat['eeg'][:, roi_bool_idx],
#                                                                                                         y_hat_b=y_hat['features'][:, roi_bool_idx],
#                                                                                                         y_hat_ab=y_hat['full'][:, roi_bool_idx],
#                                                                                                         y_true=y['test'][:, roi_bool_idx],
#                                                                                                         verbose=False).cpu().detach().numpy().T
                    
#                 for fmri_sid in benchmark.metadata.subj_id.unique(): 
#                     for roi in rois:
#                         bool_idx = (benchmark.metadata.roi_name == roi) & (benchmark.metadata.subj_id == fmri_sid)
#                         cur = {'fmri_sid': fmri_sid, 'time': time, 
#                                 'roi': roi, 'regression_type': 'shared',
#                                 'category': category,
#                                 'r2': statistics['r2']['shared'][bool_idx].mean()}
#                         if perform_sig_testing: 
#                             cur['r2_null']= statistics['r2_null']['shared'][bool_idx].mean(axis=0),
#                             cur['r2_var']= statistics['r2_var']['shared'][bool_idx].mean(axis=0),
#                         out.append(cur)

#                 # free up gpu memory
#                 del statistics['r2']['eeg'], statistics['r2']['full'], statistics['r2']['shared']
#                 del X['eeg'], X['full']
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         # free up gpu memory
#         del statistics, X, pipe
#         gc.collect()
#         torch.cuda.empty_cache()
#     pd.DataFrame(out).to_pickle(f'{out_file_prefix}_roi-decoding.pkl.gz')

