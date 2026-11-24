from tqdm import tqdm 
from itertools import combinations
import warnings
import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from joblib import Parallel, delayed

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def filter_pairs(df, video_set):
    out_df = df.loc[df['video1'].isin(video_set)]
    out_df = out_df.loc[out_df['video2'].isin(video_set)]
    return out_df


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


def fmri_decoding_distance(betas, masks, combo_list, videos, n_groups=5):
    results = []
    for roi, val in tqdm(masks.items(), desc='ROIs'):
        betas_masked = betas[val, ...]
        result_for_t = Parallel(n_jobs=-1)(
            delayed(fit_and_predict)(betas_masked[:, video1, :].squeeze().T,
                                    betas_masked[:, video2, :].squeeze().T,
                                    n_groups) for video1, video2 in tqdm(combo_list, total=len(combo_list), desc='Pairwise decoding')
        )
        for accuracy, (video1, video2) in zip(result_for_t, combo_list):
            results.append([roi, videos[video1], videos[video2], accuracy])
    return pd.DataFrame(results, columns=['roi', 'video1', 'video2', 'distance'])


def fmri_correlation_distance(betas, masks, combo_list, videos):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")

        results = []
        for roi, val in tqdm(masks.items(), desc='ROIs'):
            betas_masked = betas[val, ...]
            rdm = pdist(np.nanmean(betas_masked, axis=-1).T, metric='correlation')
            for i, (video1, video2) in enumerate(combo_list):
                results.append([roi, videos[video1], videos[video2], rdm[i]])
        return pd.DataFrame(results, columns=['roi', 'video1', 'video2', 'distance'])


def eeg_decoding_distance(df, channels, combo_list, n_groups=5):
    results = []
    time_groups = df.groupby('time')
    for time, time_df in tqdm(time_groups, total=len(time_groups)):
        result_for_t = Parallel(n_jobs=-1)(
            delayed(fit_and_predict)(time_df.loc[time_df.video_name == video1, channels].to_numpy(),
                                     time_df.loc[time_df.video_name == video2, channels].to_numpy(),
                                     n_groups) for video1, video2 in combo_list
        )
        for accuracy, (video1, video2) in zip(result_for_t, combo_list):
            results.append([time, video1, video2, accuracy])
    return pd.DataFrame(results, columns=['time', 'video1', 'video2', 'distance'])


def eeg_correlation_distance(df, channels, combo_list, videos):
    results = []
    time_groups = df.groupby('time')
    for time, time_df in tqdm(time_groups, total=len(time_groups)):
        rdm = pdist(time_df[channels].to_numpy(), metric='correlation')
        for i, (video1, video2) in enumerate(combo_list):
            results.append([time, videos[video1], videos[video2],
                            rdm[i]])
    return pd.DataFrame(results, columns=['time', 'video1', 'video2', 'distance'])


def feature_distance(df, features):
    video_set = df.video_name.unique()
    videos_nCk = list(combinations(video_set, 2))
    feature_rdms = []
    for feature in tqdm(features):
        if 'alexnet' == feature or 'moten' == feature:
            cols = [col for col in df.columns if feature in col]
            array = df[cols].to_numpy()
            distance_matrix = pdist(array, metric='correlation')
        else: 
            array = np.expand_dims(df[feature].to_numpy(), axis=1)
            distance_matrix = pdist(array, metric='euclidean')
            
        for idx, (video1, video2) in enumerate(videos_nCk):
            feature_rdms.append([video1, video2, feature, distance_matrix[idx]])
    return pd.DataFrame(feature_rdms, columns=['video1', 'video2', 'feature', 'distance'])
 

def compute_eeg_feature_rsa(feature_rdms, eeg_rdms, features):
    feature_group = feature_rdms.groupby('feature')
    neural_group = eeg_rdms.groupby('time')
    rsa = []
    for feature, feature_rdm in feature_group:
        for time, time_rdm in neural_group:
            rho, _ = spearmanr(feature_rdm.distance, time_rdm.distance)
            rsa.append([feature, time, rho])
    rsa = pd.DataFrame(rsa, columns=['feature', 'time', 'Spearman rho'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    rsa['feature'] = rsa.feature.astype(cat_type)
    return rsa    


def compute_eeg_fmri_rsa(fmri_rdms, eeg_rdms, rois):
    roi_group = fmri_rdms.groupby('roi')
    neural_group = eeg_rdms.groupby('time')
    rsa = []
    for roi, roi_rdm in tqdm(roi_group):
        for time, time_rdm in neural_group:
            rho, _ = spearmanr(roi_rdm.distance, time_rdm.distance)
            rsa.append([roi, time, rho])
    rsa = pd.DataFrame(rsa, columns=['roi', 'time', 'Spearman rho'])
    cat_type = pd.CategoricalDtype(categories=rois, ordered=True)
    rsa['roi'] = rsa.roi.astype(cat_type)
    return rsa


def compute_feature_fmri_rsa(feature_rdms, fmri_rdms, features, rois):
    feature_group = feature_rdms.groupby('feature')
    neural_group = fmri_rdms.groupby('roi')
    rsa = []
    for feature, feature_rdm in tqdm(feature_group):
        for time, time_rdm in neural_group:
            rho, _ = spearmanr(feature_rdm.distance, time_rdm.distance)
            rsa.append([feature, time, rho])
    rsa = pd.DataFrame(rsa, columns=['feature', 'roi', 'Spearman rho'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    rsa['feature'] = rsa.feature.astype(cat_type)
    cat_type = pd.CategoricalDtype(categories=rois, ordered=True)
    rsa['roi'] = rsa.roi.astype(cat_type)
    return rsa
