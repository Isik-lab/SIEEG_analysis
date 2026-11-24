#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from scipy import ndimage
import torch
from torchmetrics.functional import pearson_corrcoef, r2_score, explained_variance


SCORE_FUNCTIONS = {'pearsonr': pearson_corrcoef, 
                   'r2_score': r2_score,
                   'r2_adj': r2_score,
                   'explained_variance': explained_variance}


def compute_score(y_true, y_pred, score_type='pearsonr', adjusted=0):
    y_true = torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true
    y_pred = torch.tensor(y_pred) if isinstance(y_pred, np.ndarray) else y_pred

    if score_type == 'r2_adj':
        kwargs = {'multioutput': 'raw_values', 'adjusted': adjusted}
    elif score_type == 'explained_variance' or score_type == 'r2_score':
        kwargs = {'multioutput': 'raw_values'}
    else:
        kwargs = {}
    
    if score_type not in SCORE_FUNCTIONS:
        raise ValueError(f'Unknown score_type: {score_type}; ' +
                         f'Choose from: {SCORE_FUNCTIONS.keys()}')
    
    return SCORE_FUNCTIONS[score_type](y_pred, y_true, **kwargs)


def calculate_p(r_null, r_true, n_perm, H0):
    # Get the p-value depending on the type of test
    denominator = n_perm + 1
    if H0 == 'two_tailed':
        numerator = np.sum(np.abs(r_null) >= np.abs(r_true), axis=0) + 1
        p_ = numerator / denominator
    elif H0 == 'greater':
        numerator = np.sum(r_true > r_null, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    else:  # H0 == 'less':
        numerator = np.sum(r_true < r_null, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    return p_


def perm_gpu(y_true, y_pred, score_type='pearsonr',
             n_perm=int(5e3), verbose=False, adjusted=0):
    import torch

    y_true = torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true
    y_pred = torch.tensor(y_pred) if isinstance(y_pred, np.ndarray) else y_pred

    g = torch.Generator()
    dim = y_pred.size()

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = torch.zeros((n_perm, dim[-1]))if len(dim) > 1 else torch.zeros(n_perm)
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Permute the indices
        inds = torch.randperm(dim[0], generator=g)
        
        # Compute the correlation
        null = compute_score(y_true, y_pred[inds], score_type=score_type, adjusted=adjusted)

        # Store the value
        if len(dim) > 1:
            r_null[i, :] = null
        else:
            r_null[i] = null
    return r_null


def bootstrap_gpu(y_true, y_pred, score_type='pearsonr',
                  n_perm=int(5e3), verbose=False, adjusted=0):
    import torch

    y_true = torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true
    y_pred = torch.tensor(y_pred) if isinstance(y_pred, np.ndarray) else y_pred

    g = torch.Generator()
    dim = y_pred.size()

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Bootstapping')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, dim[-1]))if len(dim) > 1 else torch.zeros(n_perm)
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Generate a random sample of indices
        inds = torch.squeeze(torch.randint(high=dim[0], size=(dim[0],1), generator=g))

        # Compute the correlation
        var = compute_score(y_true[inds], y_pred[inds], score_type=score_type, adjusted=adjusted)
        
        # Store the value
        if len(dim) > 1:
            r_var[i, :] = var
        else:
            r_var[i] = var
    return r_var


def compute_null_clusters(nulls, alpha=0.05, desc=None, verbose=False):
    """
        inputs:
            nulls: 2d numpy array, the first dim is the time points and second is the number of permutations
            alpha: is the pvalue threshold, assumed 0.05
        outputs: 
            cluster_null: random cluster values
    """
    if desc is None:
        'Cluster permutation'

    n_perm = nulls.shape[-1]
    if verbose is True:
        iterator = tqdm(range(n_perm), total=n_perm, desc=desc)
    else:
        iterator = range(n_perm)

    cluster_null = []
    for i_perm in iterator:
        # Get the current random time series
        true = nulls[:, i_perm:i_perm + 1]

        # Get the null distribution
        indices_except_one = np.delete(np.arange(n_perm), i_perm)
        null = nulls[:, indices_except_one]

        # Calculate p and identify the clusters
        p = calculate_p(null.T, true.T, n_perm-1, 'greater')
        label, n = ndimage.label(p < alpha) # Label the clusters where p is less than alpha

        # If any clusters exist, find the largest one
        if n > 0: 
            cluster_sum = 0
            for i_cluster in range(n):
                cur_val = np.sum(true[label == i_cluster+1])
                if cur_val > cluster_sum:
                    cluster_sum = cur_val
            cluster_null.append(cluster_sum)

    return np.array(cluster_null)


def cluster_correction(rs, ps, r_nulls, alpha=0.05, desc=None, verbose=False):
    """
        inputs:
            rs: np.array of the r values at each time point
            ps: np.array of the p values at each time point 
            nulls: 2d numpy array, the first dim is the time points and second is the number of permutations
            alpha: is the pvalue threshold, assumed 0.05
            desc: a string passed to tqdm for progress update, default is 'Cluster permutation'
        outputs: 
            cluster_ps: a vector of corrected p values. Unsignificant p values are unchanged
    """
    cluster_nulls = compute_null_clusters(r_nulls, desc=desc, verbose=verbose)
    cluster_ps = ps.copy()

    # Label the clusters
    label, n = ndimage.label(ps < alpha)

    # If there are clusters, sum the r values and find whether they are 
    # greater than the randomly computed clusters
    if n > 0:
        for i_cluster in range(1, n + 1):
            time_inds = label == i_cluster
            cluster_val = np.sum(rs[time_inds])
            sum_val = np.sum(cluster_val > cluster_nulls)

            # Edit p values to the cluster p values
            cluster_ps[time_inds] = 1 - (sum_val / len(cluster_nulls))
    return cluster_ps
