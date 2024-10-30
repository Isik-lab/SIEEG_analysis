#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from scipy import ndimage
import torch
from torchmetrics.functional import r2_score, explained_variance


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


def perm_gpu(y_hat, y_true, score_func, n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat.size()

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Permute the indices
        inds = torch.randperm(dim[0], generator=g)
        
        # Compute the correlation
        r_null[i, :] = score_func(y_hat, y_true[inds], multioutput='raw_values')
    return r_null


def bootstrap_gpu(y_hat, y_true, score_func, n_perm=int(5e3), verbose=False, square=False):
    import torch
    g = torch.Generator()
    dim = y_hat.size()

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Bootstapping')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Generate a random sample of indices
        inds = torch.squeeze(torch.randint(high=dim[0], size=(dim[0],1), generator=g))

        # Compute the correlation
        r_var[i, :] = score_func(y_hat[inds], y_true[inds], multioutput='raw_values')
    return r_var


def perm_unique_variance_gpu(y_hat_ab, y_hat_a, y_true, score_func,
                             n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat_a.shape

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Shared variance permutation testing')
    else:
        iterator = range(n_perm)

    r_null = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Permute the indices
        inds = torch.randperm(dim[0], generator=g)

        # Compute the correlations
        r2_ab = score_func(y_hat_ab, y_true[inds], multioutput='raw_values')
        r2_a = score_func(y_hat_a, y_true[inds], multioutput='raw_values')
        r_null[i, :] = r2_ab - r2_a
    return r_null


def bootstrap_unique_variance_gpu(y_hat_ab, y_hat_a, y_true, score_func,
                                  n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat_a.shape

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Shared variance bootstrapping')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Generate a random sample of indices
        inds = torch.squeeze(torch.randint(high=dim[0], size=(dim[0], 1), generator=g))

        # Compute the correlations
        r2_ab = score_func(y_hat_ab[inds], y_true[inds], multioutput='raw_values')
        r2_a = score_func(y_hat_a[inds], y_true[inds], multioutput='raw_values')
        r_var[i, :] = r2_ab - r2_a
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
