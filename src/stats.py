#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from scipy import ndimage


def filter_r(rs, ps, p_crit=0.05, correct=True, threshold=True):
    rs_out = rs.copy()
    if correct:
        _, ps_corrected, _, _ = multipletests(ps, method='fdr_bh')
    else:
        ps_corrected = ps.copy()

    if threshold:
        rs_out[ps_corrected >= p_crit] = 0.
    else:
        rs_out[rs_out < 0.] = 0.
    return rs_out, ps_corrected


def corr(x, y):
    x_m = x - np.nanmean(x)
    y_m = y - np.nanmean(y)
    numer = np.nansum(x_m * y_m)
    denom = np.sqrt(np.nansum(x_m * x_m) * np.nansum(y_m * y_m))
    if denom != 0:
        return numer / denom
    else:
        return np.nan


def corr2d(x, y):
    x_m = x - np.nanmean(x, axis=0)
    y_m = y - np.nanmean(y, axis=0)

    numer = np.nansum((x_m * y_m), axis=0)
    denom = np.sqrt(np.nansum((x_m * x_m), axis=0) * np.nansum((y_m * y_m), axis=0))
    denom[denom == 0] = np.nan
    return numer / denom


def mantel_permutation(a, i):
    a = squareform(a)
    inds = np.random.permutation(a.shape[0])
    a_shuffle = a[inds][:, inds]
    return squareform(a_shuffle)


def calculate_p(r_null_, r_true_, n_perm_, H0_):
    # Get the p-value depending on the type of test
    denominator = n_perm_ + 1
    if H0_ == 'two_tailed':
        numerator = np.sum(np.abs(r_null_) >= np.abs(r_true_), axis=0) + 1
        p_ = numerator / denominator
    elif H0_ == 'greater':
        numerator = np.sum(r_true_ > r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    else:  # H0 == 'less':
        numerator = np.sum(r_true_ < r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    return p_


def bootstrap(a, b, n_perm=int(5e3), square=True, verbose=True):
    # Randomly sample and recompute r^2 n_perm times
    if verbose:
        iter_loop = tqdm(range(n_perm), total=n_perm)
    else:
        iter_loop = range(n_perm)

    if a.ndim > 1 :
        r2_var = np.zeros((n_perm, a.shape[-1]))
        for i in iter_loop:
            inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                                   size=a.shape[0])
            if a.ndim == 3:
                a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
                b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
            else:
                a_sample = a[inds, :]
                b_sample = b[inds, :]
            r = corr2d(a_sample, b_sample)
            if square:
                r = np.sign(r)*(r**2)
            r2_var[i, :] = r
    else:
        r2_var = np.zeros((n_perm,))
        for i in iter_loop:
            inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                                   size=a.shape[0])
            r = corr(a[inds], b[inds])
            if square:
                r = np.sign(r)*(r**2)
            r2_var[i] = r
    return r2_var

def bootstrap_unique_variance(a, b, c, n_perm=int(5e3)):
    # Randomly sample and recompute r^2 n_perm times
    r2_var = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        if a.ndim == 3:
            a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
            b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
            c_sample = c[inds, ...].reshape(c.shape[0] * c.shape[1], c.shape[-1])
        else:  # a.ndim == 2:
            a_sample = a[inds, :]
            b_sample = b[inds, :]
            c_sample = c[inds, :]
        r2_var[i, :] = corr2d(a_sample, b_sample)**2 - corr2d(a_sample, c_sample)**2
    return r2_var


def perm(a, b, n_perm=int(5e3), H0='greater', square=True, verbose=True):
    if a.ndim > 1:
        r2_null = np.zeros((n_perm, a.shape[-1]))
        if a.ndim == 3:
            a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
            b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
            r = corr2d(a_not_shuffle, b)
        else: #a.ndim == 2:
            r = corr2d(a, b)
    else: #a.ndim == 1:
        r = corr(a, b)
        r2_null = np.zeros((n_perm,))

    if square:
        r_out = (r**2) * np.sign(r)
    else:
        r_out = r.copy()

    if verbose:
        iter_loop = tqdm(range(n_perm), total=n_perm)
    else:
        iter_loop = range(n_perm)

    for i in iter_loop:
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        elif a.ndim == 2:
            a_shuffle = a[inds, :]
        else:# a.ndim == 1:
            a_shuffle = a[inds]

        if a.ndim > 1:
            r = corr2d(a_shuffle, b)
        else:
            r = corr(a_shuffle, b)

        if square:
            r2 = (r**2) * np.sign(r)
        else:
            r2 = r.copy()

        if a.ndim > 1:
            r2_null[i, :] = r2
        else:
            r2_null[i] = r2

    # Get the p-value depending on the type of test
    p = calculate_p(r2_null, r_out, n_perm, H0)
    return r_out, p, r2_null


def perm_unique_variance(a, b, c, n_perm=int(5e3), H0='greater'):
    if a.ndim == 3:
        a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        c = c.reshape(c.shape[0] * c.shape[1], c.shape[-1])
        r2 = corr2d(a_not_shuffle, b)**2 - corr2d(a_not_shuffle, c)**2
    else:
        r2 = corr2d(a, b)**2 - corr2d(a, c)**2

    # Shuffle a and recompute r^2 n_perm times
    r2_null = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_shuffle = a[inds, :]
        r2_null[i, :] = corr2d(a_shuffle, b)**2 - corr2d(a_shuffle, c)**2

    p = calculate_p(r2_null, r2, n_perm, H0)
    return r2, p, r2_null


def corr2d_gpu(x, y):
    import torch
    x_m = x - torch.nanmean(x, dim=0)
    y_m = y - torch.nanmean(y, dim=0)

    numer = torch.nansum((x_m * y_m), dim=0)
    denom = torch.sqrt(torch.nansum((x_m * x_m), dim=0) * torch.nansum((y_m * y_m), dim=0))
    denom[denom == 0] = float('nan')
    return numer / denom


def sign_square(a):
    return np.sign(a) * (a ** 2)


def perm_gpu(y_hat, y_true, n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat.shape

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
        r_null[i, :] = sign_square(corr2d_gpu(y_hat, y_true[inds]))
    return r_null


def bootstrap_gpu(y_hat, y_true, n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat.shape

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Generate a random sample of indices
        inds = torch.squeeze(torch.randint(high=dim[0], size=(dim[0],1), generator=g))

        # Compute the correlation
        r_var[i, :] = sign_square(corr2d_gpu(y_hat[inds], y_true[inds]))
    return r_var


def perm_shared_variance_gpu(y_hat_a, y_hat_b, y_hat_ab, y_true,
                             n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat_a.shape

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Permute the indices
        inds = torch.randperm(dim[0], generator=g)

        # Compute the correlations
        r2_a = sign_square(corr2d_gpu(y_hat_a, y_true[inds]))
        r2_b = sign_square(corr2d_gpu(y_hat_b, y_true[inds]))
        r2_ab = sign_square(corr2d_gpu(y_hat_ab, y_true[inds]))
        r_null[i, :] = r2_a + r2_b - r2_ab
    return r_null


def bootstrap_shared_variance_gpu(y_hat_a, y_hat_b, y_hat_ab, y_true,
                                  n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    dim = y_hat_a.shape

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, dim[-1]))
    for i in iterator:
        g.manual_seed(i) # Set the random seed

        # Generate a random sample of indices
        inds = torch.squeeze(torch.randint(high=dim[0], size=(dim[0], 1), generator=g))

        # Compute the correlations
        r2_a = sign_square(corr2d_gpu(y_hat_a[inds], y_true[inds]))
        r2_b = sign_square(corr2d_gpu(y_hat_b[inds], y_true[inds]))
        r2_ab = sign_square(corr2d_gpu(y_hat_ab[inds], y_true[inds]))
        r_var[i, :] = r2_a + r2_b - r2_ab
    return r_var


def compute_null_clusters(nulls, alpha=0.05, desc=None):
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

    cluster_null = []
    for i_perm in tqdm(range(n_perm), total=n_perm, desc=desc):
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


def cluster_correction(rs, ps, r_nulls, alpha=0.05, desc=None):
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
    cluster_nulls = compute_null_clusters(r_nulls, desc=desc)
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
