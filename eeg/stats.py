#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from scipy import ndimage
import torch
from torchmetrics.functional import pearson_corrcoef, r2_score, explained_variance
from scipy.stats import wilcoxon, t
from mne.stats import permutation_cluster_1samp_test


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


def compare_latencies_paired(latencies_dict, cat_pairs=None):
    """
    Compare latencies between categories using paired nonparametric tests
    latencies_dict: {category: [n_participants] or scalar latency estimates}
    """
    
    if cat_pairs is None:
        cat_pairs = [(i, j) for i in range(len(latencies_dict)) 
                     for j in range(i+1, len(latencies_dict))]
    
    comparisons = {}
    for cat_a, cat_b in cat_pairs:
        lat_a = latencies_dict[cat_a]
        lat_b = latencies_dict[cat_b]
        
        # If single value, can't do paired test - use point estimates
        if np.isscalar(lat_a) or lat_a.size == 1:
            diff = lat_a - lat_b
            comparisons[(cat_a, cat_b)] = {
                'diff_ms': diff,
                'test': 'point_estimate'
            }
        else:
            # Paired Wilcoxon if per-participant latencies available
            W, p_val = wilcoxon(lat_a, lat_b)
            comparisons[(cat_a, cat_b)] = {
                'W_statistic': W,
                'p_value': p_val,
                'median_diff': np.median(lat_a - lat_b),
                'test': 'wilcoxon'
            }
    
    return comparisons


def bootstrap_latency_ci(corr_data, times, 
                         n_bootstrap=1000,
                         n_permutations=100,
                         alpha=0.05, tail=1,
                         df_adjustment=2, seed=None, 
                         verbose=False):  # Reduced for speed
    """
    For each bootstrap resample:
    1. Compute correlations
    2. Run full cluster permutation test
    3. Extract onset latency from significant cluster
    
    This is computationally expensive but most rigorous.
    """
    latencies_boot = []
    n_subs = corr_data.shape[1]
    t_crit = t.ppf(1 - alpha, n_subs - df_adjustment) # Critical t-value for your specific df
    
    if verbose:
        boot_range = tqdm(range(n_bootstrap), desc='Bootstrapping latencies')
    else:
        boot_range = range(n_bootstrap)
    for boot in boot_range:
        boot_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_corrs = corr_data[:, boot_idx]
        
        # Cluster test
        try:
            _, clusters, p_values, _ = permutation_cluster_1samp_test(
                boot_corrs.T,
                n_permutations=n_permutations, 
                threshold=t_crit, 
                tail=tail,
                seed=seed,
                verbose='ERROR')
            
            # Extract first significant cluster
            sig_idx = np.where(p_values < alpha)[0]
            if len(sig_idx) > 0:
                first_cluster = clusters[sig_idx[0]]
                onset_idx = first_cluster[0][0]
                latencies_boot.append(float(times[onset_idx]))
        except:
            pass  # Skip if no clusters found
    
    if len(latencies_boot) > 0:
        latencies_boot = np.asarray(latencies_boot, dtype=float)
        return np.quantile(latencies_boot, [alpha/2, 1 - (alpha/2)])
    else:
        return np.nan, np.nan
    

def get_onset_latency(corr_data, times, 
                      n_permutations=1000, 
                      alpha=0.05, tail=1,
                      df_adjustment=2, seed=None):
    """
    corr_data: [n_time, n_participants]
    Performs permutation cluster test against H0 (r=0)
    Returns: onset_latency, p_value, significant_timepoints
    """
    
    # Remove participants with NaN (if any)
    valid_data = corr_data[:, ~np.isnan(corr_data).any(axis=0)]
    
    # Convert correlations to t-statistics for cluster test
    n_subs = valid_data.shape[1]
    t_crit = t.ppf(1 - alpha, n_subs - df_adjustment) # Critical t-value for your specific df

    # Cluster permutation test against H0 (t=0)
    T_obs, clusters, p_values, H0 = permutation_cluster_1samp_test(
        corr_data.T,  # transpose to [n_participants, n_time]
        n_permutations=n_permutations,
        threshold=t_crit,
        tail=tail,  # one-tailed: test if r > 0
        out_type='indices',  # returns indices, not boolean arrays
        seed=seed,
        verbose='ERROR'
    )
    
    # Find first significant cluster
    sig_clusters_idx = np.where(p_values < alpha)[0]
    
    if len(sig_clusters_idx) == 0:
        return {'onset_latency_ms': np.nan,
            'sig_clusters': [np.nan],
            'cluster_pvals': [np.nan],
            'sig_timepoints': np.zeros(corr_data.shape[0], dtype=bool)
    }

    
    # Get onset latency from first significant cluster
    first_sig_cluster_idx = clusters[sig_clusters_idx[0]]
    onset_timepoint_idx = first_sig_cluster_idx[0][0]
    onset_latency_ms = times[onset_timepoint_idx]

    # Get p-values and clusters for significant clusters
    cluster_pvals = [p_values[sig_idx] for sig_idx in sig_clusters_idx]
    sig_clusters = [clusters[sig_idx] for sig_idx in sig_clusters_idx]

    # Create boolean array of significant timepoints
    sig_timepoints = np.zeros(corr_data.shape[0], dtype=bool)
    for cluster in sig_clusters:
        sig_timepoints[cluster[0]] = True

    return {'onset_latency_ms': onset_latency_ms,
            'sig_clusters': sig_clusters,
            'cluster_pvals': cluster_pvals,
            'sig_timepoints': sig_timepoints
    }

def bootstrap_ci(arr, n_bootstrap=1000, 
                 seed=None, verbose=False):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr)

    n_subs = arr.shape[1]
    out = np.zeros((n_bootstrap, arr.shape[0]))
    if verbose:
        boot_range = tqdm(range(n_bootstrap), desc='Bootstrapping')
    else:
        boot_range = range(n_bootstrap)
    for i in boot_range:
        idx = np.random.choice(n_subs, n_subs, replace=True)
        arr_sample = arr[:, idx].mean(axis=1)
        out[i] = arr_sample
    return np.quantile(out, [0.025, 0.975], axis=0)