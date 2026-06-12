import inspect
from tqdm import tqdm 
import numpy as np
import pandas as pd
import gc
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import torch 
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
try:
    from deepjuice.alignment import TorchRidgeGCV
    _HAS_DEEPJUICE = True
except ImportError:
    TorchRidgeGCV = None
    _HAS_DEEPJUICE = False
from himalaya.ridge import GroupRidgeCV, RidgeCV
from himalaya.backend import set_backend
from himalaya.scoring import r2_score_split, r2_score
from kneed import KneeLocator
from sklearn.decomposition import PCA as sklearn_PCA
import torch
import torch.nn as nn
from typing import Optional, Union
from torchmetrics.functional import pearson_corrcoef, r2_score, explained_variance


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eeg import stats
from eeg.pca import PCA
from eeg.tools import to_torch


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


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Works like sklearn.preprocessing.StandardScaler but for PyTorch tensors.
    Supports GPU tensors and maintains compatibility with PyTorch's device management.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small value to avoid division by zero
        """
        self.epsilon = epsilon
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None
        self.is_fitted_ = False
        self.device_ = None
        self.dtype_ = None
    
    def fit(self, X: torch.Tensor) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Tensor of shape (n_samples, n_features) or (n_samples, ...)
            
        Returns:
            self
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X)).float()

        # Store device and dtype for later use
        self.device_ = X.device
        self.dtype_ = X.dtype
        
        # Compute mean and std across the first dimension (samples)
        self.mean_ = torch.mean(X, dim=0, keepdim=False)
        self.std_ = torch.std(X, dim=0, keepdim=False, unbiased=True)
        
        # Ensure std has minimum value to avoid division by zero
        self.std_ = torch.clamp(self.std_, min=self.epsilon)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Tensor to transform
            
        Returns:
            Transformed tensor
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform. Call fit() first.")
        
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X)).float()

        # Ensure tensors are on the same device and dtype
        mean = self.mean_.to(X.device).to(X.dtype)
        std = self.std_.to(X.device).to(X.dtype)
        
        return (X - mean) / std
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit to data, then transform it.
        
        Args:
            X: Tensor to fit and transform
            
        Returns:
            Transformed tensor
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Transformed tensor
            
        Returns:
            Original scale tensor
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(X)}")
        
        mean = self.mean_.to(X.device).to(X.dtype)
        std = self.std_.to(X.device).to(X.dtype)
        
        return X * std + mean
    
    def get_params(self) -> dict:
        """Get scaler parameters (mean and std)."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted first.")
        
        return {
            'mean': self.mean_.clone().cpu(),
            'std': self.std_.clone().cpu(),
        }
    
    def set_params(self, mean: torch.Tensor, std: torch.Tensor) -> 'StandardScaler':
        """Set scaler parameters manually."""
        self.mean_ = mean.clone()
        self.std_ = std.clone()
        self.device_ = mean.device
        self.dtype_ = mean.dtype
        self.is_fitted_ = True
        return self
    
    def __repr__(self) -> str:
        if self.is_fitted_:
            return (f"StandardScaler(mean={self.mean_.shape}, "
                   f"std={self.std_.shape}, fitted=True)")
        return "StandardScaler(fitted=False)"


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
    def split(neural, train_idx, test_idx):
        if type(neural) is pd.core.frame.DataFrame: 
            return neural.iloc[train_idx].to_numpy(), neural.iloc[test_idx].to_numpy()
        else:
            return neural[train_idx], neural[test_idx]
        

    train_idx = behavior.loc[behavior['stimulus_set'] == 'train'].index
    test_idx = behavior.loc[behavior['stimulus_set'] == 'test'].index

    train = {}
    test = {}
    for key, neural in neurals.items():
        if type(neural) == dict:
            train[key], test[key] = {}, {}
            for inner_key, inner_neural in neural.items():
                train[key][inner_key], test[key][inner_key] = split(inner_neural, train_idx, test_idx)
        else:
            train[key], test[key] = split(neural, train_idx, test_idx)

    if behavior_categories is None:
        cols = [col for col in behavior.columns if 'rating-' in col]
        train['behavior'] = behavior.iloc[train_idx][cols].to_numpy()
        test['behavior'] = behavior.iloc[test_idx][cols].to_numpy()
    else:
        for key, cols in behavior_categories.items(): 
            train[key] = behavior.iloc[train_idx][cols].to_numpy()
            test[key] = behavior.iloc[test_idx][cols].to_numpy()
    return train, test


def feature_scaler(train, test=None, dim=0, device='cpu'):
    """Calculate the mean and std of the train set. 
    Normalize the train and test by this mean and std. 
    If the input is a numpy array, it is converted to a torch tensor to perform normalization. 

    Args:
        train (torch.Tensor or np.ndarray): 1 or 2D tensor of samples x features (or reversed).
                                            If np.ndarray, output will be a torch tensor. 
        test (torch.Tensor or np.ndarray): 1 or 2D tensor of samples x features (or reversed).
                                            If np.ndarray, output will be a torch tensor. 
        dim (int, optional): The dimension over which to calculate the mean and std. Defaults to 0.
        device (str, optional): whether on cpu or cuda. Default is cpu.

    Returns:
        train_scored (torch.Tensor): 1 or 2D tensor of samples x features normalized by train mean and std
        test_scored (torch.Tensor): 1 or 2D tensor of samples x features normalized by train mean and std
    """

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    if test is not None:
        test_scaled = scaler.transform(test)
        return train_scaled, test_scaled
    else:
        return train_scaled


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


def pca_rotation(X_train, X_test, groups=None, n_components=None):
    """_summary_

    Args:
        X_train (torch.Tensor): X training data
        X_test (torch.Tensor): X test data
        groups (numpy.array, optional ): if not None a 1D np.array with length equal to dim 1 of 
                                         X_train and X_test that defines the groups of the features
                                         Default is None
        n_components: number of components to keeep from PCA

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
            if n_components is None: 
                if torch.sum(idx) > X_train.size()[0]: 
                    pca = PCA()
                    pca.fit(X_train[:, idx])
                    ev_ratio = pca.explained_variance_ratio_
                    kneedle = KneeLocator(np.arange(len(ev_ratio))+1, ev_ratio,
                                        curve="convex", direction="decreasing",
                                        interp_method="polynomial")
                    n_components = kneedle.knee
                else:
                    n_components = int(torch.sum(idx))

            pca = PCA(n_components=n_components)
            X_train_.append(pca.fit_transform(X_train[:, idx]))
            X_test_.append(pca.transform(X_test[:, idx]))
            groups_.append(torch.ones(n_components)*group)
        return torch.cat(X_train_, dim=1), torch.cat(X_test_, dim=1), torch.cat(groups_)
    else:
        if n_components is None: 
            if X_train.size()[0] < X_train.size()[1]: 
                pca = PCA()
                pca.fit(X_train)
                ev_ratio = pca.explained_variance_ratio_
                kneedle = KneeLocator(np.arange(len(ev_ratio))+1, ev_ratio,
                                        curve="convex", direction="decreasing",
                                        interp_method="polynomial")
                n_components = kneedle.knee
            else:
                n_components = X_train.size()[1]

        pca = PCA(n_components=n_components)
        return pca.fit_transform(X_train), pca.transform(X_test), groups


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


def banded_ridge(X_train, y_train, X_test, y_test,
                 groups,
                 alpha_start=-2, alpha_stop=5,
                 device='cuda', rotate_x=True):
    """Use himalaya GroupRidgeCV to perform the regression 
    and predict the response in the held out data

    Args:
        X_train (torch.Tensor): training X data
        y_train (torch.Tensor): training y data
        X_test (torch.Tensor): testing X data
        y_test (torch.Tensor): testing y data
        alpha_start (int, optional): smallest power for alpha. Defaults to -2.
        alpha_stop (int, optional): largest power for alpha. Defaults to 5.
        device (str, optional): device location of tensors. Defaults to 'cuda'.
        rotate_x (bool, optional): rotate X using PCA prior to fitting regression
    
    Returns: 
        y_hat (torch.Tensor): predicted y values
    """
    if device == 'cpu':
        backend = set_backend("torch")
    else:
        backend = set_backend("torch_cuda")

    # Standardize X data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Standardize y data
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    alphas = np.logspace(alpha_start, alpha_stop, num=(alpha_stop-alpha_start)*2)

    if rotate_x:
        X_train_scaled, X_test_scaled, groups = pca_rotation(X_train_scaled, X_test_scaled, groups)
        print(f'X Size after rotation = {X_train_scaled.size()}')

    model = GroupRidgeCV(
        groups=groups,
        solver_params={'alphas': alphas},
        fit_intercept=False
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled, split=False)
    r2 = r2_score(y_test_scaled, y_pred)
    y_pred_split = model.predict(X_test_scaled, split=True)
    r2_split = r2_score_split(y_test_scaled, y_pred_split, include_correlation=True)
    return {'r2': r2, 'r2_split': r2_split, 'yhat': y_pred}


def _torch_ridge_gcv(X, y, X_test, alphas, scoring='pearsonr',
                     return_alpha=False, return_betas=False):
    """DeepJuice-free replacement for TorchRidgeGCV.

    Selects a per-target ridge penalty by maximizing the exact leave-one-out
    score over `alphas`, then refits on all data and predicts `X_test`. For
    scoring='pearsonr' the selection metric is the leave-one-out Pearson r
    (scale invariant, matching TorchRidgeGCV); otherwise negative LOO MSE is
    used, which ranks alphas identically to R²/explained variance.

    Unlike himalaya's RidgeCV (R²-scored), this does not over-regularize
    low-SNR targets, where R² selection collapses to a near-constant fit.
    Inputs are assumed already standardized (fit_intercept=False).
    """
    squeeze_out = y.ndim == 1
    y2 = y.unsqueeze(1) if squeeze_out else y
    alphas_t = torch.as_tensor(np.asarray(alphas, dtype=float),
                               dtype=X.dtype, device=X.device)

    # Economy SVD: X = U diag(S) Vh. Ridge hat matrix is U diag(d) U^T with
    # filter factors d = S^2 / (S^2 + alpha), giving closed-form LOO residuals.
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S2 = S ** 2
    UTy = U.T @ y2                                   # (k, t)
    U2 = U ** 2                                       # (n, k)
    y_c = y2 - y2.mean(dim=0, keepdim=True)
    sse_y = (y_c ** 2).sum(dim=0)                     # (t,)
    t = y2.shape[1]

    best_metric = torch.full((t,), float('-inf'), dtype=X.dtype, device=X.device)
    best_alpha = torch.zeros(t, dtype=X.dtype, device=X.device)
    for a in alphas_t:
        d = S2 / (S2 + a)                             # (k,)
        yhat = U @ (d.unsqueeze(1) * UTy)            # (n, t)
        h = (U2 * d.unsqueeze(0)).sum(dim=1)         # (n,) hat diagonal
        denom = (1.0 - h).clamp_min(1e-8).unsqueeze(1)
        yhat_loo = (yhat - h.unsqueeze(1) * y2) / denom
        if scoring == 'pearsonr':
            yl_c = yhat_loo - yhat_loo.mean(dim=0, keepdim=True)
            num = (yl_c * y_c).sum(dim=0)
            den = torch.sqrt((yl_c ** 2).sum(dim=0) * sse_y).clamp_min(1e-12)
            metric = num / den
        else:
            metric = -((yhat_loo - y2) ** 2).mean(dim=0)
        metric = torch.nan_to_num(metric, nan=float('-inf'))
        improve = metric > best_metric
        best_metric = torch.where(improve, metric, best_metric)
        best_alpha = torch.where(improve, a.expand_as(best_alpha), best_alpha)

    inv = S.unsqueeze(1) / (S2.unsqueeze(1) + best_alpha.unsqueeze(0))   # (k, t)
    coef = Vh.T @ (inv * UTy)                         # (p, t)
    out = {}
    if X_test is not None:
        yhat_test = X_test @ coef                     # (m, t)
        out['yhat'] = yhat_test.squeeze(1) if squeeze_out else yhat_test
    if return_alpha:
        out['alpha'] = best_alpha
    if return_betas:
        out['betas'] = coef.T
    return out


def ridge(X, y,
          X_test=None, y_test=None,
          groups=None,
          alpha_start=-2, alpha_stop=5,
          scoring='pearsonr', device='cuda',
          n_components=None,
          rotate_x=False, return_alpha=False, return_betas=False):
    """Use deepjuice TorchRidgeGCV to perform the regression 
    and predict the response in the held out data

    Args:
        X_train (torch.Tensor): training X data
        y_train (torch.Tensor): training y data
        X_test (torch.Tensor): testing X data
        groups (torch.Tensor): defines groups for performing PCA rotation
        alpha_start (int, optional): smallest power for alpha. Defaults to -2.
        alpha_stop (int, optional): largest power for alpha. Defaults to 5.
        scoring (str, optional): type of scoring function. Defaults to 'pearsonr'.
        device (str, optional): device location of tensors. Defaults to 'cuda'.
        rotate_x (bool, optional): rotate X using PCA prior to fitting regression
        return_alpha (bool, optional): return fitted alphas. Defaults to False.
        return_betas (bool, optional): return beta coefficients. Defaults to False.
        n_components (int, optional): define the number of components to use in PCA. Defaults to None. 
    
    Returns: 
        y_hat (torch.Tensor): predicted y values
    """

    # Standardize X data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    # Standardize y data
    y_train_scaled = scaler.fit_transform(y)
    if y_test is not None:
        y_test_scaled = scaler.transform(y_test)

    if rotate_x and X_test is not None:
        X, X_test, _ = pca_rotation(X, X_test, groups, n_components)

    interval = np.array([[.25, .5, .75, 1]]).T
    powers = np.expand_dims(10**np.linspace(alpha_start, alpha_stop, num=(alpha_stop - alpha_start)+1), axis=0)
    alphas = np.matmul(interval, powers).flatten()
    alphas.sort()

    if _HAS_DEEPJUICE:
        pipe = TorchRidgeGCV(alphas=alphas,
                            alpha_per_target=True,
                            device=device,
                            scale_X=False,
                            fit_intercept=False,
                            scoring=scoring)
        pipe.fit(X_train_scaled, y_train_scaled)

        out = dict()
        if X_test is not None:
            yhat = pipe.predict(X_test_scaled)
            out['yhat'] = yhat
        if return_alpha:
            out['alpha'] = pipe.alpha_
        if return_betas:
            out['betas'] = pipe.coef_.T
    else:
        # Fallback when deepjuice is unavailable. himalaya's RidgeCV selects
        # alpha by R², which over-regularizes low-SNR targets (e.g. EEG->fMRI),
        # so use a Pearson-scored leave-one-out ridge that mirrors TorchRidgeGCV.
        out = _torch_ridge_gcv(X_train_scaled, y_train_scaled,
                               X_test_scaled if X_test is not None else None,
                               alphas, scoring=scoring,
                               return_alpha=return_alpha, return_betas=return_betas)
        yhat = out.get('yhat')

    if y_test is not None:
        out['score'] = compute_score(y_test_scaled, yhat,
                                    score_type=scoring,
                                    adjusted=X_train_scaled.size()[1])
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
        X_train, X_test, _ = pca_rotation(X_train, X_test)
    
    coeffs = torch.linalg.lstsq(X_train, y_train).solution
    y_pred = X_test @ coeffs
    return {'yhat': y_pred.squeeze()}
