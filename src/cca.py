import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from typing import Tuple, Optional

class CrossValidatedCCA(BaseEstimator, TransformerMixin):
    """
    CCA implementation with cross-validation to determine optimal number of components.
    Uses correlation between canonical variates on held-out data as validation metric.
    """
    def __init__(
        self,
        max_components: Optional[int] = None,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.max_components = max_components
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_n_components = None
        self.best_score = -np.inf
        self.cca_ = None
        self.scaler_X_ = StandardScaler()
        self.scaler_Y_ = StandardScaler()
    
    def _get_correlations(self, X_trans: np.ndarray, Y_trans: np.ndarray) -> float:
        """Calculate average correlation between canonical variates."""
        correlations = [
            np.corrcoef(X_trans[:, i], Y_trans[:, i])[0, 1]
            for i in range(X_trans.shape[1])
        ]
        return np.mean(correlations)
    
    def _cross_validate_components(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_components: int
    ) -> float:
        """Perform cross-validation for a specific number of components."""
        cv_scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Fit CCA
            cca = CCA(n_components=n_components)
            cca.fit(X_train, Y_train)
            
            # Transform validation data
            X_val_trans, Y_val_trans = cca.transform(X_val, Y_val)
            
            # Calculate correlation score
            score = self._get_correlations(X_val_trans, Y_val_trans)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'CrossValidatedCCA':
        """
        Fit the CCA model with optimal number of components.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features_X)
            First set of features
        Y : array-like of shape (n_samples, n_features_Y)
            Second set of features
        
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Scale the data
        X_scaled = self.scaler_X_.fit_transform(X)
        Y_scaled = self.scaler_Y_.fit_transform(Y)
        
        # Determine max components if not specified
        if self.max_components is None:
            self.max_components = min(X.shape[1], Y.shape[1])
        
        # Try different numbers of components
        for n_components in range(1, self.max_components + 1):
            cv_score = self._cross_validate_components(X_scaled, Y_scaled, n_components)
            
            if cv_score > self.best_score:
                self.best_score = cv_score
                self.best_n_components = n_components
        
        # Fit final model with best number of components
        self.cca_ = CCA(n_components=self.best_n_components)
        self.cca_.fit(X_scaled, Y_scaled)
        
        return self
    
    def transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the learned CCA transformation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features_X)
            First set of features
        Y : array-like of shape (n_samples, n_features_Y)
            Second set of features
            
        Returns:
        --------
        X_trans : array-like of shape (n_samples, n_components)
            Transformed X data
        Y_trans : array-like of shape (n_samples, n_components)
            Transformed Y data
        """
        X_scaled = self.scaler_X_.transform(X)
        Y_scaled = self.scaler_Y_.transform(Y)
        return self.cca_.transform(X_scaled, Y_scaled)
    
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the mean correlation between canonical variates on test data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features_X)
            First set of features
        Y : array-like of shape (n_samples, n_features_Y)
            Second set of features
            
        Returns:
        --------
        score : float
            Mean correlation between canonical variates
        """
        X_trans, Y_trans = self.transform(X, Y)
        return self._get_correlations(X_trans, Y_trans)

# Example usage:
def main():
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features_X = 10
    n_features_Y = 8
    
    # Create correlated datasets
    X = np.random.randn(n_samples, n_features_X)
    W = np.random.randn(n_features_X, n_features_Y)
    Y = X @ W + np.random.randn(n_samples, n_features_Y) * 0.1
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Fit CCA with cross-validation
    cca = CrossValidatedCCA(max_components=min(n_features_X, n_features_Y))
    cca.fit(X_train, Y_train)
    
    # Get test score
    test_score = cca.score(X_test, Y_test)
    
    print(f"Best number of components: {cca.best_n_components}")
    print(f"Best CV score: {cca.best_score:.3f}")
    print(f"Test score: {test_score:.3f}")

if __name__ == "__main__":
    main()