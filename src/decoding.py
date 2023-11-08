from tqdm import tqdm 
import numpy as np
import pandas as pd
from src import stats

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def eeg_feature_decoding(neural_df, feature_df, features, channels):
    # initialize pipe and kfold splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    pipe = Pipeline([('scale', StandardScaler()), ('rr', Ridge())])

    results = []
    time_groups = neural_df.groupby('time')
    for time, time_df in time_groups:
        X = time_df[channels].to_numpy()
        for feature in features: 
            y = feature_df[feature].to_numpy()

            y_pred = []
            y_true = []
            for train_index, test_index in cv.split(X):
                pipe.fit(X[train_index], y[train_index])
                y_pred.append(pipe.predict(X[test_index]))
                y_true.append(y[test_index])
            r = stats.corr(np.concatenate(y_pred), np.concatenate(y_true))
            results.append([time, feature, r])

    results = pd.DataFrame(results, columns=['time', 'feature', 'r'])
    cat_type = pd.CategoricalDtype(categories=features, ordered=True)
    results['feature'] = results.feature.astype(cat_type)
    return results
