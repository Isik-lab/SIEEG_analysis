import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return value
    else:
        low_value = df.loc[df[colname] < value, colname].max()
        high_value = df.loc[df[colname] > value, colname].min()
        if (value - low_value) < (high_value - value):
            return low_value
        else:
            return high_value


def smooth(array, window_size=20, step_size=1):
    kernel = np.ones(window_size) / window_size
    return np.convolve(array, kernel, mode='full')[::step_size]


def resample(time, data, new_sample_rate=10):
    new_time = np.arange(time.min(), time.max(), new_sample_rate)
    interpolation_function = interp1d(time, data, kind='linear', fill_value="extrapolate")
    return new_time, interpolation_function(new_time)
