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


def bin_time_windows_cut(df, window_size=50, start_time=0, end_time=300):
    """
    Bin time values into windows using pd.cut with a fixed end time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'time' column
    window_size : int
        Size of each time window (default: 50)
    end_time : int
        Minimum time value to consider (default: 0)
    end_time : int
        Maximum time value to consider (default: 300)
        
    Returns:
    --------
    pandas.Series
        Series containing the binned time windows
    """
    bins = np.arange(start_time, end_time + window_size + 1, window_size)
    labels = bins[:-1]
    return pd.cut(df['time'].clip(upper=end_time), 
                 bins=bins, 
                 labels=labels, 
                 right=False)
