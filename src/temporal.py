import pandas as pd
import numpy as np


def resample_frame(frame, resample_rate):
    # Downsample the dataframe to the EEG output rate
    frame.drop(columns='trial', inplace=True)
    frame['time'] = pd.to_timedelta(frame['time'], unit='s')
    frame.set_index('time', inplace=True)
    frame = frame.resample(resample_rate).mean()
    frame.reset_index(inplace=True)
    frame['time'] = frame['time'].dt.total_seconds()
    return frame


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


def trim_time(frame, start=-.2, end=1):
    frame.drop(columns='trial', inplace=True)
    adj_start = find_neighbours(start, frame, 'time')
    frame = frame.loc[frame.time >= adj_start]

    # Adjust the time to the start time
    min_value = frame.time.min()
    if min_value != start:
        frame['time'] = np.round(frame['time'] + (start - min_value), decimals=3)

    frame = frame.loc[frame.time <= end]
    return frame


def smoothing(df, channels, grouping=['video1', 'video2'], precision=3):
    rolling_df = df.groupby(grouping).apply(lambda x: x[['time'] + channels].rolling(window=5, min_periods=2, step=2).mean())
    rolling_df = rolling_df.reset_index().dropna()
    rolling_df['time'] = rolling_df['time'].round(precision)
    cols = [col for col in rolling_df.columns if 'level' not in col]
    return rolling_df[cols]
