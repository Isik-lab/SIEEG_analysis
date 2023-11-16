import numpy as np
from src import temporal
import pandas as pd
from tqdm import tqdm 


def filter_trials(df, artifact_trials):
    idx = np.where(np.invert(artifact_trials))[0]
    out = df.loc[idx]
    out = out.reset_index().drop(columns='trial')
    if len(out) > len(artifact_trials):
        out['trial_adjusted'] = out.groupby('time').cumcount()
    else:
        out['trial_adjusted'] = np.arange(0, len(out))
    out.rename(columns={'trial_adjusted': 'trial'}, inplace=True)
    return out


def downsample_and_filter(df, resample_rate, start_time, end_time):
    df_downsampled = df.groupby('trial').apply(temporal.resample_frame, resample_rate).reset_index().drop(columns='level_1')
    df_downsampled = df_downsampled.groupby('trial').apply(temporal.trim_time, start_time, end_time).reset_index().drop(columns='level_1')
    return df_downsampled


def combine_data(eeg, trials, eyetracking):
    out = pd.merge(eeg, trials[['trial', 'video_name', 'condition', 'response', 'stimulus_set']], on='trial', how='left')
    if eyetracking is not None:
        out = out.merge(eyetracking, on=['trial', 'time'])
    return out


def process_eyetracking(eyetracking, artifacts, offsets_df,
                        resample_rate='4ms', start_time=-0.2, end_time=1):
    df = filter_trials(eyetracking, artifacts)
    #clean up columns
    df.drop(columns=['run'], inplace=True)

    # add the eeg offsets time and adjust the time
    df = df.merge(offsets_df, on='trial').sort_values(by=['trial', 'time'])
    df['time'] = np.round(df['time'] + df['offset_eyetrack'], decimals=3)

    return downsample_and_filter(df, resample_rate, start_time, end_time) 


def regress_out_gaze(df, channels): 
    print('regressing out gaze')
    residuals = []
    for trial, trial_df in tqdm(df.groupby('trial'), total=len(df.trial.unique()), desc='Trial-wise regression'):
        raw_gaze = trial_df[['gaze_x', 'gaze_y']].to_numpy().T
        raw_eeg = trial_df[channels].to_numpy().T
        eeg_corrected = raw_eeg.copy() # make a matrix to put the output in 

        if not np.all(np.isnan(raw_gaze)):
            gaze_nans = np.invert(np.any(np.isnan(raw_gaze), axis=0))

            # remove the missing data
            raw_eeg = raw_eeg[:, gaze_nans]
            raw_gaze = raw_gaze[:, gaze_nans]

            # solve and add back to data
            b = np.linalg.inv(raw_gaze @ raw_gaze.T) @ raw_gaze @ raw_eeg.T
            eeg_corrected[:, gaze_nans] = raw_eeg - (raw_gaze.T @ b).T
            
        for idx, time in enumerate(trial_df.time):
            data_dict = {
                'time': time,
                'trial': trial,
                'video_name': trial_df.iloc[idx].video_name,
                'condition': trial_df.iloc[idx].condition,
                'response': trial_df.iloc[idx].response
            }
            # Now add the EEG data for this time point
            data_dict.update({channel: eeg_data for channel, eeg_data in zip(channels, eeg_corrected[:, idx])})
            
            # Append the dictionary to our list
            residuals.append(data_dict)
    return pd.DataFrame(residuals)


def filter_catch_trials(df):
    out = df.copy()
    catch_trials = np.invert(out.condition.to_numpy().astype('bool'))
    response_trials = out.response.to_numpy().astype('bool')
    trial_to_remove = catch_trials + response_trials
    return out[~trial_to_remove].reset_index(drop=True).drop(columns=['condition', 'response'])


def label_repetitions(df):
    out = df.copy()
    out['repetition'] = df.groupby(['time', 'video_name']).cumcount() + 1
    out['even'] = False
    out.loc[(out.repetition % 2) == 0, 'even'] = True
    return out
