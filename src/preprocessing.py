import numpy as np
from src import temporal
import pandas as pd
from tqdm import tqdm 



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
                'response': trial_df.iloc[idx].response,
                'stimulus_set': trial_df.iloc[idx].stimulus_set
            }
            # Now add the EEG data for this time point
            data_dict.update({channel: eeg_data for channel, eeg_data in zip(channels, eeg_corrected[:, idx])})
            
            # Append the dictionary to our list
            residuals.append(data_dict)
    return pd.DataFrame(residuals)


def filter_catch_trials(df):
    out = df.copy()
    catch_trials = np.invert(df['condition'].to_numpy().astype('bool'))
    response_trials = df['response'].to_numpy().astype('bool')
    trial_to_remove = catch_trials + response_trials
    return out[~trial_to_remove].reset_index(drop=True).drop(columns=['condition', 'response'])


def label_repetitions(df):
    out = df.copy()
    out['repetition'] = df.groupby(['time', 'video_name']).cumcount() + 1
    out['even'] = False
    out.loc[(out.repetition % 2) == 0, 'even'] = True
    return out
