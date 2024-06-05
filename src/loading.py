import pandas as pd
from pathlib import Path


def load_behavior(path):
    """load behavioral annotations

    Args:
        path (str): fMRI benchmark directory

    Returns:
        pandas.core.frame.DataFrame:
            behavioral annotations loading
    """
    return pd.read_csv(f'{path}/stimulus_data.csv')


def load_fmri(path):
    """load fMRI data

    Args:
        path (str): fMRI benchmark directory

    Returns:
        response data (pandas.core.frame.DataFrame): reorganized fMRI responses
        metadata (pandas.core.frame.DataFrame): Info about each target in the fMRI data
    """
    return pd.read_csv(f'{path}/response_data.csv.gz'), pd.read_csv(f'{path}/metadata.csv')


def load_eeg(file_path):
    """Load EEG data

    Args:
        file_path (str): path to preprocessed EEG data file

    Returns:
        eeg_data (pandas.core.frame.DataFrame): preproceesed EEG data
    """
    return pd.read_csv(file_path)


def check_videos(eeg_df, annotation_df, fmri_df=None):
    """Filter the videos in the feature DataFrame based on those that are present in the EEG data. 
        Data may be missing from a particular video following data cleaning.
        It also reorders the data so that the EEG and annotation data are in the same order.

    Args:
        eeg_df (pandas.core.frame.DataFrame): eeg data frame
        annotation_df (pandas.core.frame.DataFrame): annotation data frame
        fmri_df (pandas.core.frame.DataFrame, optional): fMRI data frame. Defaults to None

    Returns:
        pandas.core.frame.DataFrame: filtered annotation dataframe
    """
    # Get EEG videos and sort
    eeg_videos = eeg_df.video_name.unique()
    eeg_out = eeg_df.sort_values('video_name').reset_index(drop=True)

    # Clean the behavioral data and get the indices for fmri_df if not None
    filtered_annot = annotation_df[annotation_df['video_name'].isin(eeg_videos)]
    anot_idx = filtered_annot.sort_values('video_name').index
    annot_out = filtered_annot.sort_values('video_name').reset_index(drop=True)

    if fmri_df is None:
        return eeg_out, annot_out
    else:
        fmri_out = fmri_df.iloc[anot_idx].reset_index(drop=True)
        return eeg_out, annot_out, fmri_out



def strip_eeg(eeg_df):
    """_summary_

    Args:
        eeg_df (pandas.core.frame.DataFrame): dataframe with the eeg data

    Returns:
        pandas.core.frame.DataFrame: dataframe with just the channel-wise response data
    """
    cols = [col for col in eeg_df.columns if 'channel' in col]
    return eeg_df[cols]
