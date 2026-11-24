import pandas as pd
from pathlib import Path
from glob import glob 
from tqdm import tqdm
import re
import numpy as np


def load_behavior(path):
    """load behavioral annotations

    Args:
        path (str): fMRI benchmark directory

    Returns:
        pandas.core.frame.DataFrame:
            behavioral annotations loading
    """
    return pd.read_csv(f'{path}/stimulus_data.csv')


def roi_fmri_summary(path):
    """Summarize the fMRI responses to model the average
     response in each ROI rather than the voxelwise responses

    Args:
        path (str): fMRI benchmark directory

    Returns:
        response data (pandas.core.frame.DataFrame): fMRI data organized per ROI and subj
        metadata (pandas.core.frame.DataFrame): Info about each of the fMRI targets
    """
    #load the data
    metadata = pd.read_csv(f'{path}/metadata.csv')
    response_data = pd.read_csv(f'{path}/response_data.csv.gz')
    video_names = pd.read_csv(f'{path}/stimulus_data.csv').video_name.to_list()

    #summarize by roi
    out_response = []
    out_meta = []
    for (subj, roi), meta in metadata.groupby(['subj_id', 'roi_name']):
        if roi != 'none':
            ids = meta.voxel_id.to_list()
            responses = response_data.iloc[ids].mean().to_list()
            out_meta.append({'subj_id': subj, 'roi_name': roi})
            for video_name, response in zip(video_names, responses):
                out_response.append({'id': f'sub-{subj}_roi-{roi}',
                                     'video_name': video_name,
                                     'response': response})
    out_response = pd.DataFrame(out_response).pivot(index='video_name',
                                                    columns='id',
                                                    values='response').reset_index(drop=True)
    return out_response, pd.DataFrame(out_meta)


def load_fmri(path, roi_mean=True, smoothing=False):
    """load fMRI data

    Args:
        path (str): fMRI benchmark directory
        roi_mean (bool, optional): Return the average ROI response instead of voxelwise response. Default is False
        smoothing (bool, optional): Load the fMRI data with smoothed betas. Only valid if roi_mean is False. Default is False

    Returns:
        response data (pandas.core.frame.DataFrame): reorganized fMRI responses
        metadata (pandas.core.frame.DataFrame): Info about each target in the fMRI data
    """
    if roi_mean:
        return roi_fmri_summary(path)
    else:
        if not smoothing:
            return pd.read_csv(f'{path}/response_data.csv.gz').T, pd.read_csv(f'{path}/metadata.csv')
        else:
            return pd.read_csv(f'{path}/response_data_smoothed.csv.gz').T, pd.read_csv(f'{path}/metadata.csv')



def load_eeg(file_path):
    """Load EEG data

    Args:
        file_path (str): path to preprocessed EEG data file

    Returns:
        eeg_data (pandas.core.frame.DataFrame): preproceesed EEG data
    """
    if 'csv' in file_path:
        return pd.read_csv(file_path)
    elif 'parquet' in file_path:
        return pd.read_parquet(file_path)


def load_model_activations(file_path):
    """Load AlexNet or motion energy activation data

    Args:
        file_path (str): path to activations
    Returns:
        array (numpy.ndarray): activations
    """
    return np.load(file_path)


def check_videos(eeg_df, annotation_df, other=None):
    """Filter the videos in the feature DataFrame based on those that are present in the EEG data. 
        Data may be missing from a particular video following data cleaning.
        It also reorders the data so that the EEG and annotation data are in the same order.

    Args:
        eeg_df (pandas.core.frame.DataFrame): eeg data frame
        annotation_df (pandas.core.frame.DataFrame): annotation data frame
        other (list of pandas.core.frame.DataFrame or np.ndarray, optional): fMRI data frame, alexnet, moten, etc. Defaults to None

    Returns:
        pandas.core.frame.DataFrame: filtered annotation dataframe
    """
    # Get EEG videos and sort
    eeg_videos = eeg_df.video_name.unique()
    eeg_out = eeg_df.sort_values('video_name').reset_index(drop=True)

    # Clean the behavioral data and get the indices for other if not None
    filtered_annot = annotation_df[annotation_df['video_name'].isin(eeg_videos)]
    anot_idx = filtered_annot.sort_values('video_name').index
    annot_out = filtered_annot.sort_values('video_name').reset_index(drop=True)

    if other is None:
        return eeg_out, annot_out
    else:
        other_out = []
        for item in other:
            if type(item) == pd.core.frame.DataFrame:
                other_out.append(item.iloc[anot_idx].reset_index(drop=True))
            elif type(item) == np.ndarray:
                other_out.append(item[anot_idx])
                
        return eeg_out, annot_out, other_out


def strip_eeg(eeg_df):
    """_summary_

    Args:
        eeg_df (pandas.core.frame.DataFrame): dataframe with the eeg data

    Returns:
        pandas.core.frame.DataFrame: dataframe with just the channel-wise response data
    """
    return eeg_df.pivot(index='video_name', columns='channel', values='signal')


def get_subj_time(file_name):
    """_summary_

    Args:
        file_name (_type_): _description_

    Returns:
        subject_number (int): the number of the subject 
        subject_number (int): the number of the subject 
    """
    pattern = re.compile(r'sub-(\d+)_time-(\d+)')
    match = pattern.search(file_name)
    if match:
        subject_number = int(match.group(1))
        time_number = match.group(2)
        return subject_number, time_number
    else:
        return None


def load_decoding_files(path, file_pattern, targets):
    files = glob(f'{path}/{file_pattern}')
    out = []
    for file in tqdm(files, desc='Loading files'):
        sub, time = get_subj_time(file)
        df = pd.read_csv(file).rename(columns={'0': 'r'})
        df['subj_id'] = sub
        df['time_id'] = time
        #add the targets based on the dictionary
        for key, val in targets.items():
            df[key] = val
        
        #average over ROIs in the fMRI data
        if len(df) > df['targets'].nunique():
            df = df.groupby(['time_id', 'targets']).mean(numeric_only=True).reset_index()
        out.append(df)
    return pd.concat(out)
