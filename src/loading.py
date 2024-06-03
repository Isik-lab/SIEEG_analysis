import pandas as pd
from pathlib import Path


def load_annotations(path):
    """load behavioral annotations

    Args:
        path (str): fMRI benchmark directory

    Returns:
        annotations (pandas.core.frame.DataFrame):
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


def create_parent_directories(file_path):
    # Get the directory name from the file path
    directory = Path(file_path).parent
    
    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)
