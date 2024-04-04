#
import os
import argparse
from glob import glob
import numpy as np
from numpy import matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src import plotting
from src.stats import calculate_p, cluster_correction
from scipy import ndimage
from tqdm import tqdm
import warnings


def load_files(files, cat=None, cat_order=None, subj_exclude=[]):
    group = []
    subjs = []
    for file in tqdm(files, desc='File loading'):
        subj = file.split('/')[-1].split('_')[0]
        if subj not in subj_exclude: 
            subjs.append(subj)
            if 'pkl' in file:
                df = pd.read_pickle(file)
            else:
                df = pd.read_csv(file)
            df['subj'] = subj
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
            group.append(df)
    group = pd.concat(group)
    if cat and cat_order:
        cat_type = pd.CategoricalDtype(categories=cat_order, ordered=True)
        group[cat] = group[cat].astype(cat_type)
    return group, subjs


def mean_of_arrays(series):
    # Stack arrays vertically and compute mean along the first axis (rows)
    return np.nanmean(np.vstack(series),axis=0)


def compute_confidence_intervals(arr):
    # Define a function to calculate the 2.5% and 97.5% percentiles of an array
    lower = np.nanpercentile(arr, 2.5)
    upper = np.nanpercentile(arr, 97.5)
    return lower, upper


def cluster_correction_df(group, column_name):
    feature_name = group[column_name].iloc[0]
    rs = group['r2'].to_numpy()
    ps = group['p'].to_numpy()
    r_nulls = np.vstack(group['r2_null'].to_numpy())
    corrected_ps = cluster_correction(rs, ps, r_nulls, desc=f'{feature_name} progress')
    return  pd.Series(corrected_ps, index=group['time'])


def calculate_p_df(row):
    r_value = row['r2']  # The 'r' value for the current row
    r_null_array = row['r2_null']  # The 'r_null' array for the current row
    return calculate_p(r_null_array, r_value, n_perm_=len(r_null_array), H0_='greater')


class fMRI_GroupAnalysis:
    def __init__(self, args):
        self.process = 'fMRI_GroupAnalysis'
        self.input_path = f'{args.top_dir}/data/interim'
        self.out_path = f'{args.top_dir}/data/interim/{self.process}'
        self.group_fmri_file = f'{self.out_path}/fmri_roi_decoding.csv'
        self.rois = ['EVC', 'MT', 'EBA',
                     'LOC', 'FFA', 'PPA',
                     'pSTS', 'face-pSTS', 'aSTS']
        self.regress_gaze = args.regress_gaze
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        self.input_pattern = f'{self.input_path}/fMRIDecoding/sub-*_reg-gaze-{self.regress_gaze}_roi-decoding.pkl.gz'

    def run(self):
        df, subjs = load_files(glob(self.input_pattern),
                            cat='roi', cat_order=self.rois)
        df.drop(columns=['r2_null', 'r2_var'], inplace=True)
        print(sorted(subjs))
        stats = True if 'r2_null' in df.columns.tolist() else False
        
        if stats: 
            # Average across subjects
            mean_df = df.groupby(['regression_type', 'category',
                                  'time', 'roi', 'fmri_sid'
                                ], observed=True).agg({
                                    'r2': 'mean',  # For scalar values, use the built-in 'mean' function
                                    'r2_null': mean_of_arrays,  # For numpy arrays, use the custom function
                                    'r2_var': mean_of_arrays
                                }).reset_index()
            
            # Compute the confidence intervals across subjec ts
            mean_df[['lower_ci', 'upper_ci']] = mean_df['r2_var'].apply(lambda arr: pd.Series(compute_confidence_intervals(arr)))

            # Calculate the p values across subjects
            mean_df['p'] = mean_df.apply(calculate_p_df, axis=1)
            
            # Perform cluster correction on the p values
            result = []
            for category, r_category in tqdm(mean_df.groupby('category'), desc='Category correction'): 
                r_category = r_category.sort_values(by=['roi', 'time']).reset_index(drop=True)
                corrected_p_series = r_category.groupby('roi', observed=True).apply(cluster_correction_df, column_name='roi')
                corrected_p_series = corrected_p_series.apply(pd.Series).stack().reset_index(name='p_corrected')
                corrected_p_series['category'] = category
                result.append(mean_df.merge(corrected_p_series, on=['category', 'roi', 'time'])) # Combine with original data frame
            result = pd.concat(result)

            # Make a column for plotting the significant clusters
            result['sig_plotting'] = np.nan
            result.loc[result.p_corrected < 0.05, 'sig_plotting'] = -0.05

            # Drop the array columns which have been used to compute p values and CIs 
            result.drop(columns=['r2_null', 'r2_var'], inplace=True)
        else:
            result = df.groupby(['regression_type', 'category',
                                'time', 'roi', 'fmri_sid'],
                                observed=True).agg({'r2': 'mean'}).sort_index().reset_index()

        # Save the output to disk
        result.to_csv(self.group_fmri_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis')  
    args = parser.parse_args()
    fMRI_GroupAnalysis(args).run()


if __name__ == '__main__':
    main()