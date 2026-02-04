from glob import glob
import argparse
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_scatter(a, b, out_name):
    r = pearsonr(a, b).statistic

    sns.set_context(context='talk', font_scale=0.75)
    _, ax = plt.subplots()
    ax.scatter(a, b, s=30)
    ax.plot([0, 1], [0, 1], 'k')
    ax.set_xlabel('0.5 s ratings')
    ax.set_ylabel('3 s ratings')
    ax.text(0.8, 0.2, f'r = {r:.3f}')
    ax.set_xlim([0,1.1])
    ax.set_ylim([0,1.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_name)

def load_single_file(file_path, n_trials=27,
                     col_names=['video_name', 'interaction', 'loading_error']):
    subj_id = re.search(r'sub-([^_]+)', file_path).group(1)
    # Determine if the file has the header line
    with open(file_path, 'r') as file:
        first_line = file.readline()
    
    # Load the file and rename the columns
    if 'URL' in first_line: 
        df_ = pd.read_csv(file_path, header=0, 
                          names=col_names)
    else:
        df_ = pd.read_csv(file_path, names=col_names)
    
    # Remove subjects if they are missing a catch trial
    # due to a saving error
    n_catch = np.sum(df_.video_name.str.contains('catch'))

    # Remove subjects if they said that every video failed to load
    load_error = np.all(df_['loading_error'].to_numpy())

    if n_catch == 2 and not load_error: 
        # Fix the extra characters in the video names
        df_['video_name'] = df_['video_name'].str.replace("['", "")
        df_['video_name'] = df_['video_name'].str.replace("']", "")

        # Add the subject and condition information 
        df_['subject_id'] = subj_id
        df_['condition'] = re.search(r'condition-([^_]+)', file_path).group(1)

        # If the subject said the video did not load, 
        # remove that trial
        df_ = df_.loc[~df_['loading_error']]
        df_.drop(columns=['loading_error'], inplace=True)

        # If there are duplicate lines, remove them
        return df_.drop_duplicates(keep='last'), subj_id
    else:
        return None, subj_id


def load_data(files, condition_map):
    df_ = []
    bad_subjs = []
    for file in tqdm(files, desc='loading files'):
        file_df, subj_id = load_single_file(file)
        if file_df is not None:
            condition_number = re.search(r'condition-(\d+)', file).group(1)
            file_df['condition'] = condition_map[condition_number]
            file_df['condition_number'] = int(condition_number)
            df_.append(file_df)
        else:
            bad_subjs.append(subj_id)
    df_ = pd.concat(df_, ignore_index=True).reset_index(drop=True)
    return df_, bad_subjs


def get_condition_counts(df, condition_goal=10):
    counts = dict()
    conditions_to_collect = []
    for condition, cond_df in df.groupby('condition_number'):
        condition_count = cond_df.subject_id.nunique()
        counts[condition] = condition_count
        if condition_count < condition_goal: 
            conditions_to_collect.append(condition)
    print(counts)
    print(conditions_to_collect)
    return counts, conditions_to_collect


class BehaviorSummary:
    def __init__(self, args):
        self.process = 'BehaviorSummary'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/data'
        self.behavior_path = f'{self.derivatives_path}/raw/behavior_raw_annotations'
        self.condition_path = f'{self.derivatives_path}/raw/behavior_conditions'
        self.out_path = f'{self.derivatives_path}/interim/{self.process}'
        self.bad_subjs_file = f'{self.out_path}/bad_subjs.csv'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)

    def id_bad_subjs(self, catch_df):
        bad_subjs = []
        for vid, vid_df in catch_df.groupby('video_name'):
            vid_df.reset_index(drop=True, inplace=True)
            vals = vid_df.interaction.to_numpy()
            vals = (vals - 1) / (5 - 1) # Rescale 0-1
            vals = 1 - vals # Flip high-low
            vals = vals.reshape((-1, 1))

            # Calculat the pairwise distance between all subjects
            pair_dist = squareform(pdist(vals, metric='euclidean'))

            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im = ax.imshow(pair_dist)
            ax.set_title(f'{vid}: mean = {vals.mean():.2f}, std = {vals.std():.2f}')
            fig.colorbar(im, cax=cax, orientation='vertical')
            catch_vid = vid.split('.mp4')[0]
            plt.savefig(f'{self.out_path}/{catch_vid}.pdf')

            # Calculate the average distance from each subject to all others
            subj_dist = pair_dist.mean(axis=1)

            # Get the mean and std of the subject dist
            dist_avg = subj_dist.mean()
            dist_std = subj_dist.std()

            # Remove subjects more than 2 stds away from the mean
            dist_thresh = dist_avg + (dist_std * 2)
            print(np.sum(subj_dist > dist_thresh))
            subjs = vid_df.loc[subj_dist > dist_thresh,
                                'subject_id'].tolist()
            bad_subjs += subjs
        bad_subjs = list(np.unique(np.array(bad_subjs)))
        return [str(s) for s in bad_subjs] 


    def clean_data(self, df):
        catch_df = df.loc[df.video_name.str.contains('catch')].reset_index(drop=True)
        bad_subjs = self.id_bad_subjs(catch_df)
        df_out = df.loc[~df.subject_id.isin(bad_subjs)].reset_index(drop=True)
        return df_out, bad_subjs

    def save_bad_subjs(self, bad_subjs):
        df = pd.DataFrame({'subject_id': bad_subjs})
        df.to_csv(self.bad_subjs_file, index=False)

    def save_subject_data(self, df):
        df.to_csv(f'{self.out_path}/subject_rating_data.csv', index=False)

    def save_video_data(self, df):    
        df_ = df.groupby(['condition', 'video_name']).mean(numeric_only=True).reset_index()
        df_.rename(columns={'interaction': 'rating'}, inplace=True)

        df_ = df_.loc[~df_.video_name.str.contains('catch')].reset_index(drop=True)
        df_.rating = (df_.rating - 1) / (5-1) # Rescale from 1-5 to 0-1
        df_.rating = 1 - df_.rating # Flip the scale, so higher is more communicative
        df_.sort_values(by='video_name', inplace=True)
        df_.to_csv(f'{self.out_path}/ratings.csv', index=False)
        return df_
    
    def split_half_reliability(self, df):
        df_ = df.loc[~df.video_name.str.contains('catch')].reset_index(drop=True)

        df_['interaction'] = (df_['interaction']-1)/(5-1) # Rescale from 1-5 to 0-1
        df_['interaction'] = 1 - df_['interaction'] # Flip the scale, so higher is more communicative

        out = {}
        for expert in df_.subject_id.unique():
            loo = df_.loc[df.subject_id == expert]
            loo = loo.drop_duplicates(subset='video_name', keep='last')
            others = df_.loc[df.subject_id != expert]
            others = others.groupby('video_name').mean(numeric_only=True).reset_index()

            cdf_ = loo[['video_name', 'interaction']].merge(others[['video_name', 'interaction']], on='video_name', how='left')

            x = cdf_.interaction_x.to_numpy()
            y = cdf_.interaction_y.to_numpy()
            r = pearsonr(x, y).statistic

            plot_scatter(x, y,
                         f'{self.out_path}/reliability-{expert}.pdf')
            print(f'{expert} reliability with other experts = {r:.3f}')
            out[expert] = loo
        return out

    def get_condition_map(self):
        condition_map = {}
        condition_files = sorted(glob(f'{self.condition_path}/*.csv'))
        for cf in condition_files:
            match = re.search(r'condition-(\d+)', cf)
            if match:
                condition_number = match.group(1)
                with open(cf, 'r') as f:
                    condition_map[condition_number] = f.readline().strip()
        return condition_map
    
    def get_annotations(self):
        df = pd.read_csv(f'{self.derivatives_path}/raw/CaptionData/stimulus_data.csv')
        return df.sort_values(by='video_name').reset_index(drop=True)
    
    def correlate_ratings(self, ratings_500, 
                          ratings_3000):
        for expert, rating_500 in  ratings_500.items():
            rating_500 = rating_500.sort_values(by='video_name')
            rating_500 = rating_500.merge(ratings_3000, on='video_name', how='left')

            x = rating_500.interaction.to_numpy()
            y = rating_500['rating-communication'].to_numpy()

            print(f'{expert} reliability with 3s ratings = {pearsonr(x, y).statistic:.3f}')
            plot_scatter(x, y, f'{self.out_path}/{expert}_rating_comparison.pdf')


    def og_split(self):
        df_ = pd.read_csv(f'{self.derivatives_path}/raw/annotations/individual_subject_ratings.csv')

        df_['repetition'] = df_.groupby('video_name').cumcount()
        df_['even'] = df_['repetition'] % 2 == 0
        df_['likert_response'] = 1 - df_['likert_response']
        df_split = df_.groupby(['video_name', 'even', 'question_name']).mean(numeric_only=True).reset_index()
        df_split = df_split.pivot(index=['video_name', 'even'], columns='question_name',
                                   values='likert_response').reset_index()

        even = df_split.loc[df_split['even']].sort_values('video_name')['communicating'].to_numpy()
        odd = df_split.loc[~df_split['even']].sort_values('video_name')['communicating'].to_numpy()
        print(f'3 s reliability = {pearsonr(even, odd).statistic:.3f}')

    def run(self):
        condition_map = self.get_condition_map()

        raw_data, error_subjs = load_data(sorted(glob(f'{self.behavior_path}/*.csv')),
                                           condition_map)

        # Clean data
        filtered_df = []
        bad_subjs = []
        for _, cond_df in raw_data.groupby('condition'):
            cond_df, cond_bad_subjs = self.clean_data(cond_df)
            bad_subjs += cond_bad_subjs
            filtered_df.append(cond_df)
        filtered_df = pd.concat(filtered_df)

        # self.save_bad_subjs(error_subjs + bad_subjs)
        self.save_subject_data(filtered_df)
        get_condition_counts(filtered_df)

        # split_500 = self.split_half_reliability(filtered_df)
        self.og_split()

        df = self.save_video_data(filtered_df)
        annotations = self.get_annotations()
        # self.correlate_ratings(split_500, annotations)

        # Combine new and old and save
        df.drop(columns=['condition_number', 'condition'], inplace=True)
        df.rename(columns={'rating': 'rating-communication_500ms'}, inplace=True)
        annotations = annotations.merge(df, on='video_name')

        annotations.to_csv(f'{self.out_path}/annotations.csv', index=False)
        annotations.to_csv(f'{self.derivatives_path}/raw/CaptionData/stimulus_data_w_500ms_ratings.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis')
    args = parser.parse_args()
    BehaviorSummary(args).run()

if __name__ == '__main__':
    main()
