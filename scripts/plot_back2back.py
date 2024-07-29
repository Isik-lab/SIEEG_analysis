import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm 
from src.stats import calculate_p, cluster_correction
from scipy import ndimage
from pathlib import Path
import os
from matplotlib.lines import Line2D


swapped = False
rois = ['EVC', 'LOC', 'EBA', 'pSTS', 'aSTS']

if swapped: 
    in_file_prefix = f'data/interim/Back2Back_swapped'
    out_file_prefix = 'data/interim/PlotBack2Back/swapped_'
else:
    in_file_prefix = f'data/interim/Back2Back'
    out_file_prefix = 'data/interim/PlotBack2Back/'

features = ['alexnet', 'moten', 'expanse', 'object',
            'agent_distance', 'facingness',
            'communication', 'joint_action',
            'valence', 'arousal']
if not os.path.isfile(f'{out_file_prefix}plot.csv'):
    back2back_df = []
    for feature in tqdm(features, desc='Feature group summary', leave=True):
        #Load data
        df = []
        files = glob(f'{in_file_prefix}/*{feature}*.parquet')
        for i_file, file in enumerate(files):
            subj_df = pd.read_parquet(file)
            subj_df['eeg_subj_id'] = i_file
            subj_df = subj_df.loc[subj_df.roi_name.isin(rois)].reset_index() # Filter to save time
            df.append(subj_df)
        df = pd.concat(df, ignore_index=True)

        #Average across EEG subjects
        mean_df = df.groupby(['time', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()

        # Average across fMRI subjects
        mean_df = df.groupby(['time', 'roi_name']).mean(numeric_only=True).reset_index()
        mean_df.sort_values(by=['roi_name', 'time'], inplace=True)

        # Group stats
        # Variance
        var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
        scores_var = mean_df[var_cols].to_numpy()
        mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
        mean_df.drop(columns=var_cols, inplace=True)
        # P-values
        null_cols = [col for col in mean_df.columns if 'null_perm_' in col]
        stats_df = []
        for roi_name, roi_df in mean_df.groupby('roi_name'):
            scores_null = roi_df[null_cols].to_numpy().T
            scores = roi_df['value'].to_numpy().T
            ps = calculate_p(scores_null, scores, 5000, 'greater')
            roi_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T,
                                             verbose=True, desc=f'{roi_name} cluster correction')
            roi_df.drop(columns=null_cols, inplace=True)
            stats_df.append(roi_df)
        stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True)
        stats_df['feature'] = feature
        back2back_df.append(stats_df)
    back2back_df = pd.concat(back2back_df, ignore_index=True).reset_index(drop=True)
    back2back_df.to_csv(f'{out_file_prefix}plot.csv', index=False)
else:
    back2back_df = pd.read_csv(f'{out_file_prefix}plot.csv')


# Plot the results
for feature, mean_df in back2back_df.groupby('feature'):
    _, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (roi_name, roi_df) in zip(axes, mean_df.groupby('roi_name')):
        sns.lineplot(x='time', y='value', data=roi_df, ax=ax,
                    zorder=1, color='black')
        ax.fill_between(x=roi_df['time'], 
                        y1=roi_df['low_ci'], y2=roi_df['high_ci'],
                        zorder=0, color='gray', alpha=0.5)

        label, n = ndimage.label(roi_df['p'] < 0.05)
        print(f'{n=}')
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = roi_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
            ax.hlines(y=-0.01, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color='black', zorder=0, linewidth=2)
        if not np.isnan(onset_time):
            ax.set_title(f'{roi_name} ({onset_time:.0f} ms)')
        else:
            ax.set_title(roi_name)
    plt.savefig(f'{out_file_prefix}{feature}.pdf')

rois = ['EVC', 'EBA', 'aSTS']
title_names = ['EVC', 'EBA', 'aSTS-SI']
features = ['alexnet', 'expanse', 'agent_distance', 'communication']
legend_names = ['AlexNet conv2', 'spatial expanse', 'agent distance', 'communication']
back2back_df = back2back_df.loc[back2back_df['roi_name'].isin(rois)].reset_index(drop=True)
back2back_df = back2back_df.loc[back2back_df['feature'].isin(features)].reset_index(drop=True)

back2back_df['roi_name'] = pd.Categorical(back2back_df['roi_name'], categories=rois, ordered=True)
back2back_df['feature'] = pd.Categorical(back2back_df['feature'], categories=features, ordered=True)

stats_pos_start = {'EVC': -.2, 'EBA':  -.12, 'aSTS': -.12}
# Plot the results
sns.set_context('poster')
_, axes = plt.subplots(len(title_names), 1, figsize=(19, 13.25), sharex=True)
axes = axes.flatten()
smooth_kernel = np.ones(30)/30
colors = ['#404040', '#F5DD40', '#8558F4', '#73D2DF']
for (ax, title), (roi_name, cur_df) in zip(zip(axes, title_names), back2back_df.groupby('roi_name')):
    order_counter = 0
    stats_pos = stats_pos_start[roi_name]
    custom_lines = []
    for color, (feature, feature_df) in zip(colors, cur_df.groupby('feature')):
        order_counter +=1

        smoothed_data = {}
        for key in ['low_ci', 'high_ci', 'value']:
            smoothed_data[key] = np.convolve(feature_df[key], smooth_kernel, mode='same')

        alpha = 0.1 if color == '#404040' else 0.2
        ax.fill_between(x=feature_df['time'], 
                    y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                    edgecolor=None, color=color, alpha=alpha, 
                    zorder=order_counter)
        order_counter +=1
        ax.plot(feature_df['time'], smoothed_data['value'],
                color=color, zorder=order_counter,
                linewidth=5)
        custom_lines.append(Line2D([0], [0], color=color, lw=5))

        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 55 if onset_time < 100 else 70
                ax.text(x=onset_time-shift, y=stats_pos-.007,
                        s=f'{onset_time:.0f} ms', fontsize=14)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color=color, zorder=0, linewidth=4)
        stats_pos -= 0.04

    ymin, ymax = ax.get_ylim()
    ax.set_xlim([-200, 1000])
    ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                linestyles='dashed', colors='grey',
                linewidth=5, zorder=0)
    ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                linewidth=5, zorder=0)
    ax.set_ylabel('Prediction ($r$)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)
    if roi_name == 'EVC':
        ax.legend(custom_lines, legend_names, loc='upper right')

ax.set_xlabel('Time (ms)')
plt.savefig(f'{out_file_prefix}feature-roi_plot.pdf')
