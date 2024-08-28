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
if swapped: 
    in_file_prefix = f'data/interim/Back2Back_swapped'
    out_file_prefix = 'data/interim/PlotBack2Back/swapped_'
else:
    in_file_prefix = f'data/interim/Back2Back'
    out_file_prefix = 'data/interim/PlotBack2Back/'

rois = ['EVC', 'MT', 'FFA', 'PPA', 'LOC', 'EBA', 'pSTS', 'aSTS']
features = ['alexnet', 'moten', 'expanse', 'object',
            'agent_distance', 'facingness',
            'joint_action','communication', 
            'valence', 'arousal']
roi_titles = ['EVC', 'MT', 'FFA', 'PPA', 'LOC', 'EBA', 'pSTS-SI', 'aSTS-SI']
feature_titles = ['AlexNet-conv2', 'motion energy', 
                  'spatial expanse', 'object directedness',
                  'agent distance', 'facingness',
                  'joint action', 'communication', 'valence', 'arousal']
colors = ['#404040', '#404040', '#F5DD40', '#F5DD40', '#8558F4', '#8558F4', '#73D2DF', '#73D2DF', '#D57D7F', '#D57D7F']


reduced_rois = ['EVC', 'LOC', 'aSTS']
reduced_features = ['alexnet', 'expanse', 'agent_distance', 'communication']
reduced_rois_titles = ['EVC', 'LOC', 'aSTS-SI']
reduced_features_legends = ['AlexNet conv2', 'spatial expanse', 'agent distance', 'communication']
reduced_colors = ['#404040', '#F5DD40', '#8558F4', '#73D2DF']

smooth_kernel = np.ones(10)/10

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


full_stats_df = back2back_df.loc[back2back_df['feature'].isin(features)].reset_index(drop=True)
full_stats_df['feature'] = pd.Categorical(full_stats_df['feature'], categories=features, ordered=True)
full_stats_df = full_stats_df.loc[full_stats_df['roi_name'].isin(rois)].reset_index(drop=True)
full_stats_df['roi_name'] = pd.Categorical(full_stats_df['roi_name'], categories=rois, ordered=True)

# Plot the results
ymin = -0.2
for roi, (roi_name, roi_df) in zip(roi_titles, full_stats_df.groupby('roi_name', observed=True)):
    sns.set_context(context='paper', font_scale=2)
    fig, axes = plt.subplots(5, 2, figsize=(19, 15.83), sharex=True, sharey=True)
    axes = axes.flatten()
    ymax = 0.4 if roi != 'EVC' else 0.75

    order_counter = 0
    stats_pos = -.12
    for ifeature, (feature_name, feature_df) in enumerate(roi_df.groupby('feature', observed=True)):
        feature, color, ax = feature_titles[ifeature], colors[ifeature], axes[ifeature]
        alpha = 0.1 if color == 'black' else 0.2
        alpha += 0.2 if color == '#F5DD40' else 0
        smoothed_data = {}
        for key in ['low_ci', 'high_ci', 'value']:
            smoothed_data[key] = np.convolve(feature_df[key], smooth_kernel, mode='same')

        ax.fill_between(x=feature_df['time'], 
                    y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                    edgecolor=None, color=color, alpha=alpha, 
                    zorder=order_counter)
        order_counter +=1
        ax.plot(feature_df['time'], smoothed_data['value'],
                color=color, zorder=order_counter,
                linewidth=5)

        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 100 if onset_time < 100 else 110
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                        s=f'{onset_time:.0f} ms',
                        fontsize=12)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color=color, zorder=0, linewidth=2)

        ax.set_title(feature)
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                    linestyles='dashed', colors='grey',
                    linewidth=3, zorder=0)
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                    linewidth=3, zorder=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([ymin, ymax])
        if ifeature % 2 == 0:
            ax.set_ylabel('Prediction ($r$)')

        if ifeature >= 8:
            ax.set_xlabel('Time (ms)')

    fig.suptitle(roi)
    plt.tight_layout()
    plt.savefig(f'{out_file_prefix}{roi}.pdf')

back2back_df = back2back_df.loc[back2back_df['roi_name'].isin(reduced_rois)].reset_index(drop=True)
back2back_df = back2back_df.loc[back2back_df['feature'].isin(reduced_features)].reset_index(drop=True)

back2back_df['roi_name'] = pd.Categorical(back2back_df['roi_name'], categories=reduced_rois, ordered=True)
back2back_df['feature'] = pd.Categorical(back2back_df['feature'], categories=reduced_features, ordered=True)

stats_pos_start = {'EVC': -.2, 'LOC':  -.12, 'aSTS': -.12}
# Plot the results
sns.set_context(context='poster', font_scale=1.25)
_, axes = plt.subplots(len(reduced_rois_titles), 1, figsize=(19, 13.25), sharex=True)
axes = axes.flatten()
for iroi, (roi_name, cur_df) in enumerate(back2back_df.groupby('roi_name', observed=True)):
    ax, title = axes[iroi], reduced_rois_titles[iroi]
    order_counter = 0
    stats_pos = stats_pos_start[roi_name]
    custom_lines = []
    for color, (feature, feature_df) in zip(reduced_colors, cur_df.groupby('feature', observed=True)):
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
                shift = 60 if onset_time < 100 else 75
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                        s=f'{onset_time:.0f} ms',
                        fontsize=15.5)
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


axes[0].legend(custom_lines, reduced_features_legends,
            loc='upper right', fontsize='18')
ax.set_xlabel('Time (ms)')
plt.savefig(f'{out_file_prefix}feature-roi_plot.pdf')
