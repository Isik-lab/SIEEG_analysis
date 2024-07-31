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


rois = ['EVC', 'LOC', 'EBA', 'pSTS', 'aSTS']
legend_names = ['EVC', 'LOC', 'EBA', 'pSTS-SI', 'aSTS-SI']
out_path = 'data/interim/PlotROIDecoding'
Path(out_path).mkdir(exist_ok=True, parents=True)
files = glob('data/interim/fMRIRegression/*rois.parquet')

if not os.path.isfile(f'{out_path}/roi_plot.csv'):
    #Load data
    df = []
    for file in tqdm(files, desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    df = df.loc[df.roi_name.isin(rois)].reset_index() # Filter to save time
    print('Finished loading files')

    #Average across EEG subjects
    mean_df = df.groupby(['time', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')

    # Average across fMRI subjects
    mean_df = df.groupby(['time', 'roi_name']).mean(numeric_only=True).reset_index()
    print('Finished mean over fMRI subjects')
    mean_df.sort_values(by=['roi_name', 'time'], inplace=True)
    print(f'{mean_df.roi_name.unique()=}')

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
                                            verbose=True,
                                            desc=f'{roi_name} cluster correction')
        roi_df.drop(columns=null_cols, inplace=True)
        stats_df.append(roi_df)
    stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True)
    stats_df.to_csv(f'{out_path}/roi_plot.csv', index=False)
else:
    stats_df = pd.read_csv(f'{out_path}/roi_plot.csv')

# Make categorical for plotting
rois = ['EVC', 'EBA', 'aSTS']
title_names = ['EVC', 'EBA', 'aSTS-SI']
stats_df = stats_df.loc[stats_df['roi_name'].isin(rois)].reset_index(drop=True)
stats_df['roi_name'] = pd.Categorical(stats_df['roi_name'], categories=rois, ordered=True)

# Plot the results
sns.set_context(context='poster', font_scale=1.5)
_, ax = plt.subplots(figsize=(19, 9.5))
colors = ['black', '#976A9A', '#407FAA']
order_counter = 0
stats_pos = -.12
custom_lines = []
smooth_kernel = np.ones(10)/10
for (roi_name, roi_df), color in zip(stats_df.groupby('roi_name'), colors):
    order_counter +=1
    alpha = 0.1 if color == 'black' else 0.2
    smoothed_data = {}
    for key in ['low_ci', 'high_ci', 'value']:
        smoothed_data[key] = np.convolve(roi_df[key], smooth_kernel, mode='same')

    ax.fill_between(x=roi_df['time'], 
                y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                edgecolor=None, color=color, alpha=alpha, 
                zorder=order_counter)
    order_counter +=1
    ax.plot(roi_df['time'], smoothed_data['value'],
            color=color, zorder=order_counter,
            linewidth=5)
    custom_lines.append(Line2D([0], [0], color=color, lw=4))

    label, n = ndimage.label(roi_df['p'] < 0.05)
    onset_time = np.nan
    for icluster in range(1, n+1):
        time_cluster = roi_df['time'].to_numpy()[label == icluster]
        if icluster == 1:
            onset_time = time_cluster.min()
            shift = 75 if onset_time < 100 else 90
            ax.text(x=onset_time-shift, y=stats_pos-.006,
                    s=f'{onset_time:.0f} ms',
                    fontsize=20)
        ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                  xmax=time_cluster.max(),
                  color=color, zorder=0, linewidth=4)
    stats_pos -= 0.05

ax.legend(custom_lines, title_names, loc='upper right')
ymin, ymax = ax.get_ylim()
ax.set_xlim([-200, 1000])
ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
            linestyles='dashed', colors='grey',
            linewidth=5, zorder=0, alpha=0.5)
ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
            linewidth=5, zorder=0, alpha=0.5)
ax.set_ylabel('Prediction ($r$)')
ax.set_xlabel('Time (ms)')
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim([ymin, ymax])

plt.tight_layout()
plt.savefig(f'{out_path}/roi_plot.pdf')
