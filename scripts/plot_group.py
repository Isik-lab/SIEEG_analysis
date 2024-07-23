#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from pathlib import Path

outdir = 'data/interim/GroupPlots'
Path(outdir).mkdir(exist_ok=True, parents=True)

df = pd.read_csv('data/interim/groupAnalysis/feature_decoding_summary.csv')
_, axes = plt.subplots(3, 3, sharex=True,
                    sharey=True, figsize=(10, 10))
axes = axes.flatten()
for (feature, feature_df), ax in zip(df.groupby('target'), axes):
    sns.lineplot(x='time', y='r2',
                 data=feature_df, ax=ax,
                 zorder=1)
    time = feature_df.time.to_numpy()
    y1 = feature_df.lower_ci.to_numpy()
    y2 = feature_df.upper_ci.to_numpy()
    p = feature_df.p.to_numpy()
    ax.fill_between(time, y1, y2, alpha=0.5,
                    color='gray', zorder=0)
    label, n = ndimage.label(p < 0.05)
    print(f'{n=}')
    onset_time = np.nan
    for icluster in range(1, n+1):
        time_cluster = time[label == icluster]
        if icluster == 1:
            onset_time = time_cluster.min()
        ax.hlines(y=-0.01, xmin=time_cluster.min(),
                xmax=time_cluster.max(),
                color='black', zorder=0, linewidth=2)
    if not np.isnan(onset_time):
        ax.set_title(f'{feature} ({int(onset_time)} ms)')
    else:
        ax.set_title(feature)
plt.savefig(f'{outdir}/features.pdf')


file_dict = {'fmri': 'data/interim/groupAnalysis/fmri_encoding_summary.csv',
             'shared': 'data/interim/groupAnalysis/eeg_feature_shared_summary.csv'}
for key, file in file_dict.items():
    df = pd.read_csv(file)
    for subj, subj_df in df.groupby('subj_id'):
        _, axes = plt.subplots(3, 3, sharex=True,
                            sharey=True, figsize=(10, 10))
        axes = axes.flatten()
        for (roi, roi_df), ax in zip(subj_df.groupby('target'), axes):
            sns.lineplot(x='time', y='r2',
                        data=roi_df, ax=ax, zorder=1)
            time = roi_df.time.to_numpy()
            y1 = roi_df.lower_ci.to_numpy()
            y2 = roi_df.upper_ci.to_numpy()
            p = roi_df.p.to_numpy()
            ax.fill_between(time, y1, y2, alpha=0.5,
                            color='gray', zorder=0)
            label, n = ndimage.label(p < 0.05)
            print(f'{n=}')
            onset_time = np.nan
            for icluster in range(1, n+1):
                time_cluster = time[label == icluster]
                if icluster == 1:
                    onset_time = time_cluster.min()
                ax.hlines(y=-0.01, xmin=time_cluster.min(),
                        xmax=time_cluster.max(),
                        color='black', zorder=0, linewidth=2)
            if not np.isnan(onset_time):
                ax.set_title(f'{roi} ({int(onset_time)} ms)')
            else:
                ax.set_title(roi)
        plt.savefig(f'{outdir}/{key}_{subj}.pdf')
