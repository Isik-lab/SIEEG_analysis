import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm 
from src.stats import calculate_p, cluster_correction
from scipy import ndimage


for feature in ['alexnet']:#['moten', 'alexnet']:
    #Load data
    df = []
    files = glob(f'data/interim/Back2Back/sub-06_x2-{feature}*.parquet')
    for i_file, file in tqdm(enumerate(files), total=len(files), desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = i_file
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    print('Finished loading files')

    #Average across EEG subjects
    mean_df = df.groupby(['time', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')

    # Average across fMRI subjects
    mean_df = df.groupby(['time', 'roi_name']).mean(numeric_only=True).reset_index()
    print('Finished mean over fMRI subjects')
    mean_df.sort_values(by=['roi_name', 'time'], inplace=True)

    # Group stats
    # Variance
    scores_var = mean_df[[col for col in mean_df.columns if 'var_perm_' in col]].to_numpy()
    mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
    # P-values
    scores_null = mean_df[[col for col in mean_df.columns if 'null_perm_' in col]].to_numpy().T
    scores = mean_df['value'].to_numpy().T
    ps = calculate_p(scores_null, scores, 5000, 'greater')
    mean_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T,
                                      verbose=True,
                                      desc='cluster correction')

    # Plot the results
    _, axes = plt.subplots(3, 3, sharex=True, sharey=True)
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
    plt.savefig(f'data/interim/PlotBack2Back/{feature}.pdf')
