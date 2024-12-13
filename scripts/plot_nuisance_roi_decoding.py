import argparse
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


def load_and_summarize(files):
    #Load data
    df = []
    for file in tqdm(files, desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
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
        scores = roi_df['score'].to_numpy().T
        ps = calculate_p(scores_null, scores, 5000, 'greater')
        roi_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T,
                                            verbose=True,
                                            desc=f'{roi_name} cluster correction')
        roi_df.drop(columns=null_cols, inplace=True)
        stats_df.append(roi_df)
    df.drop(columns=var_cols, inplace=True)
    return df, pd.concat(stats_df, ignore_index=True).reset_index(drop=True) 

# Plot the results
def plot_simple(out_file, stats_df, rois, colors, title_names): 
    sns.set_context(context='poster', font_scale=1.5)
    _, ax = plt.subplots(figsize=(19, 9.5))

    order_counter = 0
    stats_pos = -.12
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for (roi_name, roi_df), color in zip(stats_df.groupby('roi_name', observed=True), colors):
        order_counter +=1
        alpha = 0.1 if color == 'black' else 0.2
        smoothed_data = {}
        for key in ['low_ci', 'high_ci', 'score']:
            smoothed_data[key] = np.convolve(roi_df[key], smooth_kernel, mode='same')

        ax.fill_between(x=roi_df['time'], 
                    y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                    edgecolor=None, color=color, alpha=alpha, 
                    zorder=order_counter)
        order_counter +=1
        ax.plot(roi_df['time'], smoothed_data['score'],
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
    plt.savefig(out_file)


def plot_full(out_file, stats_df, rois, colors, title_names):
    sns.set_context(context='paper', font_scale=2)
    _, axes = plt.subplots(4, 2, figsize=(19, 12.67), sharex=True, sharey=True)
    axes = axes.flatten()
    ymin, ymax = -0.15, 0.6

    order_counter = 0
    stats_pos = -.12
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for iroi, (roi_name, roi_df) in enumerate(stats_df.groupby('roi_name', observed=True)):
        ax, color, roi = axes[iroi], colors[iroi], title_names[iroi]
        order_counter +=1
        alpha = 0.1 if color == 'black' else 0.2
        smoothed_data = {}
        for key in ['score', 'low_ci', 'high_ci']:
            smoothed_data[key] = np.convolve(roi_df[key], smooth_kernel, mode='same')

        ax.fill_between(x=roi_df['time'], 
                    y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                    edgecolor=None, color=color, alpha=alpha, 
                    zorder=order_counter)
        order_counter +=1
        ax.plot(roi_df['time'], smoothed_data['score'],
                color=color, zorder=order_counter,
                linewidth=5)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

        label, n = ndimage.label(roi_df['p'] < 0.05)
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = roi_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 75 if onset_time < 100 else 95
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                        s=f'{onset_time:.0f} ms',
                        fontsize=12)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color=color, zorder=0, linewidth=2)

        ax.set_title(roi)
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                    linestyles='dashed', colors='grey',
                    linewidth=3, zorder=0, alpha=0.5)
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                    linewidth=3, zorder=0, alpha=0.5)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([ymin, ymax])
        if iroi % 2 == 0:
            ax.set_ylabel('Prediction ($r$)')

        if iroi >= 6:
            ax.set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig(out_file)


class PlotNuisanceROIDecoding:
    def __init__(self, args):
        self.out_dir = args.out_dir 
        self.out_csv = args.out_csv
        self.regression_dir = args.regression_dir 
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        self.simplified_plotting = args.simplified_plotting
        self.overwrite = args.overwrite 
        print(vars(self))

    def run(self):
        if self.simplified_plotting:
            rois = ['EVC', 'LOC', 'aSTS']
            title_names = ['EVC', 'LOC', 'aSTS-SI']
            colors = ['black', '#976A9A', '#407FAA']
            out_plot = f'{self.out_dir}/roi_plot.pdf'
        else:
            rois = ['EVC', 'MT', 'FFA', 'PPA', 'LOC',
                    'EBA', 'pSTS', 'aSTS']
            title_names = ['EVC', 'MT', 'FFA', 'PPA',
                           'LOC', 'EBA', 'pSTS-SI', 'aSTS-SI']
            colors = ['black', 'black', '#497059', '#497059',
                      '#976A9A', '#976A9A', '#407FAA', '#407FAA']
            out_plot = f'{self.out_dir}/all_rois_plot.pdf'

        if self.overwrite or not Path(f'{self.out_dir}/roi_plot.csv').is_file():
            files = glob(f'{self.regression_dir}/*roi*.parquet')
            all_df, df = load_and_summarize(files)
            all_df.to_parquet(f'{self.out_dir}/individual_stats.parquet', index=False)
            df.to_csv(f'{self.out_dir}/roi_plot.csv', index=False)
        else:
            df = pd.read_csv(f'{self.out_dir}/roi_plot.csv')

        # Make categorical for plotting
        df = df.loc[df['roi_name'].isin(rois)].reset_index(drop=True)
        df['roi_name'] = pd.Categorical(df['roi_name'], categories=rois, ordered=True)
        print(df.head())

        if self.simplified_plotting:
            plot_simple(out_plot, df, rois, colors, title_names)
        else: 
            plot_full(out_plot, df, rois, colors, title_names)


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--simplified_plotting', action=argparse.BooleanOptionalAction, default=False,
                        help='plot all or only select ROIs')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotNuisanceROIDecoding')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='roi_plot.csv')
    parser.add_argument('--regression_dir', '-r', type=str, help='directory for input',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fMRINuisanceRegression')
    args = parser.parse_args()
    PlotNuisanceROIDecoding(args).run()


if __name__ == '__main__':
    main()