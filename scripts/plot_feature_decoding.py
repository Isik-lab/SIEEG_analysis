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
    mean_df = df.groupby(['time', 'feature']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')
    # Group stats
    # Variance
    var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
    scores_var = mean_df[var_cols].to_numpy()
    mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
    mean_df.drop(columns=var_cols, inplace=True)
    # P-values
    null_cols = [col for col in mean_df.columns if 'null_perm_' in col]
    stats_df = []
    for feature, feature_df in mean_df.groupby('feature'):
        scores_null = feature_df[null_cols].to_numpy().T
        scores = feature_df['value'].to_numpy().T
        ps = calculate_p(scores_null, scores, 5000, 'greater')
        feature_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T,
                                            verbose=True,
                                            desc=f'{feature} cluster correction')
        feature_df.drop(columns=null_cols, inplace=True)
        stats_df.append(feature_df)
    return pd.concat(stats_df, ignore_index=True).reset_index(drop=True)

# Plot the results
def plot_simple(out_file, stats_df, features, colors, title_names): 
    sns.set_context(context='poster', font_scale=1.5)
    _, ax = plt.subplots(figsize=(19, 9.5))
    order_counter = 0
    stats_pos = -.12
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for (feature, feature_df), color in zip(stats_df.groupby('feature', observed=True), colors):
        order_counter +=1
        alpha = 0.1 if color == 'black' else 0.2
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
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 75 if onset_time < 100 else 90
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                        s=f'{onset_time:.0f} ms',
                        fontsize=20)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color=color, zorder=0, linewidth=4)
        stats_pos -= 0.02

    ax.legend(custom_lines, title_names, loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_xlim([-200, 1000])
    ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                linestyles='dashed', colors='grey',
                linewidth=5, zorder=0)
    ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                linewidth=5, zorder=0)
    ax.set_ylabel('Prediction ($r$)')
    ax.set_xlabel('Time (ms)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([ymin, ymax])

    plt.tight_layout()
    plt.savefig(out_file)


def plot_full(out_file, stats_df, features, colors, title_names):
    sns.set_context(context='paper', font_scale=2)
    _, axes = plt.subplots(4, 2, figsize=(19, 12.67), sharex=True, sharey=True)
    axes = axes.flatten()
    ymin, ymax = -0.15, 0.425

    order_counter = 0
    stats_pos = -.12
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for ifeature, (_, feature_df) in enumerate(stats_df.groupby('feature', observed=True)):
        ax, color, feature = axes[ifeature], colors[ifeature], title_names[ifeature]
        order_counter +=1
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
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 75 if onset_time < 100 else 95
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

        if ifeature >= 6:
            ax.set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig(out_file)


class PlotFeatureDecoding:
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
            features = ['expanse', 'agent_distance', 'communication']
            title_names = ['spatial expanse', 'agent distance', 'communication']
            colors = ['#F5DD40', '#8558F4', '#73D2DF']
            out_plot = f'{self.out_dir}/feature_plot.pdf'
        else:
            features = ['expanse', 'object', 'agent_distance', 'facingness',
                        'joint_action', 'communication', 'valence', 'arousal']
            title_names = ['spatial expanse', 'object directedness',
                        'agent distance', 'facingness',
                        'joint action', 'communication', 'valence', 'arousal']
            colors = ['#F5DD40', '#F5DD40', '#8558F4', '#8558F4',
                      '#73D2DF', '#73D2DF', '#D57D7F', '#D57D7F']
            out_plot = f'{self.out_dir}/all_features_plot.pdf'

        if self.overwrite or not Path(f'{self.out_dir}/{self.out_csv}').is_file():
            files = glob(f'{self.regression_dir}/*features.parquet')
            df = load_and_summarize(files)
            df.to_csv(f'{self.out_dir}/{self.out_csv}', index=False)
        else:
            df = pd.read_csv(f'{self.out_dir}/{self.out_csv}')

        # Make categorical for plotting
        df = df.loc[df['feature'].isin(features)].reset_index(drop=True)
        df['feature'] = pd.Categorical(df['feature'], categories=features, ordered=True)
        print(df.head())

        if self.simplified_plotting:
            plot_simple(out_plot, df, features, colors, title_names)
        else: 
            plot_full(out_plot, df, features, colors, title_names)


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--simplified_plotting', action=argparse.BooleanOptionalAction, default=False,
                        help='plot all or only select features')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotFeatureDecoding')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='feature_plot.csv')
    parser.add_argument('--regression_dir', '-r', type=str, help='directory for input',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/FeatureRegression')
    args = parser.parse_args()
    PlotFeatureDecoding(args).run()


if __name__ == '__main__':
    main()