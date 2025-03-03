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
from matplotlib.lines import Line2D
import shutil
from src.temporal import bin_time_windows_cut


def load_latency(files):
    #Load data
    df = []
    for file in tqdm(files, desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    print('Finished loading files')

    # Add categories for different time windows
    df['time_window'] = bin_time_windows_cut(df, window_size=50, end_time=500)
    # Remove time windows before stimulus and after 300 ms
    df = df.loc[(df.time_window >= 0) & (df.time_window < 350)].reset_index()
    df['time_window'] = df.time_window.astype('int32')

    #Average across EEG subjects
    mean_df = df.groupby(['time_window', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')

    #Average across EEG subjects
    mean_df = df.groupby(['time_window', 'roi_name']).mean(numeric_only=True).reset_index()
    mean_df = mean_df.dropna().drop(columns=['time']).rename(columns={'value': 'score'})
    mean_df.reset_index(drop=True, inplace=True)
    print('Finished mean over EEG subjects')

    ### Group stats###
    # Variance
    var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
    scores_var = mean_df[var_cols].to_numpy()
    mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
    mean_df.drop(columns=var_cols, inplace=True)

    # P-values
    null_cols = [col for col in mean_df.columns if 'null_perm_' in col]
    stats_df = []
    for _, roi_df in mean_df.groupby('roi_name'):
        scores_null = roi_df[null_cols].to_numpy().T
        scores = roi_df['score'].to_numpy().T
        roi_df['p'] = calculate_p(scores_null, scores, 5000, 'greater')
        roi_df.drop(columns=null_cols, inplace=True)
        stats_df.append(roi_df)
    stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True)
    return stats_df.rename(columns={'value': 'score'})


def load_timecourse(files):
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
    mean_df = mean_df.rename(columns={'value': 'score'})
    print('Finished mean over fMRI subjects')
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
        scores = roi_df['score'].to_numpy().T
        ps = calculate_p(scores_null, scores, 5000, 'greater')
        roi_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T,
                                            verbose=True,
                                            desc=f'{roi_name} cluster correction')
        roi_df.drop(columns=null_cols, inplace=True)
        stats_df.append(roi_df)
    stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True) 
    return stats_df.rename(columns={'value': 'score'})


# Plot the results
def plot_simple_timecourse(ax, stats_df, colors, title_names): 
    order_counter = 0
    stats_pos = -.12
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for (_, roi_df), color in zip(stats_df.groupby('roi_name', observed=True), colors):
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
                linewidth=1.5)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

        label, n = ndimage.label(roi_df['p'] < 0.05)
        onset = None
        for icluster in range(1, n+1):
            time_cluster = roi_df['time'].to_numpy()[label == icluster]
            if onset is None:
                onset = time_cluster.min()
                ax.text(x=time_cluster.min(), 
                        y=stats_pos+0.008,
                        s=f'{onset:.0f} ms',
                        fontsize=6)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                    xmax=time_cluster.max(),
                    color=color, zorder=0, linewidth=1.5)
        stats_pos -= 0.05

    ax.legend(custom_lines, title_names,
              loc='upper right', fontsize=8,
              handlelength=1,  # Length of legend lines
              handleheight=1)  # Height of legend lines (for markers)
    ymin, ymax = ax.get_ylim()
    ax.set_xlim([-200, 1000])
    ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
              linestyles='dashed', colors='grey',
              linewidth=1, zorder=0)
    ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
              linewidth=1, zorder=0)
    ax.set_ylabel('Prediction ($r$)')
    ax.set_xlabel('Time (ms)')
    ax.tick_params(axis='x', labelsize=8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([ymin, ymax])


def plot_simple_latency(ax, stats_df, colors, title_names, jitter_size=12,
                        xmin=0, xmax=300, x_padding=10):
    order_counter = -1
    jitter = -1 * jitter_size
    for (_, roi_df), (label, color) in zip(stats_df.groupby('roi_name', observed=True), zip(title_names, colors)):
        order_counter +=1
        ax.vlines(x=roi_df['time_window']+jitter, 
                    ymin=roi_df['low_ci'], ymax=roi_df['high_ci'],
                    color=color,
                    zorder=order_counter)
        order_counter +=1
        ax.scatter(roi_df['time_window']+jitter, roi_df['score'],
                   color=color, zorder=order_counter, s=15)
        
        sigs = roi_df['high_ci'][roi_df['p'] < 0.05] + 0.02
        sigs_time = roi_df['time_window'][roi_df['p'] < 0.05] + (jitter-2.5)
        for sig, sig_time in zip(sigs, sigs_time):
            ax.text(sig_time, sig, '*', fontsize='x-small')
        jitter += jitter_size

    ymin, ymax = ax.get_ylim()
    ax.set_xlim([xmin-jitter_size-x_padding, xmax+jitter_size+x_padding])
    ax.set_xlabel('Time bin (ms)')
    ax.set_xticks(list(np.arange(xmin, xmax+1, step=50)))
    ax.tick_params(axis='x', labelsize=8)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.hlines(y=0, xmin=xmin-jitter_size-x_padding, xmax=xmax+jitter_size+x_padding,
              color='grey', zorder=0, linewidth=1)
    ax.hlines(y=ymin, xmin=xmin-jitter_size-x_padding, xmax=xmax+jitter_size+x_padding,
              color='black', zorder=0, linewidth=2)
    ax.set_ylim([ymin, ymax])


# Plot the results
def plot_simple(out_file, df_time, df_latency, colors, title_names): 
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3),
                             width_ratios=[7, 3],
                             sharey=True)
    plot_simple_timecourse(axes[0], df_time, colors, title_names)
    plot_simple_latency(axes[1], df_latency, colors, title_names)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    fig.text(0.01, .95, 'A', ha='center', fontsize=12)
    fig.text(0.7, .95, 'B', ha='center', fontsize=12)
    plt.savefig(out_file)

def plot_full_timecourse(out_file, stats_df, colors, title_names):
    _, axes = plt.subplots(4, 2, figsize=(7.5, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    ymin, ymax = -0.15, 0.6

    order_counter = 0
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for iroi, (_, roi_df) in enumerate(stats_df.groupby('roi_name', observed=True)):
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
                linewidth=1.5)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

        label, n = ndimage.label(roi_df['p'] < 0.05)
        onset = None
        for icluster in range(1, n+1):
            time_cluster = roi_df['time'].to_numpy()[label == icluster]
            if onset is None:
                onset = time_cluster.min()
                ax.text(x=time_cluster.min(), 
                        y=ymin+((ymax-ymin)*0.07),
                        s=f'{onset:.0f} ms',
                        fontsize=6)
            ax.hlines(y=ymin+((ymax-ymin)*0.05),
                      xmin=time_cluster.min(),
                      xmax=time_cluster.max(),
                      color=color, zorder=0, linewidth=1.5)

        ax.set_title(roi)
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                    linestyles='dashed', colors='grey',
                    linewidth=1, zorder=0)
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                    linewidth=1, zorder=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([ymin, ymax])
        if iroi % 2 == 0:
            ax.set_ylabel('Prediction ($r$)')

        if iroi >= 6:
            ax.set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig(out_file)

def plot_full_latency(out_file, stats_df, colors, title_names):
    _, ax = plt.subplots(figsize=(7.5, 3))
    plt.subplots_adjust(right=0.85)
    order_counter = -1
    jitter = -17
    xmin, xmax = -25, 325
    for (_, roi_df), (label, color) in zip(stats_df.groupby('roi_name', observed=True), zip(title_names, colors)):
        order_counter +=1
        ax.vlines(x=roi_df['time_window']+jitter, 
                    ymin=roi_df['low_ci'], ymax=roi_df['high_ci'],
                    color=color,
                    zorder=order_counter)
        order_counter +=1
        ax.scatter(roi_df['time_window']+jitter, roi_df['score'], s=20,
                    color=color, zorder=order_counter, label=label)
        
        sigs = roi_df['high_ci'][roi_df['p'] < 0.05] + 0.02
        sigs_time = roi_df['time_window'][roi_df['p'] < 0.05] + (jitter-1.75)
        for sig, sig_time in zip(sigs, sigs_time):
            ax.text(sig_time, sig, '*', fontsize='x-small')
        jitter += 5

    ax.legend(bbox_to_anchor=(1.05, .75), loc='upper left')
    ymin, ymax = ax.get_ylim()
    ax.set_xlim([xmin, xmax])
    ax.set_ylabel('Prediction ($r$)')
    ax.set_xlabel('Time window (ms)')
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xticklabels(['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350'])
    ax.tick_params(axis='x', labelsize=8)
    ax.vlines(x=list(np.arange(25, 325, step=50)), 
              ymin=ymin, ymax=ymax,
              color='gray', linewidth=.7, alpha=0.5)
    ax.tick_params(axis='x', labelsize=10)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.hlines(y=0, xmin=xmin, xmax=xmax,
              color='grey', zorder=0, linewidth=1)
    ax.hlines(y=ymin, xmin=xmin, xmax=xmax,
              color='black', zorder=0, linewidth=2)
    ax.set_ylim([ymin, ymax])

    plt.tight_layout()
    plt.savefig(out_file)


class PlotROIDecoding:
    def __init__(self, args):
        self.out_dir = args.out_dir 
        self.out_csv = args.out_csv
        self.regression_dir = args.regression_dir 
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        self.simplified_plotting = args.simplified_plotting
        self.overwrite = args.overwrite 
        self.final_plot = args.final_plot
        print(vars(self))

    def run(self):
        if self.simplified_plotting:
            rois = ['EVC', 'LOC', 'aSTS']
            title_names = ['EVC', 'LOC', 'aSTS-SI']
            colors = ['#a1dfb9', '#38aaac', '#28192e']
            out_plot = 'roi_plot.pdf'
        else:
            rois = ['EVC', 'MT',
                    'LOC', 'EBA', 
                    'FFA', 'PPA', 
                    'pSTS', 'aSTS']
            title_names = ['EVC', 'MT', 
                           'LOC', 'EBA',
                           'FFA', 'PPA',
                           'pSTS-SI', 'aSTS-SI']
            colors = ['#a1dfb9', '#55caad', '#38aaac', '#348ba6',
                      '#366a9f', '#40498e', '#3b2e5d', '#28192e']
            out_plot = 'supplement_rois'

        if self.overwrite or not Path(f'{self.out_dir}/{self.out_csv}').is_file():
            files = glob(f'{self.regression_dir}/*rois.parquet')
            df_time = load_timecourse(files)
            df_time.to_csv(f'{self.out_dir}/{self.out_csv}', index=False)

            df_latency = load_latency(files)
            df_latency.to_csv(f'{self.out_dir}/roi_decoding_latency.csv', index=False)
        else:
            df_time = pd.read_csv(f'{self.out_dir}/{self.out_csv}')
            df_latency = pd.read_csv(f'{self.out_dir}/roi_decoding_latency.csv')
        
        # Make categorical for plotting
        df_time = df_time.loc[df_time['roi_name'].isin(rois)].reset_index(drop=True)
        df_time['roi_name'] = pd.Categorical(df_time['roi_name'], categories=rois, ordered=True)
        df_latency = df_latency.loc[df_latency['roi_name'].isin(rois)].reset_index(drop=True)
        df_latency['roi_name'] = pd.Categorical(df_latency['roi_name'], categories=rois, ordered=True)

        sns.set_context(context='paper')
        if self.simplified_plotting:
            plot_simple(f'{self.out_dir}/{out_plot}', df_time, df_latency, colors, title_names)
            shutil.copyfile(f'{self.out_dir}/{out_plot}', f'{self.final_plot}/Figure3.pdf')
        else: 
            plot_full_timecourse(f'{self.out_dir}/{out_plot}_timecourse.pdf',
                                  df_time, colors, title_names)
            plot_full_latency(f'{self.out_dir}/{out_plot}_latency.pdf', df_latency,
                              colors, title_names)
            shutil.copyfile(f'{self.out_dir}/{out_plot}_timecourse.pdf',
                            f'{self.final_plot}/{out_plot}_timecourse.pdf')
            shutil.copyfile(f'{self.out_dir}/{out_plot}_latency.pdf',
                            f'{self.final_plot}/{out_plot}_latency.pdf')


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--simplified_plotting', action=argparse.BooleanOptionalAction, default=False,
                        help='plot all or only select ROIs')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--final_plot', '-p', type=str,
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures/FinalFigures')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotROIDecoding')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='roi_decoding_timecourse.csv')
    parser.add_argument('--regression_dir', '-r', type=str, help='directory for input',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/fMRIRegression')
    args = parser.parse_args()
    PlotROIDecoding(args).run()


if __name__ == '__main__':
    main()