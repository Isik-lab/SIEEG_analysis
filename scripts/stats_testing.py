import argparse
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm 
from scipy import ndimage
from pathlib import Path
from matplotlib.lines import Line2D
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from eeg.stats import get_onset_latency, bootstrap_latency_ci, compare_latencies_paired, bootstrap_ci


def load_files(files):
    df = []
    for file in tqdm(files, desc='Loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    print('Finished loading files')
    return df

def load_summary(files, n_samples=1000, seed=42):
    df = load_files(files)
    plotting_stats = []
    onset_stats = []
    for feature, fdf in tqdm(df.groupby('feature'), desc='Calculating summary stats'):
        fdf = fdf.pivot(index='time', columns='eeg_subj_id', values='value').sort_index()
        corr_data = fdf.to_numpy()
        times = fdf.reset_index()['time'].to_numpy()
        stats = get_onset_latency(corr_data, times,
                                  n_permutations=n_samples,
                                  seed=seed)
        low_ci, high_ci = bootstrap_ci(corr_data, seed=seed,
                                       n_bootstrap=n_samples)
        latency_ci = bootstrap_latency_ci(corr_data, times,
                                            n_bootstrap=n_samples,
                                            n_permutations=n_samples,
                                            seed=seed)
        
        onset_stats.append({
            'feature': feature,
            'onset_latency': stats['onset_latency_ms'],
            'onset_low_ci': latency_ci[0],
            'onset_high_ci': latency_ci[1],
        })

        plotting_stats.append(pd.DataFrame({
            'feature': feature,
            'time': times,
            'score': corr_data.mean(axis=1),
            'significant_timepoints': stats['sig_timepoints'],
            'low_ci': low_ci,
            'high_ci': high_ci,
        }))
        print(feature)
        return pd.concat(plotting_stats, ignore_index=True), \
            pd.DataFrame(onset_stats)



def plot_full_timecourse(out_file, stats_df, colors, title_names):
    _, axes = plt.subplots(4, 2, figsize=(7.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()
    ymin, ymax = -0.15, 0.425

    order_counter = 0
    custom_lines = []
    smooth_kernel = np.ones(10)/10
    for ifeature, (_, feature_df) in enumerate(stats_df.groupby('feature', observed=True)):
        ax, color, feature = axes[ifeature], colors[ifeature], title_names[ifeature]
        order_counter +=1
        alpha = 0.1 if color == 'black' else 0.2
        alpha += 0.2 if color == '#F5DD40' else 0
        smoothed_data = {}
        for key in ['low_ci', 'high_ci', 'score']:
            smoothed_data[key] = np.convolve(feature_df[key], smooth_kernel, mode='same')

        ax.fill_between(x=feature_df['time'], 
                    y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                    edgecolor=None, color=color, alpha=alpha, 
                    zorder=order_counter)
        order_counter +=1
        ax.plot(feature_df['time'], smoothed_data['score'],
                color=color, zorder=order_counter,
                linewidth=1.5)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

        label, n = ndimage.label(feature_df['significant_timepoints'])
        onset = None
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if onset is None:
                onset = time_cluster.min()
                ax.text(x=time_cluster.min(), 
                        y=ymin+((ymax-ymin)*0.065),
                        s=f'{onset:.0f} ms',
                        fontsize=6)
            ax.hlines(y=ymin+((ymax-ymin)*0.05),
                      xmin=time_cluster.min(),
                      xmax=time_cluster.max(),
                      color=color, zorder=0, linewidth=1.5)

        ax.set_title(feature)
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                    linestyles='dashed', colors='grey',
                    linewidth=1, zorder=0)
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                    linewidth=1, zorder=0)
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
        self.final_plot = args.final_plot
        print(vars(self))

    def run(self):
        if self.simplified_plotting:
            features = ['agent_distance', 'communication']
            title_names = ['agent distance', 'communication']
            colors = ['#c83e73', '#59157e']
            out_plot = 'feature_plot.pdf'
        else:
            features = ['expanse', 'object', 'agent_distance', 'facingness',
                        'joint_action', 'communication', 'valence', 'arousal']
            title_names = ['spatial expanse', 'object directedness',
                        'agent distance', 'facingness',
                        'joint action', 'communication', 'valence', 'arousal']
            colors = ['#fa7d5e', '#e95462', '#c83e73',
                      '#a3307e', '#7e2482', '#59157e', '#331067', '#120d31']
            out_plot = 'supplement_features'

        if self.overwrite or not Path(f'{self.out_dir}/{self.out_csv}').is_file():
            files = glob(f'{self.regression_dir}/*features.parquet')
            df_time, df_onset = load_summary(files)
            df_time.to_csv(f'{self.out_dir}/{self.out_csv}', index=False)
            df_onset.to_csv(f'{self.out_dir}/onset_{self.out_csv}', index=False)
        else:
            df_time = pd.read_csv(f'{self.out_dir}/{self.out_csv}')
            df_onset = pd.read_csv(f'{self.out_dir}/onset_{self.out_csv}')

        # Make categorical for plotting
        df_time['feature'] = pd.Categorical(df_time['feature'], categories=features, ordered=True)
        df_onset['feature'] = pd.Categorical(df_onset['feature'], categories=features, ordered=True)
        plot_full_timecourse(f'{self.out_dir}/{out_plot}_timecourse.pdf',
                                 df_time, colors, title_names)


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--simplified_plotting', action=argparse.BooleanOptionalAction, default=False,
                        help='plot all or only select features')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True,
                        help='whether to redo the summary statistics')
    parser.add_argument('--final_plot', '-p', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/reports/figures/FinalFigures')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/PlotFeatureDecoding')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='feature_decoding_timecourse.csv')
    parser.add_argument('--regression_dir', '-r', type=str, help='directory for input',
                        default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis/data/interim/FeatureRegression')
    args = parser.parse_args()
    PlotFeatureDecoding(args).run()


if __name__ == '__main__':
    main()