import argparse
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm 
from pathlib import Path
from src.stats import calculate_p


def bin_time_windows_cut(df, window_size=50, start_time=0, end_time=300):
    """
    Bin time values into windows using pd.cut with a fixed end time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'time' column
    window_size : int
        Size of each time window (default: 50)
    end_time : int
        Minimum time value to consider (default: 0)
    end_time : int
        Maximum time value to consider (default: 300)
        
    Returns:
    --------
    pandas.Series
        Series containing the binned time windows
    """
    bins = np.arange(start_time, end_time + window_size + 1, window_size)
    labels = bins[:-1]
    return pd.cut(df['time'].clip(upper=end_time), 
                 bins=bins, 
                 labels=labels, 
                 right=False)


def load_and_summarize(files):
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
    df = df.loc[(df.time_window >= 0) & (df.time_window < 250)].reset_index()

    #Average across EEG subjects
    mean_df = df.groupby(['time_window', 'feature']).mean(numeric_only=True).reset_index()
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
    for _, feature_df in mean_df.groupby('feature'):
        scores_null = feature_df[null_cols].to_numpy().T
        scores = feature_df['score'].to_numpy().T
        feature_df['p'] = calculate_p(scores_null, scores, 5000, 'greater')
        feature_df.drop(columns=null_cols, inplace=True)
        stats_df.append(feature_df)
    return pd.concat(stats_df, ignore_index=True).reset_index(drop=True)


def plot_simple(out_file, stats_df, colors, title_names):
    sns.set_context(context='poster', font_scale=1.5)
    _, ax = plt.subplots(figsize=(14, 9.5))
    plt.subplots_adjust(right=0.85)
    order_counter = -1
    jitter = -8 
    xmin, xmax = -15, 215
    for (_, feature_df), (label, color) in zip(stats_df.groupby('feature', observed=True), zip(title_names, colors)):
        order_counter +=1
        ax.vlines(x=feature_df['time_window']+jitter, 
                    ymin=feature_df['low_ci'], ymax=feature_df['high_ci'],
                    color=color,
                    zorder=order_counter)
        order_counter +=1
        ax.plot(feature_df['time_window']+jitter, feature_df['score'], 'o',
                color=color, zorder=order_counter, label=label)
        
        sigs = feature_df['high_ci'][feature_df['p'] < 0.05] + 0.02
        sigs_time = feature_df['time_window'][feature_df['p'] < 0.05] + (jitter-2.5)
        for sig, sig_time in zip(sigs, sigs_time):
            ax.text(sig_time, sig, '*', fontsize='x-small')
        jitter += 8

    ax.legend(bbox_to_anchor=(1.05, .75), loc='upper left')
    ymin, ymax = ax.get_ylim()
    ax.set_xlim([xmin, xmax])
    ax.set_ylabel('Prediction ($r$)')
    ax.set_xlabel('Time (ms)')
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([0, 50, 100, 150, 200])
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.hlines(y=0, xmin=xmin, xmax=xmax,
              color='black', zorder=0, linewidth=3)
    ax.hlines(y=ymin, xmin=xmin, xmax=xmax,
              color='black', zorder=0, linewidth=3)
    ax.set_ylim([ymin, ymax])

    plt.tight_layout()
    plt.savefig(out_file)

class PlotFeatureLatency:
    def __init__(self, args):
        self.out_dir = args.out_dir 
        self.out_csv = args.out_csv
        self.regression_dir = args.regression_dir 
        self.simplified_plotting = args.simplified_plotting
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
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

        plot_simple(out_plot, df, colors, title_names)


def main():
    parser = argparse.ArgumentParser(description='Plot the Feature regression latency')
    parser.add_argument('--simplified_plotting', action=argparse.BooleanOptionalAction, default=False,
                        help='plot all or only select features')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotFeatureLatency')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='feature_plot.csv')
    parser.add_argument('--regression_dir', '-r', type=str, help='directory for input',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/FeatureRegression')
    args = parser.parse_args()
    PlotFeatureLatency(args).run()


if __name__ == '__main__':
    main()