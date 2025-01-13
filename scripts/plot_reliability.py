import argparse
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image


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
    mean_df = df.groupby(['time']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')
    # Group stats
    # Variance
    var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
    scores_var = mean_df[var_cols].to_numpy()
    mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
    mean_df.drop(columns=var_cols, inplace=True)
    return mean_df


# Plot the results
def plot_reliability(out_file, df, img): 
    sns.set_context(context='paper')
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3),
                             width_ratios=[4, 7],
                             dpi=600)
    # Plot overview image
    axes[0].imshow(img, interpolation='nearest')
    axes[0].axis('off')

    ax = axes[1]
    smoothed_data = {}
    for key in ['low_ci', 'high_ci', 'r']:
        smoothed_data[key] = np.convolve(df[key], np.ones(10)/10, mode='same')

    ax.fill_between(x=df['time'], 
                y1=smoothed_data['low_ci'], y2=smoothed_data['high_ci'],
                edgecolor=None, color='black', alpha=0.1, 
                zorder=0)
    ax.plot(df['time'], smoothed_data['r'],
            color='black', zorder=1,
            linewidth=1.5)

    ymin, ymax = ax.get_ylim()
    ax.set_xlim([-200, 1000])
    ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                linestyles='dashed', colors='grey',
                linewidth=1, zorder=0)
    ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                linewidth=1, zorder=0)
    ax.set_ylabel('Correlation ($r$)')
    ax.set_xlabel('Time (ms)')
    ax.tick_params(axis='x', labelsize=8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([ymin, ymax])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    fig.text(0.01, .95, 'A', ha='center', fontsize=12)
    fig.text(0.31, .95, 'B', ha='center', fontsize=12)
    plt.savefig(out_file)


class PlotReliability:
    def __init__(self, args):
        self.out_dir = args.out_dir 
        self.out_csv = args.out_csv
        self.reliability_dir = args.reliability_dir 
        self.overview = args.overview
        self.out_plot = args.overview.replace('schema.png', 'Figure1.pdf')
        Path(f'{self.out_dir}/PlotReliability').mkdir(exist_ok=True, parents=True)
        self.overwrite = args.overwrite 
        print(vars(self))

    def run(self):
        overview_img = Image.open(self.overview) 
        print(f'image size {overview_img.size}')

        if self.overwrite or not Path(f'{self.out_dir}/PlotReliability/{self.out_csv}').is_file():
            files = glob(f'{self.reliability_dir}/*.parquet')
            df = load_timecourse(files)
            df.to_csv(f'{self.out_dir}/PlotReliability/{self.out_csv}', index=False)
        else:
            df = pd.read_csv(f'{self.out_dir}/PlotReliability/{self.out_csv}')

        plot_reliability(self.out_plot, df, overview_img)


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim')
    parser.add_argument('--out_csv', type=str, help='output csv',
                        default='reliability.csv')
    parser.add_argument('--overview', type=str, help='experiment schematic',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures/FinalFigures/schema.png')
    parser.add_argument('--reliability_dir', '-r', type=str, help='directory for input',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegReliability')
    args = parser.parse_args()
    PlotReliability(args).run()


if __name__ == '__main__':
    main()