#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from src import loading
import matplotlib.pyplot as plt
import seaborn as sns


class PlotExampleEEG:
    def __init__(self, args):
        self.process = 'PlotExampleEEG'
        self.data_split = args.data_split
        if 'u' not in args.sid:
            self.sid = f'sub-{str(int(args.sid)).zfill(2)}'
        else:
            self.sid = args.sid
        print(vars(self))
        self.eeg_file = args.eeg_file
        self.out_dir = args.out_dir
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def load_and_average(self):
        eeg = loading.load_eeg(self.eeg_file)
        eeg_avg = eeg.groupby(['time', 'channel', 'video_name']).mean(numeric_only=True).reset_index()
        return eeg_avg
    
    def plot_eeg(self, df):
        sns.set_context(context='poster')
        _, ax = plt.subplots(figsize=(6,6))

        #Plot elipses to indicate more channels
        ax.scatter([250,250,250], [0, 0.5, 1], color='black', s=20)
        offset = 1.5
        
        # Filter to posterior channels
        df = df.loc[df.channel.str.startswith('P')].reset_index(drop=True)

        # Plot 10 posterior channels for one video
        for _, vid_df in df.groupby('video_name'):
            for _, cdf in vid_df.groupby('channel'):
                time, signal = cdf.time.to_numpy(), cdf.signal.to_numpy()
                signal_min, signal_max = signal.min(), signal.max()
                signal_centered = signal/(signal_max - signal_min)
                ax.plot(time, signal_centered+offset, 'k')
                offset += 1
                if offset > 10:
                    break
            break

        # Make the plot prettier
        ax.set_ylim([-.5, offset-0.5])
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=-0.5, ymax=offset-0.5,
                    linestyles='dashed', colors='grey',
                    linewidth=1, zorder=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Channel-wise activity')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/{self.sid}-channelwise_activity.pdf')

    def run(self):
        df = self.load_and_average()
        print(df.head())
        self.plot_eeg(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='1')
    parser.add_argument('--data_split', '-d', type=str, default='test')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotExampleEEG')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegPreprocessing/all_trials/sub-01.parquet')
    args = parser.parse_args()
    PlotExampleEEG(args).run()


if __name__ == '__main__':
    main()