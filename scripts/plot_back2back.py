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
from src.temporal import bin_time_windows_cut
import shutil


class PlotBack2Back:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.overwrite = args.overwrite
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.final_plot = args.final_plot   
        
        # Constants
        self.rois = ['EVC', 'MT', 'FFA', 'PPA', 'LOC', 'EBA', 'pSTS', 'aSTS']
        self.features = ['alexnet', 'moten', 'expanse', 'object',
                        'agent_distance', 'facingness',
                        'joint_action', 'communication', 
                        'valence', 'arousal']
        self.roi_titles = ['EVC', 'MT', 'FFA', 'PPA', 'LOC', 'EBA', 'pSTS-SI', 'aSTS-SI']
        self.feature_titles = ['AlexNet-conv2', 'motion energy', 
                             'spatial expanse', 'object directedness',
                             'agent distance', 'facingness',
                             'joint action', 'communication', 'valence', 'arousal']
        self.colors = ['#404040', '#404040', '#F5DD40', '#F5DD40', '#8558F4', '#8558F4', 
                      '#73D2DF', '#73D2DF', '#D57D7F', '#D57D7F']
        
        # Reduced set constants
        self.reduced_rois = ['EVC', 'LOC', 'aSTS']
        self.reduced_features = ['alexnet', 'agent_distance', 'communication']
        self.reduced_rois_titles = ['EVC', 'LOC', 'aSTS-SI']
        self.reduced_features_legends = ['AlexNet conv2','agent distance', 'communication']
        self.reduced_colors = ['#404040','#8558F4', '#73D2DF']
        
        self.smooth_kernel = np.ones(10)/10
        self.stats_pos_start = {'EVC': -.2, 'LOC': -.08, 'aSTS': -.08}

    def load_and_process_data(self):
        """Load and process data if not already processed"""
        if not os.path.isfile(f'{self.output_dir}/joint_decoding_timecourse.csv') or self.overwrite:
            df_timecourse = []
            df_latency = []
            for feature in tqdm(self.features, desc='Feature group summary', leave=True):
                df = self._load_feature_data(feature)

                # Time course
                df_timecourse_feature = self._calculate_timecourse(df)
                df_timecourse_feature['feature'] = feature
                df_timecourse.append(df_timecourse_feature)

                # Latency
                df_latency_feature = self._calculate_latency(df)
                df_latency_feature['feature'] = feature
                df_latency.append(df_latency_feature)
                
            # Time course
            df_timecourse = pd.concat(df_timecourse, ignore_index=True).reset_index(drop=True)
            df_timecourse.to_csv(f'{self.output_dir}/joint_decoding_timecourse.csv', index=False)

            # Latency
            df_latency = pd.concat(df_latency, ignore_index=True).reset_index(drop=True)
            df_latency.to_csv(f'{self.output_dir}/joint_decoding_latency.csv', index=False)
        else:
            df_timecourse = pd.read_csv(f'{self.output_dir}/joint_decoding_timecourse.csv')
            df_latency = pd.read_csv(f'{self.output_dir}/joint_decoding_latency.csv')
        
        return df_timecourse, df_latency

    def _load_feature_data(self, feature):
        """Load data for a specific feature"""
        df = []
        files = glob(f'{self.input_dir}/*{feature}*.parquet')
        for i_file, file in enumerate(files):
            subj_df = pd.read_parquet(file)
            subj_df['eeg_subj_id'] = i_file
            subj_df = subj_df.loc[subj_df.roi_name.isin(self.rois)].reset_index()
            df.append(subj_df)
        return pd.concat(df, ignore_index=True)
    
    def _calculate_latency(self, df):
        df_lat = df.copy()
        df_lat['time_window'] = bin_time_windows_cut(df_lat, window_size=50, end_time=500)
        # Remove time windows before stimulus and after 300 ms
        df_lat = df_lat.loc[(df_lat.time_window >= 0) & (df_lat.time_window < 200)].reset_index()
        df_lat['time_window'] = df_lat.time_window.astype('int32')

        #Average across EEG subjects
        df_lat = df_lat.groupby(['time_window', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
        df_lat = df_lat.groupby(['time_window', 'roi_name']).mean(numeric_only=True).reset_index()
        df_lat = df_lat.dropna().drop(columns=['time']).rename(columns={'value': 'score'})
        df_lat.reset_index(drop=True, inplace=True)

        ### Group stats###
        # Variance
        var_cols = [col for col in df_lat.columns if 'var_perm_' in col]
        scores_var = df_lat[var_cols].to_numpy()
        df_lat['low_ci'], df_lat['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
        df_lat.drop(columns=var_cols, inplace=True)

        # P-values
        null_cols = [col for col in df_lat.columns if 'null_perm_' in col]
        stats_df = []
        for _, roi_df in df_lat.groupby('roi_name'):
            scores_null = roi_df[null_cols].to_numpy().T
            scores = roi_df['score'].to_numpy().T
            roi_df['p'] = calculate_p(scores_null, scores, 5000, 'greater')
            roi_df.drop(columns=null_cols, inplace=True)
            stats_df.append(roi_df)
        return pd.concat(stats_df, ignore_index=True).reset_index(drop=True)

    def _calculate_timecourse(self, df):
        """Calculate statistics for the dataset"""
        # Average across EEG and fMRI subjects
        mean_df = df.groupby(['time', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
        mean_df = df.groupby(['time', 'roi_name']).mean(numeric_only=True).reset_index()
        mean_df.sort_values(by=['roi_name', 'time'], inplace=True)
        
        # Calculate variance
        var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
        scores_var = mean_df[var_cols].to_numpy()
        mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
        mean_df.drop(columns=var_cols, inplace=True)
        
        # Calculate p-values
        null_cols = [col for col in mean_df.columns if 'null_perm_' in col]
        stats_df = []
        for roi_name, roi_df in mean_df.groupby('roi_name'):
            stats_df.append(self._process_roi_stats(roi_df, null_cols, roi_name))
            
        return pd.concat(stats_df, ignore_index=True).reset_index(drop=True)

    def _process_roi_stats(self, roi_df, null_cols, roi_name):
        """Process statistics for a specific ROI"""
        scores_null = roi_df[null_cols].to_numpy().T
        scores = roi_df['score'].to_numpy().T
        ps = calculate_p(scores_null, scores, 5000, 'greater')
        roi_df['p'] = cluster_correction(scores.T, ps.T, scores_null.T)
        roi_df.drop(columns=null_cols, inplace=True)
        return roi_df
    
    def _prepare_full_stats(self, df):
        """Prepare full statistics dataframe"""
        stats_df = df.loc[df['feature'].isin(self.features)].reset_index(drop=True)
        stats_df['feature'] = pd.Categorical(stats_df['feature'], categories=self.features, ordered=True)
        stats_df = stats_df.loc[stats_df['roi_name'].isin(self.rois)].reset_index(drop=True)
        stats_df['roi_name'] = pd.Categorical(stats_df['roi_name'], categories=self.rois, ordered=True)
        return stats_df

    def _prepare_reduced_stats(self, df):
        """Prepare reduced statistics dataframe"""
        stats_df = df.loc[df['roi_name'].isin(self.reduced_rois)].reset_index(drop=True)
        stats_df = stats_df.loc[stats_df['feature'].isin(self.reduced_features)].reset_index(drop=True)
        stats_df['roi_name'] = pd.Categorical(stats_df['roi_name'], categories=self.reduced_rois, ordered=True)
        stats_df['feature'] = pd.Categorical(stats_df['feature'], categories=self.reduced_features, ordered=True)
        return stats_df

    def _get_alpha(self, color):
        """Get alpha value based on color"""
        alpha = 0.1 if color == '#404040' else 0.2
        alpha += 0.2 if color == '#F5DD40' else 0
        return alpha

    def _smooth_data(self, df):
        """Smooth data using the kernel"""
        smoothed_data = {}
        for key in ['score', 'high_ci', 'low_ci']:
            smoothed_data[key] = np.convolve(df[key], self.smooth_kernel, mode='same')
        return smoothed_data

    ######### PLOT THE SUPPLEMENTAL LATENCY ##################
    def plot_full_latency(self, stats_df):
        """Plot full results for all ROIs and features"""
        fig, axes = plt.subplots(4, 2, figsize=(7.5, 8), 
                                 sharex=True)
        axes = axes.flatten()
        xmin, xmax = -22, 172
        for (iroi, roi), (_, roi_df) in zip(enumerate(self.roi_titles), stats_df.groupby('roi_name', observed=True)):
            ax = axes[iroi]
            
            self._plot_feature_latency(roi_df, axes[iroi])
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([ymin, ymax])
            ax.vlines(x=[25, 75, 125], 
                      ymin=ymin, ymax=ymax,
                      color='gray', linewidth=.7, alpha=0.5)
            # ax.legend(bbox_to_anchor=(1.05, .75), loc='upper left')
            ax.set_xlim([xmin, xmax])
            ax.set_xticks([0, 50, 100, 150])
            ax.spines[['right', 'top']].set_visible(False)
            ax.hlines(y=0, xmin=xmin, xmax=xmax,
                    color='grey', zorder=0, linewidth=1)
            ax.set_ylim([ymin, ymax])
            if iroi % 2 == 0: 
                ax.set_ylabel('Prediction ($r$)')

            if iroi == 0:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",         # Place the legend above the figure
                    bbox_to_anchor=(0.5, 0.99),  # Adjust the anchor to center above the figure
                    ncol=5,                      # Number of columns in the legend
                    fontsize=8                  # Font size
                )
            
            if (iroi == len(self.roi_titles)-2) or (iroi == len(self.roi_titles)-1):
                ax.set_xticklabels(['0-50', '50-100', '100-150', '150-200'])
                ax.tick_params(axis='x', labelsize=7)
                ax.set_xlabel('Time window (ms)')
            
            ax.set_title(roi)

        plt.subplots_adjust(top=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])      
        plt.savefig(f'{self.output_dir}/supplemental_joint_latency.pdf')

    def _plot_feature_latency(self, roi_df, ax):
        """Plot features for a specific ROI"""
        jitter = -18
        for ifeature, (_, feature_df) in enumerate(roi_df.groupby('feature', observed=True)):
            feature = self.feature_titles[ifeature]
            color = sns.color_palette('colorblind', len(self.feature_titles))[ifeature]

            ax.vlines(x=feature_df['time_window']+jitter, 
                      ymin=feature_df['low_ci'], ymax=feature_df['high_ci'],
                      color=color)
            ax.scatter(feature_df['time_window']+jitter, feature_df['score'],
                       s=10, color=color, label=feature)
            
            sigs = feature_df['high_ci'][feature_df['p'] < 0.05] + 0.02
            sigs_time = feature_df['time_window'][feature_df['p'] < 0.05] + (jitter-1.75)
            for sig, sig_time in zip(sigs, sigs_time):
                ax.text(sig_time, sig, '*', fontsize='x-small')
            jitter += 4
            
    ######### PLOT THE SUPPLEMENTAL TIMECOURSE ##################
    def plot_full_timecourse(self, stats_df):
        """Plot full results for all ROIs and features"""
        ymin = -0.2
        for roi, (_, roi_df) in zip(self.roi_titles, stats_df.groupby('roi_name', observed=True)):
            fig, axes = plt.subplots(5, 2, figsize=(7.5, 9), sharex=True, sharey=True)
            axes = axes.flatten()
            ymax = 0.4 if roi not in ['EVC', 'PPA', 'FFA'] else 0.75
            
            self._plot_features(roi_df, axes, ymin, ymax)
            fig.suptitle(roi)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/supplemental_{roi}_timecourse.pdf')

    def _plot_features(self, roi_df, axes, ymin, ymax):
        """Plot features for a specific ROI"""
        order_counter = 0
        stats_pos = -.12
        
        for ifeature, (_, feature_df) in enumerate(roi_df.groupby('feature', observed=True)):
            feature, color, ax = self.feature_titles[ifeature], self.colors[ifeature], axes[ifeature]
            alpha = self._get_alpha(color)
            
            smoothed_data = self._smooth_data(feature_df)
            self._plot_feature_data(ax, feature_df, smoothed_data, color, alpha, order_counter, stats_pos)
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
            if ifeature >= 8:
                ax.set_xlabel('Time (ms)')
            
            order_counter += 2

    def _plot_feature_data(self, ax, feature_df, smoothed_data, color, alpha, order_counter, stats_pos):
        """Plot feature data on the given axes"""
        # Plot confidence interval
        ax.fill_between(x=feature_df['time'], 
                       y1=smoothed_data['low_ci'], 
                       y2=smoothed_data['high_ci'],
                       edgecolor=None, color=color, alpha=alpha, 
                       zorder=order_counter)
        
        # Plot mean line
        ax.plot(feature_df['time'], smoothed_data['score'],
                color=color, zorder=order_counter + 1,
                linewidth=1.5)

        # Plot significance markers
        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                     xmax=time_cluster.max(),
                     color=color, zorder=0, linewidth=1.5)

    ###################
    # PLOT THE MAIN FIGURE
    def plot_reduced_results(self, df_timecourse_reduced, df_latency_reduced):
        """Plot results for reduced set of ROIs and features"""
        n_rois = len(self.reduced_rois_titles)
        fig, axes = plt.subplots(n_rois, 2,
                                 figsize=(7.5, 5), 
                                 width_ratios=[2, 1])
        
        iterator = enumerate(zip(df_timecourse_reduced.groupby('roi_name', observed=True),
                                 df_latency_reduced.groupby('roi_name', observed=True)))
        for iroi, ((roi_name, roi_timecourse), (_, roi_latency)) in iterator:
            title = self.reduced_rois_titles[iroi]
            lines, (ymin, ymax) = self._plot_reduced_timecourse(axes[iroi, 0], 
                                                          title, roi_name,
                                                          roi_timecourse)
            ymin_round, ymax_round = np.floor(ymin*10)/10, np.ceil(ymax*10)/10
            axes[iroi, 0].vlines(x=[0, 500], ymin=ymin_round, ymax=ymax_round,
                    linestyles='dashed', colors='grey',
                    linewidth=1, zorder=0)
            axes[iroi, 0].set_ylim([ymin_round, ymax_round])
            axes[iroi, 0].set_ylabel('Prediction ($r$)')
            if iroi == (n_rois - 1):
                axes[iroi, 0].tick_params(axis='x', labelsize=8)
                axes[iroi, 0].set_xlabel('Time (ms)')
            else:
                axes[iroi, 0].set_xticklabels([])

            if roi_name == 'EVC':
                yticks = list(np.arange(ymin_round, ymax_round, 0.1)[::3])
            elif 'STS' in roi_name:
                yticks = list(np.arange(ymin_round, ymax_round, 0.1)[::2])
            else:
                yticks = list(np.arange(ymin_round, ymax_round, 0.1))
            axes[iroi, 0].set_yticks(yticks)

            # plot latency
            self._plot_reduced_latency(axes[iroi, 1], roi_latency)
            axes[iroi, 1].set_ylim([ymin_round, ymax_round])
            axes[iroi, 1].set_yticks(yticks)
            axes[iroi, 1].set_yticklabels([])
            if iroi == (n_rois - 1):
                axes[iroi, 1].set_xlim([-15, 165])
                axes[iroi, 1].set_xticks([0, 50, 100, 150])
                axes[iroi, 1].set_xticklabels(['0-50', '50-100', '100-150', '150-200'])
                axes[iroi, 1].set_xlabel('Time window(ms)')
                axes[iroi, 1].tick_params(axis='x', labelsize=8)
            else:
                axes[iroi, 1].set_xticks([0, 50, 100, 150])
                axes[iroi, 1].set_xticklabels([])
        

        fig.legend(lines, self.reduced_features_legends, 
                    loc="lower center",         # Position the legend at the bottom-center
                    bbox_to_anchor=(0.5, .0),
                    ncol=3,                      # Number of columns in the legend
                    fontsize=8)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.01, .95, 'A', ha='center', fontsize=12)
        fig.text(0.675, .95, 'B', ha='center', fontsize=12)
        fig.text(0.01, .66, 'C', ha='center', fontsize=12)
        fig.text(0.675, .66, 'D', ha='center', fontsize=12)
        fig.text(0.01, .37, 'E', ha='center', fontsize=12)
        fig.text(0.675, .37, 'F', ha='center', fontsize=12)
        plt.savefig(f'{self.output_dir}/joint_timecourse.pdf')
    ###################

    def _plot_reduced_timecourse(self, ax, title, roi_name, cur_df):
        """Plot reduced ROI results"""
        order_counter = 0
        stats_pos = self.stats_pos_start[roi_name]
        custom_lines = []
        iterator = zip(self.reduced_colors, cur_df.groupby('feature', observed=True))
        for color, (_, feature_df) in iterator:
            smoothed_data = self._smooth_data(feature_df)
            alpha = self._get_alpha(color)
            
            # Plot data
            ax.fill_between(x=feature_df['time'], 
                          y1=smoothed_data['low_ci'],
                          y2=smoothed_data['high_ci'],
                          edgecolor=None, color=color, alpha=alpha, 
                          zorder=order_counter)
            order_counter += 1
            
            ax.plot(feature_df['time'], smoothed_data['score'],
                   color=color, zorder=order_counter,
                   linewidth=1.5)
            custom_lines.append(Line2D([0], [0], color=color, lw=2))
            
            # Plot significance
            label, n = ndimage.label(feature_df['p'] < 0.05)        
            for icluster in range(1, n+1):
                time_cluster = feature_df['time'].to_numpy()[label == icluster]
                ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                        xmax=time_cluster.max(),
                        color=color, zorder=0, linewidth=1.5)
            stats_pos -= 0.04
            order_counter += 1
            
        ymin, ymax = ax.get_ylim()
        ax.set_xlim([-200, 1000])
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                    linewidth=1, zorder=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([ymin, ymax])
        ax.set_title(title)
        return custom_lines, (ymin, ymax)

    def _plot_reduced_latency(self, ax, cur_df):
        order_counter = -1
        jitter = -8 
        xmin, xmax = -15, 165
        iterator = zip(self.reduced_colors, cur_df.groupby('feature', observed=True))
        for color, (_, feature_df) in iterator:
            order_counter +=1
            ax.vlines(x=feature_df['time_window']+jitter, 
                        ymin=feature_df['low_ci'], ymax=feature_df['high_ci'],
                        color=color,
                        zorder=order_counter)
            order_counter +=1
            ax.scatter(feature_df['time_window']+jitter, feature_df['score'],
                    color=color, zorder=order_counter, s=15)
            
            sigs = feature_df['high_ci'][feature_df['p'] < 0.05] + 0.02
            sigs_time = feature_df['time_window'][feature_df['p'] < 0.05] + (jitter-2.5)
            for sig, sig_time in zip(sigs, sigs_time):
                ax.text(sig_time, sig, '*', fontsize='x-small')
            jitter += 8

        ax.spines[['right', 'top']].set_visible(False)
        ax.hlines(y=0, xmin=xmin, xmax=xmax,
                color='black', zorder=0, linewidth=1)

    def run(self):
        """Main execution method"""
        # Load and process data
        df_timecourse, df_latency = self.load_and_process_data()
        
        # Prepare full stats dataframe
        df_timecourse_full = self._prepare_full_stats(df_timecourse)
        df_latency_full = self._prepare_full_stats(df_latency)

        # Plot full results
        sns.set_context(context='paper', font_scale=1)
        self.plot_full_timecourse(df_timecourse_full)
        self.plot_full_latency(df_latency_full)
        
        # Prepare and plot reduced results
        df_timecourse_reduced = self._prepare_reduced_stats(df_timecourse)
        df_latency_reduced = self._prepare_reduced_stats(df_latency)
        self.plot_reduced_results(df_timecourse_reduced, df_latency_reduced)

        shutil.copyfile(f'{self.output_dir}/joint_timecourse.pdf',
                        f'{self.final_plot}/Figure4.pdf')
        shutil.copyfile(f'{self.output_dir}/supplemental_joint_latency.pdf',
                        f'{self.final_plot}/supplemental_joint_latency.pdf')


def main():
    parser = argparse.ArgumentParser(description='Plot the Back2Back regression results')
    parser.add_argument('--input_dir', type=str, help='Input file prefix',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/Back2Back')
    parser.add_argument('--output_dir', type=str, help='Output file prefix', 
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotBack2Back')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    parser.add_argument('--final_plot', '-p', type=str,
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures/FinalFigures')
    args = parser.parse_args()
    PlotBack2Back(args).run()

if __name__ == '__main__':
    main()
