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


class PlotBack2Back:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.overwrite = args.overwrite
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
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
        self.reduced_features_legends = ['AlexNet conv2', 'agent distance', 'communication']
        self.reduced_colors = ['#404040', '#8558F4', '#73D2DF']
        
        self.smooth_kernel = np.ones(10)/10
        self.stats_pos_start = {'EVC': -.2, 'LOC': -.12, 'aSTS': -.12}

    def load_and_process_data(self):
        """Load and process data if not already processed"""
        if not os.path.isfile(f'{self.output_dir}plot.csv') or self.overwrite:
            back2back_df = []
            for feature in tqdm(self.features, desc='Feature group summary', leave=True):
                df = self._load_feature_data(feature)
                stats_df = self._calculate_stats(df)
                stats_df['feature'] = feature
                back2back_df.append(stats_df)
                
            back2back_df = pd.concat(back2back_df, ignore_index=True).reset_index(drop=True)
            back2back_df.to_csv(f'{self.output_dir}/plot.csv', index=False)
        else:
            back2back_df = pd.read_csv(f'{self.output_dir}/plot.csv')
        
        return back2back_df

    def _load_feature_data(self, feature):
        """Load data for a specific feature"""
        df = []
        files = glob(f'{self.input_dir}/*{feature}*.parquet')
        print(files)
        for i_file, file in enumerate(files):
            subj_df = pd.read_parquet(file)
            subj_df['eeg_subj_id'] = i_file
            subj_df = subj_df.loc[subj_df.roi_name.isin(self.rois)].reset_index()
            df.append(subj_df)
        return pd.concat(df, ignore_index=True)

    def _calculate_stats(self, df):
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

    def plot_full_results(self, stats_df):
        """Plot full results for all ROIs and features"""
        ymin = -0.2
        for roi, (roi_name, roi_df) in zip(self.roi_titles, stats_df.groupby('roi_name', observed=True)):
            fig, axes = plt.subplots(5, 2, figsize=(19, 15.83), sharex=True, sharey=True)
            axes = axes.flatten()
            ymax = 0.4 if roi not in ['EVC', 'PPA', 'FFA'] else 0.75
            
            self._plot_features(roi_df, axes, ymin, ymax)
            fig.suptitle(roi)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{roi}.pdf')

    def plot_reduced_results(self, stats_df):
        """Plot results for reduced set of ROIs and features"""
        _, axes = plt.subplots(len(self.reduced_rois_titles), 1, figsize=(19, 13.25), sharex=True)
        axes = axes.flatten()
        
        for iroi, (roi_name, cur_df) in enumerate(stats_df.groupby('roi_name', observed=True)):
            self._plot_reduced_roi(iroi, roi_name, cur_df, axes)
            
        self._finalize_reduced_plot(axes)
        plt.savefig(f'{self.output_dir}/feature-roi_plot.pdf')

    def _plot_features(self, roi_df, axes, ymin, ymax):
        """Plot features for a specific ROI"""
        order_counter = 0
        stats_pos = -.12
        
        for ifeature, (feature_name, feature_df) in enumerate(roi_df.groupby('feature', observed=True)):
            feature, color, ax = self.feature_titles[ifeature], self.colors[ifeature], axes[ifeature]
            alpha = self._get_alpha(color)
            
            smoothed_data = self._smooth_data(feature_df)
            self._plot_feature_data(ax, feature_df, smoothed_data, color, alpha, order_counter, stats_pos)
            self._set_axes_properties(ax, feature, ymin, ymax, ifeature)
            
            order_counter += 2
    
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
                linewidth=5)

        # Plot significance markers
        self._plot_significance(ax, feature_df, color, stats_pos)

    def _plot_significance(self, ax, feature_df, color, stats_pos):
        """Plot significance markers and onset times"""
        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 100 if onset_time < 100 else 110
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                       s=f'{onset_time:.0f} ms',
                       fontsize=12)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                     xmax=time_cluster.max(),
                     color=color, zorder=0, linewidth=2)

    def _set_axes_properties(self, ax, feature, ymin, ymax, ifeature):
        """Set the properties for each subplot axes"""
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
            ax.set_ylabel('Unique variance ($r^2$)')
        if ifeature >= 8:
            ax.set_xlabel('Time (ms)')

    def _plot_reduced_roi(self, iroi, roi_name, cur_df, axes):
        """Plot reduced ROI results"""
        ax = axes[iroi]
        title = self.reduced_rois_titles[iroi]
        order_counter = 0
        stats_pos = self.stats_pos_start[roi_name]
        custom_lines = []
        
        for color, (feature, feature_df) in zip(self.reduced_colors, cur_df.groupby('feature', observed=True)):
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
                   linewidth=5)
            custom_lines.append(Line2D([0], [0], color=color, lw=5))
            
            # Plot significance
            self._plot_reduced_significance(ax, feature_df, color, stats_pos)
            stats_pos -= 0.04
            order_counter += 1
            
        self._set_reduced_axes_properties(ax, title)
        return custom_lines

    def _plot_reduced_significance(self, ax, feature_df, color, stats_pos):
        """Plot significance for reduced ROI plots"""
        label, n = ndimage.label(feature_df['p'] < 0.05)
        onset_time = np.nan
        
        for icluster in range(1, n+1):
            time_cluster = feature_df['time'].to_numpy()[label == icluster]
            if icluster == 1:
                onset_time = time_cluster.min()
                shift = 60 if onset_time < 100 else 75
                ax.text(x=onset_time-shift, y=stats_pos-.006,
                       s=f'{onset_time:.0f} ms',
                       fontsize=15.5)
            ax.hlines(y=stats_pos, xmin=time_cluster.min(),
                     xmax=time_cluster.max(),
                     color=color, zorder=0, linewidth=4)

    def _set_reduced_axes_properties(self, ax, title):
        """Set properties for reduced ROI axes"""
        ymin, ymax = ax.get_ylim()
        ax.set_xlim([-200, 1000])
        ax.vlines(x=[0, 500], ymin=ymin, ymax=ymax,
                 linestyles='dashed', colors='grey',
                 linewidth=5, zorder=0)
        ax.hlines(y=0, xmin=-200, xmax=1000, colors='grey',
                 linewidth=5, zorder=0)
        ax.set_ylabel('Unique variance ($r^2$)')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([ymin, ymax])
        ax.set_title(title)

    def _finalize_reduced_plot(self, axes):
        """Add final touches to reduced ROI plot"""
        custom_lines = [Line2D([0], [0], color=c, lw=5) for c in self.reduced_colors]
        axes[0].legend(custom_lines, self.reduced_features_legends,
                      loc='upper right', fontsize='18')
        axes[-1].set_xlabel('Time (ms)')

    def run(self):
        """Main execution method"""
        # Load and process data
        back2back_df = self.load_and_process_data()
        
        # Prepare full stats dataframe
        full_stats_df = self._prepare_full_stats(back2back_df)
        
        # Plot full results
        sns.set_context(context='paper', font_scale=2)
        self.plot_full_results(full_stats_df)
        
        # Prepare and plot reduced results
        reduced_stats_df = self._prepare_reduced_stats(back2back_df)
        sns.set_context(context='poster', font_scale=1.25)
        self.plot_reduced_results(reduced_stats_df)



def main():
    parser = argparse.ArgumentParser(description='Plot the Back2Back regression results')
    parser.add_argument('--input_dir', type=str, help='Input file prefix',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/Back2Back')
    parser.add_argument('--output_dir', type=str, help='Output file prefix', 
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotBack2Back/')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to redo the summary statistics')
    args = parser.parse_args()
    PlotBack2Back(args).run()

if __name__ == '__main__':
    main()
