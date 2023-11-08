import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def plot_splithalf_reliability(results, out_file):
    ymin, ymax = results.reliability.min(), results.reliability.max()

    _, axes = plt.subplots(2, sharex=True)
    for i, ax in enumerate(axes):
        if i == 0:
            sns.lineplot(x='time', y='reliability', hue='channel',
                        data=results, ax=ax)
            ax.tick_params(axis='x', which='both', length=0)
        else:
            sns.lineplot(x='time', y='reliability',
                    data=results, ax=ax)
            ax.set_xlabel('Time (s)')
        ax.set_ylabel('Reliability (r)')
        ax.hlines(y=0, xmin=results.time.min(), xmax=results.time.max(),
            colors='gray', linestyles='solid', zorder=0)
        ax.vlines(x=[0, 0.5], ymin=0, ymax=ymax,
            colors='gray', linestyles='dashed', zorder=0)
        ax.set_xlim([results.time.min(), results.time.max()])
        ax.set_ylim([ymin, ymax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.legend([])
    plt.tight_layout()
    plt.savefig(out_file)


def plot_eeg_feature_rsa(rsa, feature_order, out_file):
    feature_group = rsa.groupby('feature')
    _, axes = plt.subplots(int(np.ceil(len(feature_order)/3)), 3,
                        figsize=(10,8),
                        sharey=True, constrained_layout=True)
    axes = axes.flatten()
    trim_axs(axes, len(feature_order))
    ymin, ymax = rsa['Spearman rho'].min(), rsa['Spearman rho'].max()
    for ax, (feature, time_corr) in zip(axes, feature_group):
        sns.lineplot(x='time', y='Spearman rho', data=time_corr, ax=ax)
        if feature in ['alexnet', 'expanse', 'facingness', 'valence']:
            ax.set_ylabel('Spearman rho')
        else:
            ax.set_ylabel('')
            
        if feature in ['communication', 'valence', 'arousal']:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.vlines(x=[0, 0.5], ymin=0, ymax=ymax,
                colors='gray', linestyles='dashed', zorder=0)
        ax.hlines(y=0, xmin=time_corr.time.min(), xmax=time_corr.time.max(),
                colors='gray', linestyles='solid', zorder=0)
        ax.set_xlim([time_corr.time.min(), time_corr.time.max()])
        ax.set_ylim([ymin, ymax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(feature)
    plt.tight_layout()
    plt.savefig(out_file)


def plot_eeg_fmri_rsa(rsa, out_file):
    roi_group = rsa.groupby('roi')
    _, axes = plt.subplots(3, 3, sharey=True, sharex=True)
    axes = axes.flatten()
    ymin, ymax = rsa['Spearman rho'].min(), rsa['Spearman rho'].max()
    for ax, (roi, roi_corr) in zip(axes, roi_group):
        sns.lineplot(x='time', y='Spearman rho', data=roi_corr, ax=ax)
        if roi in ['EVC', 'LOC', 'pSTS']:
            ax.set_ylabel('Spearman rho')
        else:
            ax.set_ylabel('')

        if roi in ['pSTS', 'face-pSTS', 'aSTS']:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', length=0)

        ax.vlines(x=[0, 0.5], ymin=0, ymax=ymax,
                colors='gray', linestyles='dashed', zorder=0)
        ax.hlines(y=0, xmin=roi_corr.time.min(), xmax=roi_corr.time.max(),
                colors='gray', linestyles='solid', zorder=0)
        ax.set_xlim([roi_corr.time.min(), roi_corr.time.max()])
        ax.set_ylim([ymin, ymax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(roi)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_eeg_feature_decoding(out_file, results, features, hue=None):
    feature_group = results.groupby('feature')
    _, axes = plt.subplots(int(np.ceil(len(features)/3)), 3,
                        figsize=(10,8),
                        sharey=True, constrained_layout=True)
    axes = axes.flatten()
    trim_axs(axes, len(features))
    ymin, ymax = results['r'].min(), results['r'].max()
    for ax, (feature, time_corr) in zip(axes, feature_group):
        sns.lineplot(x='time', y='r', data=time_corr, ax=ax, hue=hue)
        if feature in ['indoor', 'agent_distance', 'communication']:
            ax.set_ylabel('Prediction (r)')
        else:
            ax.set_ylabel('')
            
        if feature in ['communication', 'valence', 'arousal']:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.vlines(x=[0, 0.5], ymin=0, ymax=ymax,
                colors='gray', linestyles='dashed', zorder=0)
        ax.hlines(y=0, xmin=time_corr.time.min(), xmax=time_corr.time.max(),
                colors='gray', linestyles='solid', zorder=0)
        ax.set_xlim([time_corr.time.min(), time_corr.time.max()])
        ax.set_ylim([ymin, ymax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(feature)
        if hue is not None:
            if feature == 'arousal':
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            else: 
                ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_pairwise_decoding(results, out_file):
    results.rename(columns={'accuracy': 'distance'}, inplace=True)
    avg_results = results.groupby('time').mean(numeric_only=True)
    ymin, ymax = avg_results.distance.min(), avg_results.distance.max()

    _, ax = plt.subplots()
    sns.lineplot(x='time', y='distance', data=avg_results, ax=ax)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Time (s)')
    ax.vlines(x=[0, 0.5], ymin=0, ymax=ymax,
        colors='gray', linestyles='dashed', zorder=0)
    ax.hlines(y=.5, xmin=results.time.min(), xmax=results.time.max(),
        colors='gray', linestyles='solid', zorder=0)
    ax.set_xlim([results.time.min(), results.time.max()])
    ax.set_ylim([ymin, ymax])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file)
