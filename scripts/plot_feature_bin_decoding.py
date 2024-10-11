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
import pickle

def p_to_text(p):
    if 0.05 > p >= 0.01:
        text = "*" 
    elif 0.01 > p >= 0.001:
        text = "**"
    elif 0.001 > p:
        text = "***"
    else:
        text = " "
    return text

simplified_plotting = False 
if simplified_plotting:
    features = ['expanse', 'agent_distance', 'communication']
    title_names = ['spatial expanse', 'agent distance', 'communication']
    colors = ['#F5DD40', '#8558F4', '#73D2DF']
else:
    features = ['expanse', 'object', 'agent_distance', 'facingness',
                'joint_action', 'communication', 'valence', 'arousal']
    title_names = ['spatial expanse', 'object directedness', 'agent distance', 'facingness',
                   'joint action', 'communication', 'valence', 'arousal']
    colors = ['#F5DD40', '#F5DD40', '#8558F4', '#8558F4', '#73D2DF', '#73D2DF', '#D57D7F', '#D57D7F']


out_path = 'data/interim/PlotFeatureBinDecoding'
Path(out_path).mkdir(exist_ok=True, parents=True)
files = glob('data/interim/FeatureBinRegression/sub*.parquet')

if not os.path.isfile(f'{out_path}/stats_annotations.pkl'):
    #Load data
    df = []
    for file in tqdm(files, desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    print('Finished loading files')
    print(df.head(50))

    #Average across EEG subjects
    mean_df = df.groupby(['time', 'feature']).mean(numeric_only=True).reset_index()
    print('Finished mean over EEG subjects')
    print(mean_df.head(50))
    # Group stats
    # Variance
    var_cols = [col for col in mean_df.columns if 'var_perm_' in col]
    scores_var = mean_df[var_cols].to_numpy()
    mean_df['low_ci'], mean_df['high_ci'] = np.percentile(scores_var, [2.5, 97.5], axis=1)
    mean_df.drop(columns=var_cols, inplace=True)
    # P-values
    null_cols = [col for col in mean_df.columns if 'null_perm_' in col]
    stats_df = []
    stats_annotations = {}
    for (feature, time), feature_df in mean_df.groupby(['feature', 'time']):
        scores_null = feature_df[null_cols].to_numpy().T
        scores = feature_df['value'].to_numpy().T
        p = calculate_p(scores_null, scores, 5000, 'greater')
        feature_df['p'] = p
        feature_df.drop(columns=null_cols, inplace=True)
        stats_df.append(feature_df)
        stats_annotations[feature, time] = p_to_text(p)
    stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True)
    print(stats_annotations)

    stats_df.to_csv(f'{out_path}/feature_plot.csv', index=False)
    with open(f'{out_path}/stats_annotations.pkl', 'wb') as f:
        pickle.dump(stats_annotations, f)
else:
    stats_df = pd.read_csv(f'{out_path}/feature_plot.csv')
    with open(f'{out_path}/stats_annotations.pkl', 'rb') as f:
        stats_annotations = pickle.load(f)

# Make categorical for plotting
stats_df = stats_df.loc[stats_df['feature'].isin(features)].reset_index(drop=True)
stats_df['feature'] = pd.Categorical(stats_df['feature'], categories=features, ordered=True)
stats_df['time'] = pd.Categorical(stats_df['time'], categories=['early', 'mid', 'late'], ordered=True)
print(stats_df.head(10))
print(stats_annotations)

# Plot the results
if simplified_plotting: 
    plt.tight_layout()
    plt.savefig(f'{out_path}/feature_plot.pdf')
else: 
    sns.set_context(context='paper', font_scale=2)
    plt.subplots(figsize=(5, 10))
    g = sns.catplot(
        data=stats_df, x="time", y="value", col="feature",
        kind="bar", height=4, aspect=.7,
        palette=sns.color_palette('flare')
    )
    g.set_axis_labels("", "Score ($r$)")
    g.set_xticklabels(["early", "mid", "late"])
    g.set_titles("{col_name}")
    g.set(ylim=(0, .4))

    # Add custom annotations
    for ax in g.axes.flat:
        feature = ax.get_title().strip()  # Extract feature name from subplot title
        for i, p in enumerate(ax.patches):
            time = ["early", "mid", "late"][i % 3]  # Determine time based on bar index
            annotation = stats_annotations.get((feature, time), '')
            ax.annotate(annotation,
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f'{out_path}/all_features_plot.pdf')