import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt


def feature2color(key=None):
    d = dict()
    d['expanse'] = np.array([0.9433962264150944, 0.4716981132075472, 0.35471698113207545, 1.])
    d['object'] = np.array([0.879245283018868, 0.3169811320754717, 0.36981132075471695, 1.])
    d['agent_distance'] = np.array([00.7547169811320755, 0.2339622641509434, 0.4339622641509434, 1.])
    d['facingness'] = np.array([0.6150943396226415, 0.1811320754716981, 0.47547169811320755, 1.])
    d['joint_action'] = np.array([0.47547169811320755, 0.13584905660377358, 0.49056603773584906, 1.])
    d['communication'] = np.array([0.33584905660377357, 0.07924528301886792, 0.47547169811320755, 1.])
    d['valence'] = np.array([0.19245283018867926, 0.06037735849056604, 0.3886792452830189, 1.])
    d['arousal'] = np.array([0.06792452830188679, 0.04905660377358491, 0.18490566037735848, 1.])
    if key is not None:
        return d[key]
    else:
        return d

data_dir = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis'
process = 'PlotExampleRatings'
figure_dir = f'{data_dir}/data/interim/{process}'
if os.path.exists(figure_dir):
    shutil.rmtree(figure_dir)
Path(figure_dir).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(f'{data_dir}/data/raw/annotations/annotations.csv')
features = ['expanse',  'object', 'agent_distance', 'facingness', 
             'joint_action', 'communication', 'valence', 'arousal']
feature_ticks = ['expanse',  'object', 'agent distance', 'facingness', 
              'joint action', 'communication', 'valence', 'arousal']
df.drop(columns=['indoor'], inplace=True)
df.rename(columns={'transitivity': 'object'}, inplace=True)
print(df.head())

df = df.sample(n=10, axis=0) # Sample only some of the videos

for i, feature in enumerate(features):
    df.rename(columns={feature: f'Rating{i}'}, inplace=True)
print(df.head())

df = pd.wide_to_long(df, stubnames='Rating', i='video_name', j='feature').reset_index(drop=False)
for i, feature in enumerate(features):
    df.feature.replace({i: feature}, inplace=True)
print(df.feature.unique())
df['feature'] = pd.Categorical(df['feature'], categories=features, ordered=True)


for vid, cur_df in df.groupby('video_name'):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='poster', style='white', rc=custom_params)
    _, ax = plt.subplots(1, figsize=(6,6))

    sns.barplot(x='feature', y='Rating',
                data=cur_df, ax=ax, palette='gray')
    ax.set_xlabel('')
    ax.set_ylim([0, 1])

    # Change the ytick font size
    label_format = '{:.1f}'
    plt.locator_params(axis='y', nbins=3)
    y_ticklocs = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
    ax.set_yticklabels([label_format.format(x) for x in y_ticklocs])
    ax.set_ylabel('Rating')

    ax.set_xticklabels(feature_ticks, ha='right', rotation=45)

    # Manipulate the color and add error bars
    for bar, feature in zip(ax.patches, features):
        color = feature2color(feature)
        bar.set_color(color)
    ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{vid.replace('mp4', 'pdf')}")
