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
    # if 0.05 > p >= 0.01:
    #     text = "*" 
    # elif 0.01 > p >= 0.001:
    #     text = "**"
    # elif 0.001 > p:
    #     text = "***"
    if 0.05 > p: 
        text = "*"
    else:
        text = " "
    return text

lateral_stream = True
if lateral_stream:
    rois = ['EVC', 'MT', 'LOC', 'EBA', 'pSTS', 'aSTS']
    out_name = 'lateral'
else:
    rois = ['FFA', 'PPA']
    out_name = 'ventral'

out_path = 'data/interim/PlotROIBinDecoding'
Path(out_path).mkdir(exist_ok=True, parents=True)
files = glob('data/interim/fMRIBinRegression/*rois.parquet')

if not os.path.isfile(f'{out_path}/roi_plot.csv'):
    #Load data
    df = []
    for file in tqdm(files, desc='loading files'):
        subj_df = pd.read_parquet(file)
        subj_df['eeg_subj_id'] = file.split('/')[-1].split('_')[0]
        df.append(subj_df)
    df = pd.concat(df, ignore_index=True)
    df = df.loc[df.roi_name.isin(rois)].reset_index() # Filter to save time
    print('Finished loading files')
    print(df.head(50))

    #Average across EEG subjects
    mean_df = df.groupby(['time', 'fmri_subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
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
    for (time, subj, roi), feature_df in mean_df.groupby(['time', 'fmri_subj_id', 'roi_name']):
        scores_null = feature_df[null_cols].to_numpy().T
        scores = feature_df['value'].to_numpy().T
        p = calculate_p(scores_null, scores, 5000, 'greater')
        feature_df['p'] = p
        feature_df.drop(columns=null_cols, inplace=True)
        stats_df.append(feature_df)
        stats_annotations[time, subj, roi, 'p'] = p_to_text(p)
        stats_annotations[time, subj, roi, 'low_ci'] = feature_df['low_ci']
        stats_annotations[time, subj, roi, 'high_ci'] = feature_df['high_ci']
    stats_df = pd.concat(stats_df, ignore_index=True).reset_index(drop=True)
    print(stats_annotations)

    stats_df.to_csv(f'{out_path}/roi_plot.csv', index=False)
    with open(f'{out_path}/stats_annotations.pkl', 'wb') as f:
        pickle.dump(stats_annotations, f)
else:
    stats_df = pd.read_csv(f'{out_path}/roi_plot.csv')
    with open(f'{out_path}/stats_annotations.pkl', 'rb') as f:
        stats_annotations = pickle.load(f)

# Make categorical for plotting
stats_df = stats_df.loc[stats_df['roi_name'].isin(rois)].reset_index(drop=True)
stats_df['roi_name'] = pd.Categorical(stats_df['roi_name'], categories=rois, ordered=True)
stats_df['time'] = pd.Categorical(stats_df['time'], categories=['early', 'mid', 'late'], ordered=True)
stats_df['fmri_subj_id'] = stats_df['fmri_subj_id'].astype('str')
stats_df['fmri_subj_id'] = pd.Categorical(stats_df['fmri_subj_id'], categories=['1', '2', '3', '4'], ordered=True)

print(stats_annotations)

# Plot the results
sns.set_context(context='paper', font_scale=3)
g = sns.catplot(
    data=stats_df, x="time", 
    y="value", col="roi_name", hue="fmri_subj_id",
    kind="bar", height=5, aspect=1,
    legend=False,
    palette=sns.color_palette('magma')
)
g.set_axis_labels("", "Score ($r$)")
g.set_xticklabels(["early", "mid", "late"])
g.set_titles("{col_name}")
g.set(ylim=(0, .6))

# Add custom annotations
for ax in g.axes.flat:
    roi = ax.get_title().strip()  # Extract feature name from subplot title
    num_bars = len(ax.patches)
    num_time_points = 3  # early, mid, late
    
    for i, p in enumerate(ax.patches):
        subj = i // num_time_points  # Calculate subject based on bar index
        time = ["early", "mid", "late"][i % num_time_points]  # Determine time based on bar index
        
        stars = stats_annotations.get((time, subj+1, roi, 'p'), '')
        low_ci = stats_annotations.get((time, subj+1, roi, 'low_ci'), '')
        high_ci = stats_annotations.get((time, subj+1, roi, 'high_ci'), '')

        ax.annotate(stars,  # Use the annotation text
                    (p.get_x() + p.get_width() / 2., high_ci+0.015),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
        ax.vlines(x=p.get_x() + p.get_width() / 2.,
                  ymin=low_ci, ymax=high_ci, colors='k',
                  linewidth=2.5)

# Create a new axis for the legend
fig = plt.gcf()
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# Get the handles and labels from one of the existing axes
handles, labels = g.axes.flat[0].get_legend_handles_labels()

# Add the legend to the new axis
legend = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=len(labels), title="fMRI participant")

fig.set_size_inches(g.fig.get_size_inches()[0], g.fig.get_size_inches()[1] + 4.5)

plt.savefig(f'{out_path}/{out_name}.pdf')
