import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob

df = []
for i_file, file in enumerate(glob('data/interim/Back2Back/*.csv.gz')):
    subj_df = pd.read_csv(file)
    subj_df['eeg_subj'] = i_file
    df.append(subj_df)
df = pd.concat(df, ignore_index=True)
mean_df = df.groupby(['time', 'subj_id', 'roi_name']).mean(numeric_only=True).reset_index()
mean_df = df.groupby(['time', 'roi_name']).mean(numeric_only=True).reset_index()


_, axes = plt.subplots(3, 3, sharex=True, sharey=True)
axes = axes.flatten()
for ax, (roi_name, roi_df) in zip(axes, mean_df.groupby('roi_name')):
    sns.lineplot(x='time', y='value', data=roi_df, ax=ax)
    ax.set_title(roi_name)
plt.savefig('data/interim/PlotBack2Back/results.pdf')