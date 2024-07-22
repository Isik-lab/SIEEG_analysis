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
mean_df = mean_df.groupby(['time', 'roi_name']).mean(numeric_only).reset_index()

sns.lineplot(x='time', y='value', hue='roi_name', data=mean_df)
plt.save_results('data/interim/PlotBack2Back/results.pdf')