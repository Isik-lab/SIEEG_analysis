#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from src.tools import corr2d
from src.cv import ShuffleBinLeaveOneOut
from itertools import combinations


# # Pilot Data Check

# ## Load and filter data

# In[3]:

process = 'CleanedEEG'
data_path = '../../data'
figure_path = f'../../reports/figures/{process}'
eeg_path = f'{data_path}/interim/SIdyads_EEG_pilot'
trial_path = f'{data_path}/raw/SIdyads_trials_pilot'
subj = 'subj003_10182023'
subj_out = subj.split('_')[0]
preproc_file = f'{eeg_path}/{subj}/{subj}_preproc.mat'
trial_files = f'{trial_path}/{subj}/timingfiles/*.csv'
out_dir = f'{data_path}/interim/{process}/{subj_out}'
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(figure_path).mkdir(parents=True, exist_ok=True)

# Parameters
n_perm = 20  # number of permutations
n_pseudo = 4  # number of pseudo-trials


# In[4]:


artifact_trials = loadmat(preproc_file)['idx_badtrial'].squeeze().astype('bool')
trials = []
for run, tf in enumerate(sorted(glob(trial_files))):
    df = pd.read_csv(tf)
    df['run'] = run
    df['run_file'] = tf
    trials.append(df)
trials = pd.concat(trials).reset_index(drop=True)
trials['artifact_trial'] = artifact_trials
n_orig_trials = len(trials)
print(f'original number of trials: {n_orig_trials}')
trials = trials[~trials.artifact_trial]
print(f'number of trials after EEG preprocessing: {len(trials)}')
print(f'percent of trials removed: {np.round(((n_orig_trials - len(trials))/n_orig_trials)*100):.0f}%')


# In[5]:


catch_trials = np.invert(trials.condition.to_numpy().astype('bool'))
response_trials = trials.response.to_numpy().astype('bool')
trial_to_remove = catch_trials + response_trials
trials = trials[~trial_to_remove].reset_index(drop=True)
print(f'number of catch trials plus false alarms: {np.sum(trial_to_remove)}')
print(f'final number of trials: {len(trials)}')
print()


# In[6]:


data_file = f'{eeg_path}/{subj}/{subj}_trialonly.mat'
data_array = loadmat(data_file)['trl']

# remove catch and false alarm trials
data_array = data_array[np.invert(trial_to_remove), :, :]

# In[7]:


trials['video_id'] = pd.factorize(trials['video_name'])[0]
n_sensors = data_array.shape[1]
n_time = data_array.shape[-1]
n_conditions = len(trials.video_id.unique())
print(f'n_sensors = {n_sensors}')
print(f'n_time = {n_time}')
print(f'n_conditions = {n_conditions}')


# ## Reliability

# In[8]:


grouped_indices = trials.groupby(['video_name']).indices
split_half = np.zeros((len(grouped_indices), 2, n_sensors, n_time))
average_data_array = np.zeros((len(grouped_indices), n_sensors, n_time))
for i, (_, val) in enumerate(grouped_indices.items()):
    split_half[i, 0, :, :] = data_array[val[::2], :, :].mean(axis=0, keepdims=True)
    split_half[i, 1, :, :] = data_array[val[1::2], :, :].mean(axis=0, keepdims=True)
    average_data_array[i, :, :] = data_array[val, :, :].mean(axis=0, keepdims=True)

# In[10]:


reliability = []
for isample in range(n_time):
    reliability.append(corr2d(split_half[:, 0, :, isample].squeeze(), split_half[:, 1, :, isample].squeeze()))
reliability = np.vstack(reliability)
_, ax = plt.subplots(2)
ax[0].plot(reliability)
ax[1].plot(reliability.mean(axis=-1))
plt.savefig(f'{figure_path}/subj-{subj_out}_reliability.png')


# ## Pairwise Decoding

# In[13]:


conditions = trials.video_id.to_numpy()
conditions_list = sorted(np.unique(conditions))
conditions_nCk = list(combinations(conditions_list, 2))

sort_indices = np.argsort(conditions)
y = conditions[sort_indices]
X = data_array[sort_indices, ...]
print(f'X shape = {X.shape}')
print(f'y shape = {y.shape}')


# In[14]:


np.random.seed(0)
cv = ShuffleBinLeaveOneOut(y, n_iter=n_perm, n_pseudo=n_pseudo) 
out = {'conditions': conditions_list,
      'conditions_nCk': conditions_nCk,
      'n_sensors': n_sensors,
      'n_conditions': n_conditions,
      'n_time': n_time,
      'n_perm': n_perm,
      'n_pseudo': n_pseudo,
      'X': X,
      'y': y,
      'train_indices': [],
      'test_indices': [],
      'permutation_number': 0,
      'labels_pseudo_train': [],
      'labels_pseudo_test': [],
      'ind_pseudo_train': [],
      'ind_pseudo_test': []}
for f, (train_indices, test_indices) in enumerate(cv.split(X)):
    out['permutation_number'] = f
    out['train_indices'] = train_indices
    out['test_indices'] = train_indices
    out['labels_pseudo_train'] = cv.labels_pseudo_train
    out['labels_pseudo_test'] = cv.labels_pseudo_test
    out['ind_pseudo_train'] = cv.ind_pseudo_train
    out['ind_pseudo_test'] = cv.ind_pseudo_test
    np.savez(f'{out_dir}/data4rdms_perm-{str(f).zfill(2)}.npz', **out)


# In[ ]:




