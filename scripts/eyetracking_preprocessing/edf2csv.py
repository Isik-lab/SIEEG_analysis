#/Applications/anaconda3/envs/nibabel/bin/python
import pandas as pd
import numpy as np
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('mode.chained_assignment',  None)

subj = 'sub-10'
top_path = '/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data'
in_path = f'{top_path}/raw/SIdyads_trials'
out_path = f'{top_path}/interim/SIdyads_eyetracking'
Path(f'{in_path}/{subj}/asc').mkdir(exist_ok=True, parents=True)
Path(out_path).mkdir(exist_ok=True, parents=True)

ms_to_s = 1000
post_stim = 1.25 * ms_to_s
pre_stim = .2 * ms_to_s

out_data = []
for edf_file in tqdm(glob(f'{in_path}/{subj}/edfs/*.edf')):
    run_str = edf_file.split('run')[-1].split('_')[0]
    run = int(run_str)
    events_file = f'{in_path}/{subj}/asc/run{run_str}_events.asc'
    samples_file = f'{in_path}/{subj}/asc/run{run_str}_samples.asc'
    left_right = 'left'

    if not os.path.exists(events_file):
        os.system(f'edf2asc -y -e {edf_file} {events_file}')

    if not os.path.exists(samples_file):
        os.system(f'edf2asc -y -s {edf_file} {samples_file}')

    with open(events_file) as f:
        events=f.readlines()

    df_samples = pd.read_table(samples_file, index_col=False,
                    names=['time', 'gaze_x', 'gaze_y', 'pupil_size',
                            'empty', 'target_x', 'target_y', 'target_distance'])
    df_samples['target_distance'] = df_samples['target_distance'].map(lambda x: x.rstrip(' .............'))
    df_samples.drop(columns=['empty'], inplace=True)
    gaze_inds = np.isclose(df_samples.pupil_size, 0) #Find the missing gaze points
    gaze_inds += df_samples['gaze_x'].str.contains('   .') # Find missing gaze
    target_inds = df_samples['target_distance'].str.contains('[a-zA-Z]') # Find missing targets
    df_samples.loc[gaze_inds, ['gaze_x', 'gaze_y']] = '-7777' #Fill with a value to be replaced
    df_samples.loc[target_inds, ['target_x', 'target_y', 'target_distance']] = '-7777' #Fill with a value to be replaced
    for col in df_samples.columns:
        df_samples[col] = df_samples[col].astype('float')
    df_samples.loc[gaze_inds, ['gaze_x', 'gaze_y', 'pupil_size']] = np.nan #Replace missing gaze data with np.nan
    df_samples.loc[gaze_inds, ['target_x', 'target_y', 'target_distance']] = np.nan #Replace missing target data with np.nan
    
    # keep only lines starting with "MSG"
    events=[ev for ev in events if ev.startswith("MSG")]
    experiment_start_index=np.where(["TRIALID" in ev for ev in events])[0][0]
    events=events[experiment_start_index:]

    df_ev=pd.DataFrame([ev.split() for ev in events])
    df_ev = df_ev[[1, 2, 3]]
    df_ev.columns = ['time', 'event', 'data']
    df_ev.loc[df_ev.data.isna(), 'data'] = df_ev.loc[df_ev.data.isna(), 'time'].copy()
    df_ev.drop(columns=['time'], inplace=True)
    df_ev_pivot = []
    for i, j in df_ev.groupby('event'):
        j.drop(columns='event', inplace=True)
        j.rename(columns={'data': i}, inplace=True)
        j.reset_index(drop=True, inplace=True)
        df_ev_pivot.append(j)
    df_ev_pivot = pd.concat(df_ev_pivot, axis=1)
    df_ev_pivot['TRIALID'] = df_ev_pivot['TRIALID'].astype('int')
    df_ev_pivot[['STIMULUS_START', 'STIMULUS_OFF']] = df_ev_pivot[['STIMULUS_START', 'STIMULUS_OFF']].astype('int')

    df_runfile = pd.read_csv(f'{in_path}/{subj}/runfiles/run{run_str}.csv')
    df_ev_pivot = df_ev_pivot.join(df_runfile)

    for i, row in tqdm(df_ev_pivot.iterrows(), total=len(df_ev_pivot)):
        onset = row.STIMULUS_START - pre_stim
        offset = row.STIMULUS_START + post_stim
        cur = df_samples[(df_samples.time >= onset) & (df_samples.time <= offset)]
        cur.reset_index(drop=True, inplace=True)

        # round time to the nearest ms 
        cur['time'] = cur['time'].round(decimals=3)

        # center to stimulus start and convert to seconds
        cur['time'] = ((cur['time'] - row.STIMULUS_START) / ms_to_s)

        # adjust the time by 1 ms if necessary
        if cur['time'].min() != -1 * (pre_stim / ms_to_s):
            cur['time'] = cur['time'] - .001

        # fill in data that is needed in the frame
        cur['run'] = run 
        cur['trial'] = row.TRIALID
        cur['video_name'] = row.video_name
        cur['block'] = row.block
        cur['condition'] = row.condition
        out_data.append(cur)
df = pd.concat(out_data)
df.sort_values(by=['run', 'trial', 'time'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(f'{out_path}/{subj}_eyetracking.csv.gz', index=False, compression='gzip')
