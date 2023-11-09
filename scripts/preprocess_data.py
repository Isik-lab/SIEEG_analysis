#/Applications/anaconda3/envs/nibabel/bin/python
import os
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from scipy.io import loadmat
from src import preprocessing


class PreprocessData:
    def __init__(self, args):
        self.process = 'PreprocessData'
        self.data_dir = args.data_dir
        self.sid = f'subj{str(args.sid).zfill(3)}'
        self.channels = []
        self.resample_rate = args.resample_rate
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.regress_gaze = args.regress_gaze
        self.eeg_fps = 1000 #Hz
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}.csv.gz'
        print(vars(self))

    def load_eeg(self):
        print('loading eeg...')
        df = pd.read_csv(f'{self.data_dir}/interim/SIdyads_EEG_pilot/{self.sid}/{self.sid}_trials.csv.gz')
        self.channels = df.drop(columns=['time','trial', 'offset']).columns
        df['offset_eyetrack'] = (df.offset / self.eeg_fps)
        return df 

    def load_artifact(self):
        # Load the trials that were removed in preprocessing
        preproc_file = f'{self.data_dir}/interim/SIdyads_EEG_pilot/{self.sid}/{self.sid}_preproc.mat'
        preproc = loadmat(preproc_file)
        return preproc['idx_badtrial'].squeeze().astype('bool')

    def load_trials(self):
        trial_files = f'{self.data_dir}/raw/SIdyads_trials_pilot/{self.sid}/timingfiles/*.csv'
        trials = []
        for run, tf in enumerate(sorted(glob(trial_files))):
            t = pd.read_csv(tf)
            t['run'] = run
            t['run_file'] = tf
            trials.append(t)
        trials = pd.concat(trials).reset_index(drop=True)
        trials.reset_index(inplace=True)
        trials.rename(columns={'index': 'trial'}, inplace=True)
        return trials
    
    def load_eyetracking(self):
        file = f'{self.data_dir}/interim/SIdyads_eyetracking_pilot/{self.sid}_eyetracking.csv.gz'
        if os.path.exists(file):
            print('loading and processing eyetracking...')
            df = pd.read_csv(file)
            df.drop(columns=['video_name', 'block', 'condition'], inplace=True)

            # uniquely number all the trials in the eyetracking data
            df.sort_values(by=['run', 'trial', 'time'], inplace=True)
            df['trial'] = df.groupby('time').cumcount()
            df.set_index(['trial', 'time'], inplace=True)
            return df 
        else:
            return None

    def save(self, df):
        print('saving...')
        df.to_csv(self.out_file, compression='gzip', index=False)
        print('Finished!')

    def run(self):
        eeg = preprocessing.downsample_and_filter(self.load_eeg(),
                                                  self.resample_rate,
                                                  self.start_time, self.end_time)
        trials = preprocessing.filter_trials(self.load_trials(), self.load_artifact())
        eyetracking = self.load_eyetracking()
        if eyetracking is not None:
            eyetracking = preprocessing.process_eyetracking(eyetracking, self.load_artifact(),
                                                            eeg[['trial', 'offset_eyetrack']].drop_duplicates(),
                                                            self.resample_rate, self.start_time, self.end_time)
        combined = preprocessing.combine_data(eeg, trials, eyetracking)

        if self.regress_gaze:
            combined = preprocessing.regress_out_gaze(combined, self.channels)

        combined = preprocessing.filter_catch_trials(combined)
        combined = preprocessing.label_repetitions(combined)
        self.save(combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=9)
    parser.add_argument('--resample_rate', type=str, default='4ms')
    parser.add_argument('--start_time', type=float, default=-0.2)
    parser.add_argument('--end_time', type=float, default=1)
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    args = parser.parse_args()
    PreprocessData(args).run()


if __name__ == '__main__':
    main()