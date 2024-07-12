import argparse
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from src.stats import perm, bootstrap, sign_square, corr2d, calculate_p, cluster_correction
from src import loading
import json
from glob import glob
from tqdm import tqdm


file_patterns = {'feature_decoding': 'x-eeg_y-scene-primitive-social-affective',
                 'fmri_encoding': 'x-eeg_y-fmri',
                 'eeg_feature_shared': []}


class groupAnalysis:
    def __init__(self, args):
        self.process = 'groupAnalysis'
        self.n_perm = args.n_perm
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.resample_rate = args.resample_rate
        self.in_dir = args.in_dir
        self.out_dir = args.out_dir
        self.summary_stat = args.summary_stat
        valid_err_msg = f"--summary_stat must be feature_decoding, fmri_encoding, or eeg_feature_shared, {self.summary_stat}"
        assert (self.summary_stat in ['feature_decoding', 'fmri_encoding', 'eeg_feature_shared']), valid_err_msg
        self.file_pattern = file_patterns[self.summary_stat]
        print(vars(self))

    def load_targets(self):
        if self.summary_stat == 'feature_decoding':
            y_data = loading.load_behavior(self.fmri_dir)
            targets = [col for col in y_data.columns if 'rating' in col]
            out = {'subj_id': list(np.nan(len(targets))), 
                  'targets': targets}
        else:
            _, y_data = loading.load_fmri(self.fmri_dir, roi_mean=True)
            out = {'subj_id': y_data.subj_id.to_list(),
                    'targets': y_data.roi_name.to_list()}
        return out

    def get_time_map(self):
        arr = np.arange(self.start_time, self.end_time, self.resample_rate)
        return {str(i).zfill(3): time for i, time in enumerate(arr)}

    def compute_p(self, r2, null):
        out = []
        for itarget in tqdm(range(r2.shape[-1]), total=r2.shape[-1], desc='Cluster correction per target', leave=True):
            p = []
            for itime in range(r2.shape[0]):
                p.append(calculate_p(null[itime, :, itarget], r2[itime, itarget], self.n_perm, H0='greater'))
            p = np.array(p)
            corrected_p = np.expand_dims(cluster_correction(r2[..., itarget], p, null[..., itarget], verbose=True), axis=1)
            out.append(corrected_p)
        return np.concatenate(out, axis=1)

    def compute_ci(self, var):
        return np.percentile(var, [2.5, 97.5], axis=1)    

    def reorg(self, time_map, targets, r2, p, ci):
        out = []
        for time_ind, time in time_map.items():
            itime = int(time_ind)
            for itarget, (subj_id, target) in enumerate(zip(targets['subj_id'], targets['targets'])):
                out.append({'time': time, 'time_ind': time_ind,
                            'subj_id': subj_id, 'target': target,
                            'r2': r2[itime, itarget], 'time_ind': time_ind, 'time': time,
                            'p': p[itime, itarget],
                            'lower_ci': [0, itime, itarget],
                            'upper_ci': [1, itime, itarget]
                            })
        return pd.DataFrame(out)

    def load_data(self, stat_names):
        out = []
        for stat_name in stat_names:
            files = glob(f'{self.in_dir}/sub*{self.file_pattern}_stat-{stat_name}*')
            result = []
            for file in tqdm(files, total=len(files), desc=f'Loading {stat_name}'):
                result.append(np.load(file))
            out.append(np.mean(result, axis=0))
        return out

    def load_shared_data(self, stat_names):
        out = []
        for stat_name in stat_names:
            feature = np.load(f'{self.in_dir}/x-alexnet-moten-scene-primitive-social-affective_y-fmri_stat-{stat_name}.npy')
            eeg_files = sorted(glob(f'{self.in_dir}/sub*x-eeg_y-fmri_stat-{stat_name}*'))
            feature_eeg_files = sorted(glob(f'{self.in_dir}/sub*x-eeg-alexnet-moten-scene-primitive-social-affective_y-fmri_stat-{stat_name}*')) 
            result = []
            for eeg_file, feature_eeg_file in tqdm(zip(eeg_files, feature_eeg_files), total=len(eeg_files), desc=f'Loading {stat_name}'):
                eeg = np.load(eeg_file)
                feature_eeg = np.load(feature_eeg_file)
                result.append((feature + eeg) - feature_eeg)
            out.append(np.mean(result, axis=0))
        return out

    def save_df(self, df):
        df.to_csv(f'{self.prefix}_summary.csv', index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        if self.summary_stat != 'eeg_feature_shared':
            r2, null, var = self.load_data(['r2', 'null', 'var'])
        else:
            r2, null, var = self.load_shared_data(['r2', 'null', 'var'])
        ci = self.compute_ci(var)
        p = self.compute_p(r2, null)
        print(f'{p.shape=}')
        del var, null

        time_map = self.get_time_map()
        targets = self.load_targets()
        results = self.reorg(time_map, targets, r2, p, ci)
        print(results.head())
        self.mk_out_dir()
        self.save_df(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--in_dir', '-r', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegStats')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/groupAnalysis')
    parser.add_argument('--summary_stat', '-s', type=str, default='feature_decoding',
                        help='the type of analysis to summarize options are feature_decoding, fmri_encoding, eeg_feature_shared')
    parser.add_argument('--n_perm', type=int, default=int(5e3),
                        help='number of permutations/resamples to run')         
    parser.add_argument('--start_time', type=int, default=-200,
                        help='time that the timecourse starts in milliseconds')
    parser.add_argument('--end_time', type=int, default=1000,
                        help='time that the timecourse ends in milliseconds')
    parser.add_argument('--resample_rate', type=float, default=2.5,
                        help='sampling interval of the signal in milliseconds')       
    args = parser.parse_args()
    groupAnalysis(args).run()


if __name__ == '__main__':
    main()