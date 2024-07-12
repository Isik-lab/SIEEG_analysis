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


class eegStats:
    def __init__(self, args):
        self.process = 'eegStats'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.roi_mean = args.roi_mean
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.resample_rate = args.resample_rate
        self.n_perm = args.n_perm
        self.compute_stats = args.compute_stats
        self.y_names = args.y_names
        self.pred_file_pattern = args.pred_file_pattern
        sub = self.pred_file_pattern.split("/")[-1].split("time")[0]
        x_y_names = self.pred_file_pattern.split("/")[-1].split("_x-")[-].split("y_hat")[0]
        self.prefix = f'{self.out_dir}/{sub}_x-{x_y_names}'
        print(vars(self)) 
        self.out_dir = args.out_dir
        self.fmri_dir = args.fmri_dir
        
        

    @staticmethod
    def compute_score(true, pred):
        return np.expand_dims(sign_square(corr2d(true, pred)), axis=0)
    
    def compute_perm_dist(self, true, pred):
        return np.expand_dims(perm(true, pred, n_perm=self.n_perm, square=True), axis=0)
    
    def compute_var_dist(self, true, pred):
        return np.expand_dims(bootstrap(true, pred, n_perm=self.n_perm, square=True), axis=0)

    def load_pred(self, file):
        return pd.read_csv(file).to_numpy()

    def load_true(self):
        true, _ = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
        behavior = loading.load_behavior(self.fmri_dir)
        idx = behavior.loc[behavior.stimulus_set == 'test'].index
        return true.iloc[idx].to_numpy()

    def compute_dists(self, true):
        r2, null, var = [], [], []
        files = sorted(glob(self.pred_file_pattern))
        for file in tqdm(files, total=len(files), desc='Computing dists through time'):
            pred = self.load_pred(file)
            r2.append(self.compute_score(true, pred))
            null.append(self.compute_perm_dist(true, pred))
            var.append(self.compute_var_dist(true, pred))
        return np.concatenate(r2, axis=0), np.concatenate(null, axis=0), np.concatenate(var, axis=0)

    # def load_targets(self):
    #     if 'fmri' not in self.y_names:
    #         y_data = loading.load_behavior(self.fmri_dir)
    #         out = {'targets': [col for col in y_data.columns if 'rating' in col]}
    #     else:
    #         _, y_data = loading.load_fmri(self.fmri_dir, roi_mean=self.roi_mean)
    #         if self.roi_mean:
    #             out = {'subj_id': y_data.subj_id.to_list(),
    #                    'targets': y_data.roi_name.to_list()}
    #         else:
    #             out = {'voxel_id': y_data.voxel_id.to_list(),
    #                    'targets': y_data.roi_name.to_list()}
    #     return out

    # def get_time_map(self):
    #     arr = np.arange(self.start_time, self.end_time, self.resample_rate)
    #     return {str(i).zfill(3): time for i, time in enumerate(arr)}

    # def compute_p(self, r2, null):
    #     out = []
    #     for itarget in tqdm(range(r2.shape[-1]), total=r2.shape[-1], desc='Cluster correction per target'):
    #         p = []
    #         for itime in range(r2.shape[0]):
    #             p.append(calculate_p(null[itime, :, itarget], r2[itime, itarget], self.n_perm, H0='greater'))
    #         p = np.array(p)
    #         corrected_p = np.expand_dims(cluster_correction(r2[..., itarget], p, null[..., itarget]), axis=1)
    #         out.append(corrected_p)
    #     return np.concatenate(out, axis=1)

    # def compute_ci(self, var):
    #     return np.percentile(var, [2.5, 97.5], axis=1)    

    # def reorg(self, time_map, targets, r2, p, ci):
    #     out = []
    #     for time_ind, time in time_map.items():
    #         itime = int(time_ind)
    #         for itarget, (subj_id, roi_name) in enumerate(zip(targets['subj_id'], targets['targets'])):
    #             out.append({'time': time, 'time_ind': time_ind,
    #                         'subj_id': subj_id, 'roi_name': roi_name,
    #                         'r2': r2[itime, itarget], 'time_ind': time_ind, 'time': time,
    #                         'p': p[itime, itarget],
    #                         'lower_ci': [0, itime, itarget],
    #                         'upper_ci': [1, itime, itarget]
    #                         })
    #     return pd.DataFrame(out)

    def save_array(self, arr, suffix):
        np.save(f'{self.prefix}_{suffix}.npy', arr)

    # def save_df(self, df):
    #     df.to_csv(f'{self.prefix}_summary.csv', index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        # time_map = self.get_time_map()
        # targets = self.load_targets()
        true = self.load_true()
        r2, null, var = self.compute_dists(true)
        self.mk_out_dir()
        self.save_array(r2, 'r2')
        self.save_array(null, 'null')
        self.save_array(var, 'var')
        # if self.compute_stats: 
        #     ci = self.compute_ci(var)
        #     print(f'{ci.shape=}')
        #     p = self.compute_p(r2, null)
        #     print(f'{p.shape=}')
        #     results = self.reorg(time_map, targets, r2, p, ci)
        #     print(results.head())
        #     self.save_df(results)
        print('finished')


def main():
    parser = argparse.ArgumentParser(description='Decoding behavior or fMRI from EEG responses')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI')
    parser.add_argument('--pred_file_pattern', '-p', type=str, help='prediction file',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/encodeDecode/eeg/sub-03_time-*_x-eeg_y-fmri_yhat.csv.gz')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/eegStats')
    parser.add_argument('--roi_mean', action=argparse.BooleanOptionalAction, default=True,
                        help='predict the roi mean response instead of voxelwise responses')
    parser.add_argument('--compute_stats', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to perform significance tessting and cluster correction')
    parser.add_argument('--y_names', '-y', type=str, default='["fmri"]',
                        help='a list of data names to be used as regression target')
    parser.add_argument('--start_time', type=int, default=-200,
                        help='time that the timecourse starts in milliseconds')
    parser.add_argument('--end_time', type=int, default=1000,
                        help='time that the timecourse ends in milliseconds')
    parser.add_argument('--resample_rate', type=float, default=2.5,
                        help='sampling interval of the signal in milliseconds')
    parser.add_argument('--n_perm', type=int, default=int(5e3),
                        help='number of permutations/resamples to run')                   
    args = parser.parse_args()
    eegStats(args).run()


if __name__ == '__main__':
    main()