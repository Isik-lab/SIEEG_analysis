import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.stats import calculate_p, cluster_correction
from pathlib import Path


class ROILatencyDist:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.out_dir = args.out_dir
        self.roi = args.roi
        self.fmri_sub = args.fmri_sub
        self.n_resamples = args.n_resamples
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def load_data(self):
        df = pd.read_parquet(f'{self.input_dir}/individual_stats.parquet')
        df = df.loc[(df.roi_name == self.roi) & (df.fmri_subj_id == self.fmri_sub)].reset_index()
        return df
    
    def run(self):
        df = self.load_data()
        null_cols = [col for col in df.columns if 'null_perm_' in col]
        
        n_subjs = df.eeg_subj_id.nunique()
        n_time = df.time.nunique()
        df.sort_values(['eeg_subj_id', 'time'], inplace=True)
        time = df.time.unique()
        scores_null = df[null_cols].to_numpy().reshape((n_subjs, n_time, -1))
        scores = df['score'].to_numpy().reshape((n_subjs, n_time))
        onsets = []
        for isample in tqdm(range(self.n_resamples), desc=f'Calculating dist'):
            np.random.seed(isample)
            idx = np.random.choice(n_subjs, n_subjs)
            scores_null_sample = scores_null[idx].mean(axis=0).T
            scores_sample = scores[idx].mean(axis=0).T
            ps = calculate_p(scores_null_sample, scores_sample, 5000, 'greater')
            ps_corrected = cluster_correction(scores_sample.T, ps.T, scores_null_sample.T)
            onset = time[ps_corrected < 0.05].min() if len(time[ps_corrected < 0.05]) > 0 else np.nan
            onsets.append(onset)
        onsets = np.array(onsets)
        np.save(f'{self.out_dir}/sub-{self.fmri_sub}_roi-{self.roi}.npy')

def main():
    parser = argparse.ArgumentParser(description='Compute the latency distributions')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ROILatencyDist')
    parser.add_argument('--input_dir', '-i', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/PlotNuisanceROIDecoding')
    parser.add_argument('--roi', '-r', type=str, default='EVC',
                        help='roi to calculate dist for')
    parser.add_argument('--fmri_sub', '-s', type=int, default=1,
                        help='fmri sub to calculate dist for')
    parser.add_argument('--n_resamples', '-n', type=int, default=1000,
                        help='number of time to resample the participants')
    args = parser.parse_args()
    ROILatencyDist(args).run()


if __name__ == '__main__':
    main()