#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm


class fMRIWholeBrain:
    def __init__(self, args):
        self.process = 'fMRIWholeBrain'
        self.data_dir = f'{args.top_dir}/data/interim'
        Path(f'{self.data_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def load_files(self):
        files = glob(f'{self.data_dir}/fMRIDecoding/sub-*_reg-gaze-False_time-*_whole-brain-decoding.csv.gz')
        print(f'{len(files)=}')
        df = []
        for file in tqdm(files, total=len(files), desc='Loading whole brain results'):
            df.append(pd.read_csv(file))
        print('concatenating into dataframe')
        df = pd.concat(df)
        return df 

    def run(self):
        df = self.load_files()
        print('computing mean across EEG subjects')
        df = df.groupby(['subj_id', 'voxel_id', 'time']).mean(numeric_only=True).reset_index()
        for id, time_df in tqdm(df.groupby('subj_id'), total=len(df.subj_id.unique()), desc='Saving fMRI subj'):
            time_df.to_csv(f'{self.data_dir}/{self.process}/sub-{str(id).zfill(2)}_whole-brain-group.csv.gz', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', '-top', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis')
    args = parser.parse_args()
    fMRIWholeBrain(args).run()


if __name__ == '__main__':
    main()
