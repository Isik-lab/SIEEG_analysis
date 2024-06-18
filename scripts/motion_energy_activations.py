# /Applications/anaconda3/envs/opencv/bin/python

import argparse
import imageio
import os
import numpy as np
import pandas as pd
from pathlib import Path
import moten
from tqdm import tqdm


class MotionEnergyActivations():
    def __init__(self, args):
        self.process = 'MotionEnergyActivations'
        self.overwrite = args.overwrite
        self.vid_dir = args.vid_dir
        self.out_dir = args.out_dir
        self.stim_data = args.stim_data
        self.out_file = f'{self.out_dir}/motion_energy.npy'
        self.vdim = 500
        self.hdim = 500
        self.fps = 30
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def get_moten(self):
        df = pd.read_csv(self.stim_data)
        # Create a pyramid of spatio-temporal gabor filters
        pyramid = moten.get_default_pyramid(vhsize=(self.vdim, self.hdim), fps=self.fps)

        out_moten = []
        for video_name in tqdm(df.video_name, desc='Video progress',
                               position=0, leave=True):
            vid_obj = imageio.get_reader(f'{self.vid_dir}/{video_name}', 'ffmpeg')
            num_frames=vid_obj.count_frames()

            vid = []
            for i in range(int(num_frames)):
                vid.append(vid_obj.get_data(i).mean(axis=-1)) #Make gray scale
            vid = np.array(vid)
            moten_features = pyramid.project_stimulus(vid)
            out_moten.append(moten_features.mean(axis=0)) #average over frames and append to array
        return np.array(out_moten)

    def run(self):
        if not os.path.exists(self.out_file) or self.overwrite:
            out_moten = self.get_moten()
            np.save(self.out_file, out_moten)
        else:
            out_moten = np.load(self.out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--stim_data', type=str, help='path to the stimulus data',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/ReorganizefMRI/stimulus_data.csv')
    parser.add_argument('--vid_dir', '-v', type=str, help='video directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/raw/videos')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim/MotionEnergyActivations')
    args = parser.parse_args()
    MotionEnergyActivations(args).run()

if __name__ == '__main__':
    main()
