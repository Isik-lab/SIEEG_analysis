import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def summarize(file, out):
    arr = np.load(file)
    print(file)
    print(f'nan values: {np.sum(np.isnan(arr))} \n')

    plt.hist(arr)
    plt.savefig(out)

top_path = '/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim'
out_path = f'{top_path}/PlotLatency'
Path(out_path).mkdir(parents=True, exist_ok=True)

in_path = f'{top_path}/FeatureLatencyDist'
for in_file_base in ['agent_distance', 'communication', 'expanse', 'object']: 
    file = f'{in_path}/{in_file_base}.npy'
    out = f'{out_path}/{in_file_base}.png'
    summarize(file, out)

in_path = f'{top_path}/ROILatencyDist'
for roi in ['EVC', 'EBA', 'FFA', 'PPA', 'LOC', 'aSTS']: 
    for subj in range(1,5):
        file = f'{in_path}/sub-{subj}_roi-{roi}.npy'
        out = f'{out_path}/sub-{subj}_roi-{roi}.png'
        summarize(file, out)