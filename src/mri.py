import nibabel as nib
import numpy as np


def gen_mask(files, rel_mask):
    roi = []
    for f in files:
        roi_hemi = nib.load(f).get_fdata().astype('bool')
        roi.append(roi_hemi)
    roi_mask = np.sum(roi, axis=0).flatten()
    return np.all(np.vstack([rel_mask, roi_mask]), axis=0)