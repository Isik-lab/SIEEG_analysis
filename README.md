# SIEEG_analysis
 Analysis of social interaction EEG data

 ## Data

The raw and preprocessed data are available here: https://osf.io/2vnw4/

The analysis scripts consume processed data under `data/interim/`:

```
data/interim/
├── ReorganizeEEG/all_trials/sub-XX.parquet   # analysis-ready EEG (primary input)
├── ReorganizefMRI/                            # fMRI benchmark
│   ├── response_data.csv.gz                   #   fMRI betas (voxels × videos)
│   ├── metadata.csv                           #   target info (subj_id, roi_name, voxel_id)
│   └── stimulus_data.csv                      #   per-video split + behavior ratings

```

### Preprocessing

EEG was minimally preprocessed in MATLAB/FieldTrip (stimulus/photodiode
alignment, baseline correction, 0.1–60 Hz band, resampled to 400 Hz, 12.5 ms
smoothing; catch and false-alarm trials removed), producing the
`sub-XX_preproc.mat` files. The MATLAB preprocessing scripts are in
`scripts/eeg_preprocessing/`. To rebuild the analysis parquet from a `.mat`, use
`mat_to_parquet.py`.


## Analyses

- **EEG reliability** — `scripts/eeg_reliability.py`
- **ROI decoding** (EEG → fMRI ROI) — `scripts/fmri_regression.py` → `plot_roi_decoding.py`
- **Whole-brain decoding** — `scripts/fmri_regression.py --no-roi_mean` → `fmri_whole_brain.py`
- **Back-to-back regression** — `scripts/back_to_back.py` → `plot_back2back.py`

Subjects are 1–6 and 8–21 (no subject 7). The fMRI regressions expect a GPU.
