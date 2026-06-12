#!/usr/bin/env python3
"""
Convert a FieldTrip v7 .mat preprocessed EEG file into the reorganized
analysis parquet (input to eeg_reliability.py, the regressions, etc.).

Assumes the standard project layout under a data/ directory:
    data/interim/SIdyads_EEG/sub-XX/sub-XX_preproc.mat   (input EEG)
    data/raw/SIdyads_trials/sub-XX/timingfiles/*.csv     (trial info)
    data/raw/annotations/test.csv                        (train/test split)
and writes:
    data/interim/ReorganizeEEG/all_trials/sub-XX.parquet

Assumes the .mat trials and timing rows are already 1:1 and in the same order.
Resamples to 400 Hz, 5-sample (12.5 ms) boxcar smooth, keeps real trials with
no false-alarm response, tags even/odd repetitions, writes long-format parquet:
    trial, channel, time, time_ind, signal, repitition, even,
    video_name, stimulus_set

Usage (from the project root, where data/ lives):
    python3 mat_to_parquet.py --sid 1
    python3 mat_to_parquet.py --sid 1 --data_dir /some/other/data

Requires: scipy numpy pandas pyarrow
"""
import os
import argparse
from glob import glob
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import convolve1d
from scipy.signal import resample_poly

RESAMPLE_HZ = 400      # methods: resample to 400 Hz
SMOOTH = 5             # methods: 5 consecutive samples = 12.5 ms at 400 Hz


def load_eeg(mat):
    meta = loadmat(mat, variable_names=["label", "fsample", "time"],
                   squeeze_me=True, struct_as_record=False)
    labels = np.array([str(x) for x in np.atleast_1d(meta["label"])])
    sfreq = float(np.atleast_1d(meta["fsample"]).ravel()[0])
    t0 = np.atleast_1d(meta["time"])
    tmin = float(np.asarray(t0[0] if t0.dtype == object else t0).ravel()[0])

    trials = loadmat(mat, variable_names=["trial"],
                     squeeze_me=True, struct_as_record=False)["trial"]
    data = np.stack([np.asarray(t, np.float32) for t in trials])  # (trials, ch, time)

    r = Fraction(RESAMPLE_HZ, int(sfreq)).limit_denominator(1000)
    data = resample_poly(data, r.numerator, r.denominator, axis=-1).astype(np.float32)
    data = convolve1d(data, np.ones(SMOOTH) / SMOOTH, axis=-1).astype(np.float32)

    times = (tmin + np.arange(data.shape[-1]) / RESAMPLE_HZ) * 1000.0  # ms
    return labels, data, times


def load_trials(timing_dir, test_csv):
    files = sorted(glob(os.path.join(timing_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No timing CSVs found in {timing_dir}")
    trials = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    test_videos = pd.read_csv(test_csv)["video_name"].tolist()
    trials["stimulus_set"] = np.where(trials.video_name.isin(test_videos),
                                      "test", "train")
    return trials


def main():
    ap = argparse.ArgumentParser(description="FieldTrip v7 .mat -> analysis parquet")
    ap.add_argument("--sid", "-s", required=True, help="subject number, e.g. 1")
    ap.add_argument("--data_dir", default="data",
                    help="data/ folder holding raw/ and interim/ (default: ./data)")
    a = ap.parse_args()

    sid = f"sub-{int(a.sid):02d}"
    dd = a.data_dir
    mat = f"{dd}/interim/SIdyads_EEG/{sid}/{sid}_preproc.mat"
    timing_dir = f"{dd}/raw/SIdyads_trials/{sid}/timingfiles"
    test_csv = f"{dd}/raw/annotations/test.csv"
    out = f"{dd}/interim/ReorganizeEEG/all_trials/{sid}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    labels, data, times = load_eeg(mat)
    trials = load_trials(timing_dir, test_csv)
    if len(trials) != data.shape[0]:
        raise ValueError(f"{sid}: .mat has {data.shape[0]} trials but timing has "
                         f"{len(trials)} rows; they must be 1:1.")
    trials.insert(0, "trial", np.arange(len(trials)))  # original trial index

    # keep real trials (condition truthy) with no false-alarm response
    keep = (trials["condition"].map(bool) & ~trials["response"].map(bool)).to_numpy()
    data = data[keep]
    trials = trials.loc[keep].reset_index(drop=True)
    trials["repitition"] = trials.groupby("video_name").cumcount()
    trials["even"] = trials["repitition"] % 2 == 0

    n_tr, n_ch, n_t = data.shape
    per = n_ch * n_t  # C-order flatten of (trial, ch, time): time fastest, then ch
    df = pd.DataFrame({
        "trial":        np.repeat(trials["trial"].to_numpy(), per),
        "channel":      np.tile(np.repeat(labels, n_t), n_tr),
        "time":         np.tile(times, n_tr * n_ch),
        "time_ind":     np.tile(np.arange(n_t), n_tr * n_ch),
        "signal":       data.reshape(-1),
        "repitition":   np.repeat(trials["repitition"].to_numpy(), per),
        "even":         np.repeat(trials["even"].to_numpy(), per),
        "video_name":   np.repeat(trials["video_name"].to_numpy(), per),
        "stimulus_set": np.repeat(trials["stimulus_set"].to_numpy(), per),
    })
    df.to_parquet(out, index=False)
    print(f"Wrote {out}: {len(df):,} rows, {df.video_name.nunique()} videos, "
          f"{n_t} timepoints, kept {n_tr}/{len(keep)} trials")


if __name__ == "__main__":
    main()
