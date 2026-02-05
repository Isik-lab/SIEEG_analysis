import mne
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne.preprocessing import ICA
from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation
from functools import partial
from pyprep import NoisyChannels


# Functions in this file:
# - load_raw_data
# - find_events_and_create_initial_epochs
# - align_to_photodiode
# - perform_photodiode_alignment

# - detect_bad_channels
# - apply_filtering_and_referencing
# - clean_epochs_and_channels
# - apply_ica_blink_removal
# - finalize_epochs
# - plot_channel_timecourses

# ============================================================================
# Initial Processing Functions
# ============================================================================

def load_raw_data(vhdr_file):
    """
    Load BrainVision data and detect bad channels.
    
    Parameters
    ----------
    vhdr_file : str
        Path to BrainVision .vhdr file
        
    Returns
    -------
    raw : mne.io.Raw
        Loaded raw EEG data
    """
    print("Loading BrainVision data...")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
    
    # Set channel type for photodiode to misc to avoid coordinate warnings
    if 'photodiode' in raw.ch_names:
        raw.set_channel_types({'photodiode': 'misc'})

    print(f"Data loaded: {raw.info['nchan']} channels, {raw.n_times} samples, {raw.info['sfreq']} Hz")

    nc = NoisyChannels(raw)
    nc.find_bad_by_correlation()
    nc.find_bad_by_deviation()
    nc.find_bad_by_nan_flat()
    bads = nc.get_bads()
    raw.drop_channels(bads)
    
    print(f"{len(bads)} bad channels detected and removed: {bads}")
    return raw


def find_events_and_create_initial_epochs(raw, prestim_time, poststim_time, trials_to_remove, subj_path, rerun_initial):
    """
    Find events and create initial epochs.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    prestim_time : float
        Prestimulus period (seconds)
    poststim_time : float
        Poststimulus period for initial loading (seconds)
    trials_to_remove : list
        Trial indices to remove
    subj_path : str
        Path to save subject-specific files
    rerun_initial : bool
        Whether to rerun initial epochs creation
        
    Returns
    -------
    epochs : mne.Epochs
        Initial epochs
    stim_events : ndarray
        Stimulus events array
    stim_event_id : int
        Stimulus event ID
    """
    # Find events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Filter events for stimulus onset
    stim_event_keys = [key for key in event_id.keys() if 'Stimulus' in key or 'S  ' in key]
    
    if len(stim_event_keys) == 0:
        raise ValueError("No stimulus events found in data")
    
    # Use the first stimulus event type found
    stim_key = stim_event_keys[0]
    stim_event_id = event_id[stim_key]
    stim_events = events[events[:, 2] == stim_event_id]
    
    if len(stim_events) == 0:
        raise ValueError("No stimulus events found after filtering")
    
    # Remove predefined bad trials from events (experimental errors)
    if trials_to_remove is not None:
        keep_mask = np.ones(len(stim_events), dtype=bool)
        keep_mask[trials_to_remove] = False
        stim_events = stim_events[keep_mask]
    
    print(f"Found {len(stim_events)} stimulus events after removing bad trials")
    
    # Create epochs with initial time window
    initial_epochs_file = os.path.join(subj_path, 'initial_epochs-epo.fif') if subj_path else None
    if initial_epochs_file and os.path.exists(initial_epochs_file) and not rerun_initial:
        print("Loading existing initial epochs...")
        epochs = mne.read_epochs(initial_epochs_file, verbose=False)
        print(f"Loaded {len(epochs)} initial epochs")
    else:
        print("Creating initial epochs...")
        epochs = mne.Epochs(raw, stim_events, event_id={stim_key: stim_event_id},
                            tmin=-prestim_time, tmax=poststim_time,
                            baseline=None, preload=True, verbose=False)
        
        print(f"Created {len(epochs)} initial epochs")
        if initial_epochs_file:
            epochs.save(initial_epochs_file, overwrite=True)
            print("Saved initial epochs")

            epochs.average().detrend().plot_joint()
            plt.savefig(os.path.join(subj_path, 'initial_epochs_joint_plot.png'), dpi=150)
            plt.close()
    
    return epochs, stim_events, stim_event_id, stim_key


# ============================================================================
# Photodiode Alignment Functions
# ============================================================================

def align_to_photodiode(raw, onset_sample_number=None,
                        frames_per_second=None,
                        low_pass_filter=50, plot_file=None, 
                        search_window=(100, 350), 
                        acceptable_range=(150, 300)):
    """
    Realign trials to photodiode onset.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with photodiode channel
    down : bool
        If True, detect downward flanks; if False, detect upward flanks
    onset_sample_number : int
        Expected sample number of trigger onset (e.g., prestim * fs)
    frames_per_second : float
        Sampling rate
    low_pass_filter : float
        Low-pass filter frequency for photodiode signal
    search_window : tuple
        Tuple of (earliest_sample, latest_sample) to consider for trigger
    acceptable_range : tuple
        Tuple of (min_sample, max_sample) acceptable for trigger
        
    Returns
    -------
    offsets : list
        Offset values for each trial
    offset_times : list
        Offset times for each trial
    bad_trials_photo : list
        List of bad trial indices
    """
    
    # Get photodiode channel index
    photo_idx = raw.ch_names.index('photodiode')
    
    # Get sampling rate if not provided
    if frames_per_second is None:
        frames_per_second = raw.info['sfreq']
    
    # Create a copy and pick only the photodiode channel for analysis
    raw_photo = raw.copy().pick([photo_idx])
    _, raw_time = raw_photo[:, :]
    
    # Extract photodiode data within time window with buffer for filtering
    buffer_samples = int(0.1 * frames_per_second)  # 100ms buffer
    extract_start = max(0, search_window[0] - buffer_samples)
    extract_end = min(raw_photo.n_times, search_window[1] + buffer_samples)
    photo_data_full, _ = raw_photo[:, extract_start:extract_end]
    photo_data_full = photo_data_full.squeeze()
    
    # Normalize photodiode data (z-score)
    photo_mean = np.nanmean(photo_data_full)
    photo_std = np.nanstd(photo_data_full)
    photo_normalized = (photo_data_full - photo_mean) / photo_std
    
    # Low-pass filter the photodiode signal
    sos = butter(4, low_pass_filter, btype='low', 
                 fs=frames_per_second, output='sos')
    photo_lpf = sosfiltfilt(sos, photo_normalized)

    # Trim to original search window
    trim_start = search_window[0] - extract_start
    trim_end = search_window[1] - extract_start
    photo_data = photo_data_full[trim_start:trim_end]
    photo_lpf = photo_lpf[trim_start:trim_end]
    
    # Z-score the filtered signal
    photo_zscored = (photo_lpf - np.mean(photo_lpf)) / np.std(photo_lpf)
    
    # Find photodiode triggers (zero crossings)
    sign_changes = np.diff(np.sign(photo_zscored))
    trigger_idx = np.where(sign_changes != 0)[0]
    
    # Get first trigger occurrence
    if len(trigger_idx) > 0:
        photosmp = trigger_idx[0]
    else:
        photosmp = np.nan
    
    # Convert to absolute sample number in raw data
    photosmp_abs_idx = search_window[0] + photosmp if not np.isnan(photosmp) else np.nan
    offset = photosmp_abs_idx - onset_sample_number if not np.isnan(photosmp_abs_idx) else 0

    # Plot if requested and trigger is outside expected range
    if plot_file is not None and (np.isnan(photosmp_abs_idx) or
        photosmp_abs_idx < acceptable_range[0] or 
        photosmp_abs_idx > acceptable_range[1]):
        Path(os.path.dirname(plot_file)).mkdir(parents=True, exist_ok=True)
        _, ax = plt.subplots(2, 1, figsize=(5, 2.5))
        ax[0].plot(photo_data)
        ax[0].set_title('Raw Photodiode Signal')
        if not np.isnan(photosmp_abs_idx):
            ax[0].axvline(photosmp_abs_idx, color='g', linestyle='--', label=f'Trigger: {photosmp_abs_idx}')
        ax[1].plot(photo_zscored)
        ax[1].axhline(0, color='r', linestyle='--', label='Zero')
        if not np.isnan(photosmp_abs_idx):
            ax[1].axvline(photosmp_abs_idx, color='g', linestyle='--', label=f'Trigger: {photosmp_abs_idx}')
        ax[1].set_title(f'Z-Scored Low-Pass Filtered Signal (Trigger at {photosmp_abs_idx})')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
    
    # Identify bad trials
    if (np.isnan(photosmp_abs_idx) or 
        photosmp_abs_idx < acceptable_range[0] or 
        photosmp_abs_idx > acceptable_range[1]):
        bad_trial = True
    else:
        bad_trial = False
    
    return offset, bad_trial


def perform_photodiode_alignment(epochs, raw, prestim_time, aligned_poststim_time, 
                                photodiode_lpf, earliest_time, latest_time, subj_path, debug, 
                                stim_events, stim_key, stim_event_id):
    """
    Perform photodiode alignment on epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Initial epochs
    raw : mne.io.Raw
        Original raw data
    prestim_time : float
        Prestimulus period (seconds)
    aligned_poststim_time : float
        Poststimulus period after photodiode alignment (seconds)
    photodiode_lpf : float
        Low-pass filter for photodiode
    earliest_time : float
        Time to exclude around expected trigger
    latest_time : float
        Latest acceptable time for trigger
    subj_path : str
        Path to save subject-specific files
    debug : bool
        If True, process only first 10 trials for debugging
    stim_events : ndarray
        Stimulus events array
    stim_key : str
        Stimulus event key
    stim_event_id : int
        Stimulus event ID
        
    Returns
    -------
    epochs_final : mne.Epochs
        Photodiode-aligned epochs
    """
    print("Starting photodiode alignment...")
    frames_per_second = raw.info['sfreq']
    search_window = (int((prestim_time - 0.1) * frames_per_second), 
                     int((prestim_time + 0.2) * frames_per_second))
    acceptable_range = (int((prestim_time - earliest_time) * frames_per_second), 
                     int((prestim_time + latest_time) * frames_per_second))
    
    offsets, bad_trials_photo = [], []
    
    # Process each trial individually for photodiode alignment
    for trial_idx in tqdm(range(len(epochs)), desc='Photodiode alignment',
                            leave=False, position=1):
        trial_data = epochs.get_data()[trial_idx]
        info = epochs.info
        raw_trial = mne.io.RawArray(trial_data, info, verbose=False)
        offset_index, bad_trial = align_to_photodiode(
            raw_trial,
            onset_sample_number=int(prestim_time * frames_per_second),
            search_window=search_window,
            acceptable_range=acceptable_range,
            frames_per_second=frames_per_second,
            low_pass_filter=photodiode_lpf,
            plot_file=os.path.join(subj_path, 'photodiode_plots', f'photodiode_trial{trial_idx+1}.png') if subj_path else None
        )
        offsets.append(offset_index)

        if bad_trial:
            bad_trials_photo.append(trial_idx)

        if trial_idx == 10 and debug:
            break
    
    # Apply offsets to event sample numbers
    if debug:
        print("Debug mode: processed only first 10 trials for photodiode alignment")
        events_shifted = stim_events[:11].copy()
    else:
        events_shifted = stim_events.copy()
    events_shifted[:, 0] = events_shifted[:, 0] + np.array(offsets).astype(int)

    # Create final epochs with shifted events using original raw data
    epochs_final = mne.Epochs(raw, events_shifted, event_id={stim_key: stim_event_id},
                              tmin=-prestim_time, tmax=aligned_poststim_time,
                              baseline=None, preload=True, verbose=False)
    
    epochs_final.drop(bad_trials_photo, reason='Bad photodiode signal')

    epochs_final.drop_channels('photodiode')  # Remove photodiode channel after alignment
    
    print(f"Photodiode alignment complete: {len(epochs_final)} epochs after alignment (removed {len(bad_trials_photo)} bad trials)")
    return epochs_final


# ============================================================================
# Low/High Pass Filter
# ============================================================================

def apply_filtering_and_referencing(epochs_final, subj_path, rerun_filtered):
    """
    Apply filtering and referencing to epochs.
    
    Parameters
    ----------
    epochs_final : mne.Epochs
        Epochs to filter and reference
    subj_path : str
        Path to save subject-specific files
    rerun_filtered : bool
        Whether to rerun filtering
        
    Returns
    -------
    epochs_final : mne.Epochs
        Filtered and referenced epochs
    """
    filtered_epochs_file = os.path.join(subj_path, 'filtered_epochs-epo.fif') if subj_path else None
    
    if not (filtered_epochs_file and os.path.exists(filtered_epochs_file) and not rerun_filtered):
        print("Applying filtering and referencing...")
        epochs_final.filter(l_freq=1, h_freq=None, method='iir', 
                           iir_params=dict(order=4, ftype='butter'), verbose=False) # High-pass filter (1 Hz)
        epochs_final.filter(l_freq=None, h_freq=60, method='iir',
                           iir_params=dict(order=4, ftype='butter'), verbose=False) # Low-pass filter (60 Hz)
        mne.set_eeg_reference(epochs_final, ref_channels='average', verbose=False)
        
        print(f"Filtering complete: {len(epochs_final)} epochs")
        if filtered_epochs_file:
            epochs_final.save(filtered_epochs_file, overwrite=True)
            print("Saved filtered epochs")
    
    return epochs_final

# ============================================================================
# Detect Bad Trials
# ============================================================================


def clean_epochs(epochs, subj_path, rerun_epoch_cleaning):
    """
    Reject bad epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to clean
    subj_path : str
        Path to save subject-specific files
    rerun_epoch_cleaning : bool
        Whether to rerun cleaning
        
    Returns
    -------
    epochs_clean : mne.Epochs
        Cleaned epochs
    """
    print('Starting epoch cleaning...')
    cleaned_epochs_file = os.path.join(subj_path, 'cleaned_epochs-epo.fif') if subj_path else None
    
    if not (cleaned_epochs_file and os.path.exists(cleaned_epochs_file) and not rerun_epoch_cleaning):        
        from autoreject import AutoReject 
        # Copy epochs before rejection to plot bad ones
        epochs_clean = epochs.copy()
        num_epochs_before = len(epochs)

        # Median Absolute Deviation Filtering to identify bad epochs
        ar = AutoReject(verbose=True)
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)
        # Drop the bad epochs
        epochs_clean.drop(reject_log.bad_epochs, reason='AutoReject') 
        if subj_path:
            reject_log.plot('horizontal')
            plt.savefig(os.path.join(subj_path, 'autoreject_log.png'), dpi=150)
            plt.close()
        
        num_epochs_after = len(epochs_clean)
        num_removed = num_epochs_before - num_epochs_after
        
        # Get indices of bad epochs
        bad_indices = [i for i, log in enumerate(epochs.drop_log) if log]
        
        print(f"Epoch rejection: {num_epochs_before} -> {num_epochs_after} epochs ({num_removed} removed)")
        
        # Plot summary of rejected epochs
        if subj_path:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Bar plot of epochs before/after
            ax1.bar(['Before Rejection', 'After Rejection'], [num_epochs_before, num_epochs_after], 
                    color=['red', 'green'])
            ax1.set_ylabel('Number of Epochs')
            ax1.set_title(f'Epoch Rejection Summary\n{num_removed} epochs removed')
            ax1.grid(True, alpha=0.3)
            
            # Time course of bad epochs (average across channels)
            if bad_indices:
                bad_epochs = epochs[bad_indices]
                bad_data = bad_epochs.get_data()  # (n_bad_epochs, n_channels, n_times)
                bad_avg_channels = np.mean(bad_data, axis=1)  # (n_bad_epochs, n_times)
                bad_avg_channels = bad_avg_channels - bad_avg_channels.mean(axis=1, keepdims=True) # Demean to plot on one axis
                
                # Plot up to 10 bad epochs to avoid clutter
                max_plot = min(10, len(bad_indices))
                for i in range(max_plot):
                    ax2.plot(bad_epochs.times, bad_avg_channels[i], 
                            label=f'Epoch {bad_indices[i]}', alpha=0.7)
                
                if len(bad_indices) > max_plot:
                    ax2.text(0.02, 0.98, f'Showing first {max_plot} of {len(bad_indices)} bad epochs', 
                            transform=ax2.transAxes, fontsize=8, verticalalignment='top')
                
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude (V)')
                ax2.set_title(f'Channel-averaged time courses of rejected epochs')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No epochs rejected', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Rejected Epochs')
            
            plt.tight_layout()
            plt.savefig(os.path.join(subj_path, 'epoch_rejection_summary.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    return epochs_clean

# ============================================================================
# ICA to Remove Blinks
# ============================================================================

def apply_ica_blink_removal(epochs, subj_path, rerun_ica):
    """
    Apply ICA for blink removal.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to apply ICA to
    subj_path : str
        Path to save subject-specific files
    rerun_ica : bool
        Whether to rerun ICA
        
    Returns
    -------
    epochs_final : mne.Epochs
        ICA-cleaned epochs
    """
    ica_epochs_file = os.path.join(subj_path, 'ica_epochs-epo.fif') if subj_path else None
    
    if not (ica_epochs_file and os.path.exists(ica_epochs_file) and not rerun_ica):
        print("Starting ICA artifact removal...")
        ica = ICA(n_components=20, method='picard', 
                  random_state=42, verbose=False,
                  fit_params=dict(ortho=True, extended=True) )
        ica.fit(epochs, verbose=False)

        # Find the first available channel from the preferred list for EOG detection
        preferred_channels = ['Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8']
        ch_name = None
        for ch in preferred_channels:
            if ch in epochs.ch_names:
                ch_name = ch
                break
        # Fallback to first channel if none of the preferred channels exist
        if ch_name is None:
            ch_name = epochs.ch_names[0]
        eog_indices, _ = ica.find_bads_eog(epochs, ch_name=ch_name)
        ica.exclude = eog_indices
        
        print(f"ICA: Excluding {len(ica.exclude)} components out of {ica.n_components_}")
        
        # Plot ICA components topomaps
        _ = ica.plot_components(inst=epochs, show=False)
        plt.suptitle(f'ICA Components (Excluded: {ica.exclude})', fontsize=16)
        plt.savefig(os.path.join(subj_path, 'ica_components_topomaps.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot ICA sources time courses
        _ = ica.plot_sources(epochs, show=False)
        plt.suptitle(f'ICA Sources Time Courses (Excluded: {ica.exclude})', fontsize=16)
        plt.savefig(os.path.join(subj_path, 'ica_sources_timecourses.png'), dpi=150, bbox_inches='tight')
        plt.close()

        ica.plot_overlay(epochs.average(), exclude=ica.exclude, show=False)
        plt.suptitle(f'ICA Overlay (Excluded: {ica.exclude})', fontsize=16)
        plt.savefig(os.path.join(subj_path, 'ica_overlay.png'), dpi=150, bbox_inches='tight')
        plt.close()
            
        # Apply ICA
        epochs_ica = ica.apply(epochs, verbose=False)
        
        print(f"ICA complete: {len(epochs_ica)} epochs")
        if ica_epochs_file:
            epochs_ica.save(ica_epochs_file, overwrite=True)
            print("Saved ICA epochs")
    
    return epochs_ica


# ============================================================================
# Remove frontal channels and baseline correction
# ============================================================================


def remove_frontal_channels(epochs, prestim_time, subj_path):
    """
    Finalize epochs by removing frontal channels and applying baseline correction.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to finalize
    prestim_time : float
        Prestimulus period for baseline correction
    subj_path : str
        Path to save subject-specific files
        
    Returns
    -------
    epochs : mne.Epochs
        Corrected epochs
    """
    # Remove frontal electrodes to reduce eye movement artifacts
    frontal_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']
    existing_frontal = [ch for ch in frontal_channels if ch in epochs.ch_names]
    if existing_frontal:
        print(f"Removing frontal channels: {existing_frontal}")
        epochs.drop_channels(existing_frontal)

    
    # Save final epochs
    if subj_path:
        final_epochs_file = os.path.join(subj_path, 'final_epochs-epo.fif')
        epochs.save(final_epochs_file, overwrite=True)
        print("Saved final epochs")
    
    return epochs