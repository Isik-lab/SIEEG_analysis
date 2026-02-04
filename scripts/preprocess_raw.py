import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eeg import preprocess as ep

def preprocess_eeg_pipeline(vhdr_file, prestim_time=0.2, 
                            poststim_time=1.25, aligned_poststim_time=1.0,
                            photodiode_lpf=50, 
                            earliest_time=0.05, latest_time=0.1, 
                            trials_to_remove=None,
                            subj_path=None, debug=False, 
                            **kwargs):
    """
    Complete EEG preprocessing pipeline.
    
    Parameters
    ----------
    vhdr_file : str
        Path to BrainVision .vhdr file (MUST include .vhdr extension)
    prestim_time : float
        Prestimulus period (seconds)
    poststim_time : float
        Poststimulus period for initial loading (seconds)
    aligned_poststim_time : float
        Poststimulus period after photodiode alignment (seconds)
    photodiode_lpf : float
        Low-pass filter for photodiode
    earliest_time : float
        Time to exclude around expected trigger
    plotting : bool
        Whether to plot intermediate steps
    trials_to_remove : list
        Trial indices to remove
    debug : bool
        If True, process only first 10 trials for debugging
    subj_path : str
        Path to save subject-specific plots
    **kwargs : dict
        Keyword arguments for controlling rerun behavior:
        - rerun_initial : bool, default False
            If False, load initial_epochs-epo.fif if exists. If True, rerun initial epochs and all subsequent steps.
        - rerun_aligned : bool, default False
            If False, load aligned_epochs-epo.fif if exists. If True, rerun alignment and all subsequent steps.
        - rerun_filtered : bool, default False
            If False, load filtered_epochs-epo.fif if exists. If True, rerun filtering and all subsequent steps.
        - rerun_cleaned : bool, default False
            If False, load cleaned_epochs-epo.fif if exists. If True, rerun cleaning and all subsequent steps.
        - rerun_ica : bool, default False
            If False, load ica_epochs-epo.fif if exists. If True, rerun ICA and all subsequent steps.

    Returns
    -------
    raw_preprocessed : mne.io.Raw or mne.Epochs
        Preprocessed EEG data
    """
    
    print(f"Starting EEG preprocessing pipeline for: {vhdr_file}")
    
    # Get rerun flags from kwargs
    rerun_initial = kwargs.get('rerun_initial', False)
    rerun_aligned = kwargs.get('rerun_aligned', False) or rerun_initial
    rerun_filtered = kwargs.get('rerun_filtered', False) or rerun_aligned
    rerun_cleaned = kwargs.get('rerun_cleaned', False) or rerun_filtered
    rerun_ica = kwargs.get('rerun_ica', False) or rerun_cleaned
    
    print(f"Rerun flags - Initial: {rerun_initial}, Aligned: {rerun_aligned}, Filtered: {rerun_filtered}, Cleaned: {rerun_cleaned}, ICA: {rerun_ica}")
    
    # Verify file exists
    vhdr_path = Path(vhdr_file)
    if not vhdr_path.exists():
        raise FileNotFoundError(f"BrainVision file not found: {vhdr_file}")
    
    # Check if we can skip loading raw data by loading existing later-stage epochs
    ica_epochs_file = os.path.join(subj_path, 'ica_epochs-epo.fif') if subj_path else None
    if ica_epochs_file and os.path.exists(ica_epochs_file) and not rerun_ica:
        print("Loading existing ICA epochs...")
        epochs_final = mne.read_epochs(ica_epochs_file, verbose=False)
        print(f"Loaded {len(epochs_final)} ICA epochs")
    else:
        cleaned_epochs_file = os.path.join(subj_path, 'cleaned_epochs-epo.fif') if subj_path else None
        if cleaned_epochs_file and os.path.exists(cleaned_epochs_file) and not rerun_cleaned:
            print("Loading existing cleaned epochs...")
            epochs_final = mne.read_epochs(cleaned_epochs_file, verbose=False)
            print(f"Loaded {len(epochs_final)} cleaned epochs")
        else:
            filtered_epochs_file = os.path.join(subj_path, 'filtered_epochs-epo.fif') if subj_path else None
            if filtered_epochs_file and os.path.exists(filtered_epochs_file) and not rerun_filtered:
                print("Loading existing filtered epochs...")
                epochs_final = mne.read_epochs(filtered_epochs_file, verbose=False)
                print(f"Loaded {len(epochs_final)} filtered epochs")
            else:
                aligned_epochs_file = os.path.join(subj_path, 'aligned_epochs-epo.fif') if subj_path else None
                if aligned_epochs_file and os.path.exists(aligned_epochs_file) and not rerun_aligned:
                    print("Loading existing aligned epochs...")
                    epochs_final = mne.read_epochs(aligned_epochs_file, verbose=False)
                    print(f"Loaded {len(epochs_final)} aligned epochs")
                else:
                    # Need to load raw data and process from beginning
                    raw = ep.load_raw_data(vhdr_file)
                    
                    # Find events and create initial epochs
                    epochs, stim_events, stim_event_id, stim_key = ep.find_events_and_create_initial_epochs(
                        raw, prestim_time, poststim_time, trials_to_remove, subj_path, rerun_initial)
                    
                    # Perform photodiode alignment
                    epochs_final = ep.perform_photodiode_alignment(
                        epochs, raw, prestim_time, aligned_poststim_time,
                        photodiode_lpf, earliest_time, latest_time, subj_path, debug,
                        stim_events, stim_key, stim_event_id)
                    
                    # Save aligned epochs
                    if aligned_epochs_file:
                        epochs_final.save(aligned_epochs_file, overwrite=True)
                        print("Saved aligned epochs")
            
            # Apply filtering and referencing
            epochs_final = ep.apply_filtering_and_referencing(epochs_final, subj_path, rerun_filtered)
        
        # Clean epochs and interpolate bad channels
        epochs_final = ep.clean_epochs_and_channels(epochs_final, subj_path, rerun_cleaned)
    
    # Apply ICA for artifact removal
    epochs_final = ep.apply_ica_artifact_removal(epochs_final, subj_path, rerun_ica)

    # Finalize epochs
    epochs_final = ep.finalize_epochs(epochs_final, prestim_time, subj_path)
    
    print(f"Preprocessing pipeline complete: {len(epochs_final)} final epochs")
    return epochs_final


def process_subjects(input_path, output_path, s_list=None, 
                     prestim_time=0.2, poststim_time=1.25,
                     aligned_poststim_time=1.0, **kwargs):
    """
    Process all subjects in the subject list.

    Parameters
    ----------
    input_path : str
        Path to raw EEG data directory
    output_path : str
        Path to save preprocessed data
    s_list : list
        List of subject IDs (e.g., ['01', '02', ...])
    prestim_time : float
        Prestimulus period
    poststim_time : float
        Poststimulus period
    aligned_poststim_time : float
        Aligned poststimulus period
    **kwargs : dict
        Keyword arguments passed to preprocess_eeg_pipeline for rerun control

    Returns
    -------
    all_evokeds : list
        List of evoked objects for all subjects
    """
    if s_list is None:
        s_list = [f'{i:02d}' for i in range(1, 22) if i != 7]  # Exclude sub-07, up to 21

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_evokeds = []

    iterator = tqdm(s_list, desc='Processing subjects',
                    leave=True, position=0, total=len(s_list))
    for subj_id in iterator:

        # Construct file paths correctly
        subj_dir = input_path / f'sub-{subj_id}'
        vhdr_file = subj_dir / f'sub-{subj_id}.vhdr'

        if not vhdr_file.exists():
            print(f"✗ File not found: {vhdr_file}")
            continue

        # Define trials to remove
        if subj_id == '01':
            trials_to_remove = [880]  # Python is 0-indexed
        else:
            trials_to_remove = None

        # Create per-subject plot path
        subj_path = os.path.join(output_path, f'sub-{subj_id}')
        Path(subj_path).mkdir(parents=True, exist_ok=True)

        try:
            # Preprocess
            epochs_preprocessed = preprocess_eeg_pipeline(
                str(vhdr_file),  # Pass full path with .vhdr extension
                prestim_time=prestim_time,
                poststim_time=poststim_time,
                aligned_poststim_time=aligned_poststim_time,
                trials_to_remove=trials_to_remove,
                subj_path=subj_path,
                **kwargs
            )

            if epochs_preprocessed is None:
                print(f"Skipping sub-{subj_id} due to preprocessing error")
                continue

        except Exception as e:
            print(f"Error processing subject {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Compute evoked (grand average)
        evoked = epochs_preprocessed.average()
        evoked.comment = f'sub-{subj_id}'
        all_evokeds.append(evoked)

        # Save as epochs
        epochs_file = output_path / f'sub-{subj_id}' / f'sub-{subj_id}_preproc-epo.fif'
        epochs_preprocessed.save(str(epochs_file), overwrite=True)

    return all_evokeds


def plot_grand_average_topomaps(grand_avg, output_path, time_step=0.1):
    """
    Create topomap plots for grand average at specified time steps.
    
    Parameters
    ----------
    grand_avg : mne.Evoked
        Grand average evoked object
    output_path : str
        Path to save figure
    time_step : float
        Time step for topomaps
    """
    times = np.arange(grand_avg.tmin, grand_avg.tmax, time_step)
    fig = grand_avg.plot_topomap(times=times, show=False)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Topomap plot saved to {output_path}")
    plt.close()


def plot_erp_trace(grand_avg, output_path, layout='standard_1020'):
    """
    Create ERP trace plot.
    
    Parameters
    ----------
    grand_avg : mne.Evoked
        Grand average evoked object
    output_path : str
        Path to save figure
    layout : str
        MNE layout name
    """
    
    fig = grand_avg.plot(layout=layout, gfp=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ERP trace plot saved to {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    top_path = '/orcd/data/ngk/001/users/emaliem/SIEEG_analysis'
    # Define paths
    input_path = os.path.join(top_path, 'data', 'raw', 'SIdyads_EEG')
    output_path = os.path.join(top_path, 'data', 'interim', 'PreprocessRaw')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Subject list
    s_list = [f'{int(s):02d}' for s in ['01', '02', '03', '04', '05', '06', '08', '09', '10', '11', 
              '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']]
    
    # Parameters
    prestim_time = 0.2
    raw_poststim_time = 1.25
    aligned_poststim_time = 1.0
    
    # Process all subjects
    print(f"Processing {len(s_list)} subjects...")
    all_evokeds = process_subjects(
        input_path, output_path, s_list,
        prestim_time=prestim_time,
        poststim_time=raw_poststim_time,
        aligned_poststim_time=aligned_poststim_time
    )    
    if len(all_evokeds) == 0:
        print("No subjects were successfully processed!")
        exit()
    
    print(f"\n{'='*60}")
    print(f"Computing grand average from {len(all_evokeds)} subjects...")
    print(f"{'='*60}")
    
    # Compute grand average
    grand_avg = mne.grand_average(all_evokeds)
    topo_file = output_path / 'topoplot.png'
    plot_grand_average_topomaps(grand_avg, str(topo_file), time_step=0.1)
    trace_file = output_path / 'traceplot.png'
    plot_erp_trace(grand_avg, str(trace_file))
    print("\n✓ All processing complete!")