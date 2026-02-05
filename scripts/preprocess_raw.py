import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

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
    
    # Apply ICA for blink removal
    epochs_final = ep.apply_ica_blink_removal(epochs_final, subj_path, rerun_ica)

    # Finalize epochs
    epochs_final = ep.finalize_epochs(epochs_final, prestim_time, subj_path)
    
    print(f"Preprocessing pipeline complete: {len(epochs_final)} final epochs")
    return epochs_final

# ============================================================================
# MAIN EXECUTION
# ============================================================================

class preprocess_raw:
    def __init__(self, args):
        self.sid = f'{args.sid:02d}'
        self.data_dir = args.data_dir
        self.prestim_time = args.prestim_time
        self.poststim_time = args.poststim_time
        self.aligned_poststim_time = args.aligned_poststim_time
        self.rerun_initial = args.rerun_initial
        self.rerun_aligned = args.rerun_aligned
        self.rerun_filtered = args.rerun_filtered
        self.rerun_cleaned = args.rerun_cleaned
        self.rerun_ica = args.rerun_ica
        self.input_path = os.path.join(self.data_dir, 'data', 'raw', 'SIdyads_EEG')
        self.output_path = os.path.join(self.data_dir, 'data', 'interim', 'PreprocessRaw', f'sub-{self.sid}')
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def run(self):
        # Construct file paths
        subj_dir = Path(self.input_path) / f'sub-{self.sid}'
        vhdr_file = subj_dir / f'sub-{self.sid}.vhdr'

        if not vhdr_file.exists():
            raise FileNotFoundError(f"BrainVision file not found: {vhdr_file}")

        # Define trials to remove
        if self.sid == '01':
            trials_to_remove = [880]  # Python is 0-indexed
        else:
            trials_to_remove = None

        try:
            # Preprocess
            epochs_preprocessed = preprocess_eeg_pipeline(
                str(vhdr_file),  # Pass full path with .vhdr extension
                prestim_time=self.prestim_time,
                poststim_time=self.poststim_time,
                aligned_poststim_time=self.aligned_poststim_time,
                trials_to_remove=trials_to_remove,
                subj_path=self.output_path,
                rerun_initial=self.rerun_initial,
                rerun_aligned=self.rerun_aligned,
                rerun_filtered=self.rerun_filtered,
                rerun_cleaned=self.rerun_cleaned,
                rerun_ica=self.rerun_ica
            )

            if epochs_preprocessed is None:
                print(f"Preprocessing failed for sub-{self.sid}")
                return

        except Exception as e:
            print(f"Error processing subject {self.sid}: {e}")
            import traceback
            traceback.print_exc()
            return

        # Compute evoked (grand average for this subject)
        evoked = epochs_preprocessed.average()
        evoked.comment = f'sub-{self.sid}'

        # Save as epochs
        epochs_file = os.path.join(self.output_path, f'sub-{self.sid}_preproc-epo.fif')
        epochs_preprocessed.save(epochs_file, overwrite=True)

        print(f"✓ Processing complete for sub-{self.sid}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=int, default=2, help='Subject ID')
    parser.add_argument('--data_dir', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/SIEEG_analysis',
                         help='Data directory')
    parser.add_argument('--prestim_time', type=float, default=0.2,
                         help='Prestimulus period (seconds)')
    parser.add_argument('--poststim_time', type=float, default=1.25,
                         help='Poststimulus period for initial loading (seconds)')
    parser.add_argument('--aligned_poststim_time', type=float, default=1.0,
                         help='Poststimulus period after photodiode alignment (seconds)')
    parser.add_argument('--rerun_initial', action='store_false',
                         help='Rerun initial epochs and all subsequent steps')
    parser.add_argument('--rerun_aligned', action='store_false',
                         help='Rerun alignment and all subsequent steps')
    parser.add_argument('--rerun_filtered', action='store_false',
                         help='Rerun filtering and all subsequent steps')
    parser.add_argument('--rerun_cleaned', action='store_false',
                         help='Rerun cleaning and all subsequent steps')
    parser.add_argument('--rerun_ica', action='store_false',
                         help='Rerun ICA and all subsequent steps')
    args = parser.parse_args()
    preprocess_raw(args).run()


if __name__ == '__main__':
    main()