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

def load_existing_epochs(subj_path, kwargs):
    """Load the most advanced existing epochs file that doesn't need rerunning."""
    cleaned_file = os.path.join(subj_path, 'cleaned_epochs-epo.fif')
    if os.path.exists(cleaned_file) and not kwargs.get('rerun_epoch_cleaning', False):
        print("Loading existing cleaned epochs...")
        epochs = mne.read_epochs(cleaned_file, verbose=False)
        print(f"Loaded {len(epochs)} cleaned epochs")
        return epochs, 'cleaned'
    
    frontal_file = os.path.join(subj_path, 'frontal_drop_epochs-epo.fif')
    if os.path.exists(frontal_file) and not kwargs.get('rerun_frontal_drop', False):
        print("Loading existing frontal drop epochs...")
        epochs = mne.read_epochs(frontal_file, verbose=False)
        print(f"Loaded {len(epochs)} frontal drop epochs")
        return epochs, 'frontal_drop'
    
    ica_file = os.path.join(subj_path, 'ica_epochs-epo.fif')
    if os.path.exists(ica_file) and not kwargs.get('rerun_ica', False):
        print("Loading existing ICA epochs...")
        epochs = mne.read_epochs(ica_file, verbose=False)
        print(f"Loaded {len(epochs)} ICA epochs")
        return epochs, 'ica'
    
    filtered_file = os.path.join(subj_path, 'filtered_epochs-epo.fif')
    if os.path.exists(filtered_file) and not kwargs.get('rerun_filtered', False):
        print("Loading existing filtered epochs...")
        epochs = mne.read_epochs(filtered_file, verbose=False)
        print(f"Loaded {len(epochs)} filtered epochs")
        return epochs, 'filtered'
    
    aligned_file = os.path.join(subj_path, 'aligned_epochs-epo.fif')
    if os.path.exists(aligned_file) and not kwargs.get('rerun_aligned', False):
        print("Loading existing aligned epochs...")
        epochs = mne.read_epochs(aligned_file, verbose=False)
        print(f"Loaded {len(epochs)} aligned epochs")
        return epochs, 'aligned'
    
    return None, None

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
        - rerun_ica : bool, default False
            If False, load ica_epochs-epo.fif if exists. If True, rerun ICA and all subsequent steps.
        - rerun_frontal_drop : bool, default False
            If True, rerun frontal channel removal and all subsequent steps.
        - rerun_epoch_cleaning : bool, default False
            If False, load cleaned_epochs-epo.fif if exists. If True, rerun cleaning and all subsequent steps.

    Returns
    -------
    raw_preprocessed : mne.io.Raw or mne.Epochs
        Preprocessed EEG data
    """
    
    # Get rerun flags from kwargs
    rerun_initial = kwargs.get('rerun_initial', False)
    rerun_aligned = kwargs.get('rerun_aligned', False) or rerun_initial
    rerun_filtered = kwargs.get('rerun_filtered', False) or rerun_aligned
    rerun_ica = kwargs.get('rerun_ica', False) or rerun_filtered
    rerun_frontal_drop = kwargs.get('rerun_frontal_drop', False) or rerun_ica
    rerun_epoch_cleaning = kwargs.get('rerun_epoch_cleaning', False) or rerun_frontal_drop
    
    print(f"Rerun flags - Initial: {rerun_initial}, Aligned: {rerun_aligned}, Filtered: {rerun_filtered}, ICA: {rerun_ica}, Frontal Drop: {rerun_frontal_drop}, Cleaned: {rerun_epoch_cleaning}")
    
    # Verify file exists
    vhdr_path = Path(vhdr_file)
    if not vhdr_path.exists():
        raise FileNotFoundError(f"BrainVision file not found: {vhdr_file}")
    
    # Try to load existing epochs from the most advanced stage
    epochs_final, loaded_stage = load_existing_epochs(subj_path, kwargs)
    
    if epochs_final is None:
        # Need to process from the beginning
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
        
        # Apply ICA for blink removal
        epochs_final = ep.apply_ica_blink_removal(epochs_final, subj_path, rerun_ica)

        if rerun_frontal_drop:
            # Drop frontal channels epochs
            epochs_final = ep.remove_frontal_channels(epochs_final, prestim_time, subj_path)
            # Save frontal drop epochs
            frontal_drop_epochs_file = os.path.join(subj_path, 'frontal_drop_epochs-epo.fif') if subj_path else None
            if frontal_drop_epochs_file:
                epochs_final.save(frontal_drop_epochs_file, overwrite=True)
                print("Saved frontal drop epochs")

        if rerun_epoch_cleaning:
            # Clean epochs and interpolate bad channels
            epochs_final = ep.clean_epochs(epochs_final, subj_path, rerun_epoch_cleaning)
    else:
        # Apply any subsequent steps that need rerunning
        if loaded_stage not in ['frontal_drop', 'cleaned'] and rerun_frontal_drop:
            epochs_final = ep.remove_frontal_channels(epochs_final, prestim_time, subj_path)
            frontal_drop_epochs_file = os.path.join(subj_path, 'frontal_drop_epochs-epo.fif') if subj_path else None
            if frontal_drop_epochs_file:
                epochs_final.save(frontal_drop_epochs_file, overwrite=True)
                print("Saved frontal drop epochs")
        
        if loaded_stage != 'cleaned' and rerun_epoch_cleaning:
            epochs_final = ep.clean_epochs(epochs_final, subj_path, rerun_epoch_cleaning)
    
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
        self.rerun_ica = args.rerun_ica
        self.rerun_frontal_drop = args.rerun_frontal_drop
        self.rerun_epoch_cleaning = args.rerun_epoch_cleaning
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
                rerun_ica=self.rerun_ica,
                rerun_frontal_drop=self.rerun_frontal_drop,
                rerun_epoch_cleaning=self.rerun_epoch_cleaning
            )

            if epochs_preprocessed is None:
                print(f"Preprocessing failed for sub-{self.sid}")
                return

        except Exception as e:
            print(f"Error processing subject {self.sid}: {e}")
            import traceback
            traceback.print_exc()
            return

        # Very last step: Apply baseline correction
        print("Applying baseline correction...")
        epochs_preprocessed.apply_baseline(baseline=(-self.prestim_time, 0))

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
    parser.add_argument('--rerun_initial',  action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun initial epochs and all subsequent steps')
    parser.add_argument('--rerun_aligned',  action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun alignment and all subsequent steps')
    parser.add_argument('--rerun_filtered',  action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun filtering and all subsequent steps')
    parser.add_argument('--rerun_ica', action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun ICA and all subsequent steps')
    parser.add_argument('--rerun_frontal_drop', action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun frontal channel removal and all subsequent steps')
    parser.add_argument('--rerun_epoch_cleaning',  action=argparse.BooleanOptionalAction, default=False,
                         help='Rerun cleaning and all subsequent steps')
    args = parser.parse_args()
    preprocess_raw(args).run()


if __name__ == '__main__':
    main()