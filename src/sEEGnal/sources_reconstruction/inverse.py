"""

Estimate covariance and inverse solutions for source reconstruction.

Functions
---------
estimate_inverse_solution
    Select and run the configured inverse-solution method.
estimate_lcmv
    Build and save an LCMV spatial filter from cleaned EEG epochs.

Federico Ramírez-Toraño
11/02/2026

"""

# Imports
import sys
import traceback
from datetime import datetime as dt, timezone

import mne

import sEEGnal.tools.mne_tools as mne_tools
from sEEGnal.tools.bids_tools import read_forward_model, write_inverse_solution



def estimate_inverse_solution(config, BIDS):
    """
    Select and run the configured inverse-solution method.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    results : dict
        Execution status and inverse-solution information.
    """


    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    try:

        # Build the function name
        func = f"estimate_{config['source_reconstruction']['inverse']['method']}"
        to_estimate = getattr(sys.modules[__name__], func)

        # Estimate the inverse solution using the defined method
        inverse_solution = to_estimate(config, BIDS)

        # Save the metadata
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'ok',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            'inverse_solution': inverse_solution
        }

    except Exception as e:

        # Save the error
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            "details": f"Exception: {str(e)}, {traceback.format_exc()}"
        }

    return results


def estimate_lcmv(config, BIDS):
    """
    Build and save an LCMV spatial filter from cleaned EEG epochs.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    filters : mne.beamformer.Beamformer
        Estimated LCMV spatial filter with effective-reference provenance.
    """


    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    # Load the clean EEG
    ica_config = {
        'desc': 'cleaning',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, freq_limits=freq_limits, crop_seconds=crop_seconds, metadata_badchannels=True, exclude_badchannels=False, interpolate_badchannels=False, set_annotations=True, rereference='average')

    raw = mne_tools.apply_ica(config, BIDS, raw, ica_config)

    epochs = mne_tools.prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=[2,45],
        metadata_badchannels=True,
        exclude_badchannels=True,
        interpolate_badchannels=False,
        epoch_definition=epoch_definition
    )

    # Estimate the data covariance
    data_cov = mne.compute_covariance(
        epochs,
        method=config['source_reconstruction']['covariance']['method'],
        rank=config['source_reconstruction']['covariance']['rank']
    )

    # Estimate the rank of the covariance matrix
    rank = mne.compute_rank(data_cov,info=epochs.info)

    # Load the forward model
    forward_model = read_forward_model(config,BIDS)

    # Estimate the LCMV beamformers filters
    filters = mne.beamformer.make_lcmv(
        info=epochs.info,
        forward=forward_model,
        data_cov=data_cov,
        reg=config['source_reconstruction']['inverse']['reg'],
        noise_cov=None,  # resting-state
        pick_ori=config['source_reconstruction']['inverse']['pick_ori'],
        weight_norm=config['source_reconstruction']['inverse']['weight_norm'],
        reduce_rank=True,
        rank=rank
    )
    # LCMV weights are tied to the referenced sensor covariance used here;
    # preserve that state for validation every time the filters are applied.
    mne_tools._copy_reference_state(epochs, filters)

    # Save the inverse solution
    write_inverse_solution(config, BIDS, filters)

    return filters
