# -*- coding: utf-8 -*-
"""

Detect bad channels in EEG recordings using complementary criteria.

First, it fits an Extended Infomax ICA and classifies its independent
components. It then detects bad channels based on signal amplitude,
component activity, power spectrum, impedance and other criteria.

Functions
---------
badchannel_detection
    Coordinate bad-channel detection, quality control and result reporting.
eeg_badchannel_detection
    Apply all configured bad-channel criteria to an EEG recording.
estimate_badchannel_component
    Fit, classify and save the ICA used by component-based detection.

Federico Ramírez-Toraño
27/04/2023

"""

# Imports
import traceback
from datetime import datetime as dt, timezone

import mne
import mne_icalabel.iclabel as iclabel

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mne_tools
import sEEGnal.preprocess.find_badchannels as find_badchannels
from sEEGnal.tools.qc_tools import badchannels_qc

# Set the output levels
mne.utils.set_log_level(verbose='ERROR')


# Modules
def badchannel_detection(config, BIDS):
    """
    Coordinate bad-channel detection, quality control and result reporting.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    results : dict
        Execution status and detected bad-channel information.
    """


    # Add the subsystem flag
    config['subsystem'] = 'preprocess'

    # For EEG
    if BIDS.datatype == 'eeg':

        try:

            # Detect badchannels in the current recording
            badchannels = eeg_badchannel_detection(config, BIDS)

            # QC
            badchannels_qc(config, BIDS)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {
                'result': 'ok',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'badchannels': badchannels
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

    else:

        # Not accepted type to process
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'file': BIDS.basename,
            'details': 'Not accepted type of file to process',
            'date': formatted_now
        }

    return results


def eeg_badchannel_detection(config, BIDS):
    """
    Apply all configured bad-channel criteria to an EEG recording.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    badchannels : list of str
        Combined bad channels from all detection criteria.
    """


    # Estimate Independent Components
    estimate_badchannel_component(config, BIDS)

    # Create an empty list to append badchannels
    badchannels = []
    badchannels_description = []

    # Find channels with biologically impossible amplitude
    impossible_amplitude_badchannels = find_badchannels.impossible_amplitude_detection(config, BIDS)
    badchannels.extend(impossible_amplitude_badchannels)
    current_badchannel_description = [
        'bad_impossible_amplitude_badchannels' for i in range(len(impossible_amplitude_badchannels))
    ]
    badchannels_description.extend(current_badchannel_description)

    # Find abnormal power spectrum
    components_badchannels = find_badchannels.component_detection(config, BIDS)
    badchannels.extend(components_badchannels)
    current_badchannel_description = ['bad_component' for i in range(len(components_badchannels))]
    badchannels_description.extend(current_badchannel_description)

    # Find channels with gel bridge
    gel_bridge_badchannels = find_badchannels.gel_bridge_detection(config, BIDS)
    badchannels.extend(gel_bridge_badchannels)
    current_badchannel_description = ['bad_gel_bridge' for i in range(len(gel_bridge_badchannels))]
    badchannels_description.extend(current_badchannel_description)

    # Find channels with high variance
    high_deviation_badchannels = find_badchannels.high_deviation_detection(config, BIDS)
    badchannels.extend(high_deviation_badchannels)
    current_badchannel_description = ['bad_high_deviation' for i in range(len(high_deviation_badchannels))]
    badchannels_description.extend(current_badchannel_description)

    # Save the results
    bids.write_badchannels(config, BIDS, badchannels, badchannels_description)
    return badchannels


def estimate_badchannel_component(config, BIDS):
    """
    Fit, classify and save the ICA used by component-based detection.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """


    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config[
        'preprocess'
    ]['badchannel_detection']['crop_seconds']
    resample_frequency = config[
        'component_estimation'
    ]['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']

    # Bad-channel metadata is intentionally not loaded here: this ICA is
    # estimated before the bad-channel detectors have produced their results.
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        notch_filter=True,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds
    )

    # Let MNE determine the numerically stable data rank while omitting any
    # acquisition intervals already marked BAD in the original recording.
    ica = mne_tools.fit_ica(raw, reject_by_annotation=True)

    # ICLabel does not honor Raw annotations when extracting its temporal
    # features, so materialize the same usable samples seen by ICA.fit.
    classification_raw = mne_tools.get_unannotated_raw(raw)

    # Classify the independent components and retain the probability assigned
    # to every ICLabel category for each component.
    ica.labels_scores_ = iclabel.iclabel_label_components(
        classification_raw,
        ica,
        inplace=True,
        backend='onnx'
    )

    max_probability = ica.labels_scores_.max(axis=1)
    unclear_components = {
        component
        for component, probability in enumerate(max_probability)
        if probability
        < config['component_estimation']['unclear_threshold']
    }

    classified_labels = [
        'brain',
        'muscle',
        'eog',
        'ecg',
        'line_noise',
        'ch_noise'
    ]

    for label in classified_labels:
        ica.labels_[label] = [
            component
            for component in ica.labels_[label]
            if component not in unclear_components
        ]

    ica.labels_['other'] = sorted(
        set(ica.labels_['other']) | unclear_components
    )

    bids.write_ica(
        config,
        BIDS,
        ica,
        desc='badchannels'
    )
