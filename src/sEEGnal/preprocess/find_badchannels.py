# -*- coding: utf-8 -*-
"""

Locate bad EEG channels using signal and electrode-quality criteria.

Functions
---------
impossible_amplitude_detection
    Detect channels with implausibly low or high signal amplitudes.
component_detection
    Detect channels dominated by components classified as noise.
gel_bridge_detection
    Detect nearby, highly correlated channels consistent with gel bridging.
high_deviation_detection
    Detect channels whose activity deviates strongly from the channel set.

Federico Ramírez-Toraño
26/03/2024

"""

# Imports
import numpy as np
from scipy.stats import median_abs_deviation

import sEEGnal.tools.mne_tools as mne_tools


# Modules
def impossible_amplitude_detection(config, BIDS):
    """
    Detect channels with implausibly low or high signal amplitudes.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    badchannels : list of str
        Channels with implausible signal amplitudes.
    """


    # Parameters for loading EEG  recordings
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    resample_frequency = config['component_estimation']['resample_frequency']
    freq_limits         = [
        config['preprocess']['badchannel_detection']['impossible_amplitude']['low_freq'],
        config['preprocess']['badchannel_detection']['impossible_amplitude']['high_freq']
    ]
    crop_seconds        = config['preprocess']['badchannel_detection']['crop_seconds']
    epoch_definition    = config['preprocess']['badchannel_detection']['impossible_amplitude']['epoch_definition']

    # Load the raw data
    epochs = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, freq_limits=freq_limits, crop_seconds=crop_seconds, epoch_definition=epoch_definition)

    # De-mean the channels
    epoch_data = epochs.get_data().copy()

    # Estimate the average standard deviation of each epoch
    epoch_data_std = epoch_data.std(axis=2)

    # Use the low and high threshold to define impossible amplitudes
    hits_low = epoch_data_std < config['preprocess']['badchannel_detection']['impossible_amplitude']['low_threshold']
    hits_high = epoch_data_std > config['preprocess']['badchannel_detection']['impossible_amplitude']['high_threshold']

    # Get the number of occurrences per channel in percentage
    hits = hits_low | hits_high
    hits = hits.sum(axis=0) / epoch_data_std.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['preprocess']['badchannel_detection']['impossible_amplitude']['percentage_threshold'])
    impossible_amplitude_badchannels = [epochs.ch_names[hit] for hit in hits]

    return impossible_amplitude_badchannels


def component_detection(config, BIDS):
    """
    Detect channels dominated by components classified as noise.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    badchannels : list of str
        Channels dominated by noisy ICA components.
    """


    # Parameters for loading EEG  recordings
    ica_config = {
        'desc': 'badchannels',
        'components_to_include': ['other'],
        'components_to_exclude': []
    }
    freq_limits         = [
        config['preprocess']['badchannel_detection']['component_detection']['low_freq'],
        config['preprocess']['badchannel_detection']['component_detection']['high_freq']
    ]
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    resample_frequency = config['component_estimation']['resample_frequency']
    crop_seconds        = config['preprocess']['badchannel_detection']['crop_seconds']
    epoch_definition    = config['preprocess']['badchannel_detection']['component_detection']['epoch_definition']

    # Load the raw EEG
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, crop_seconds=crop_seconds)

    # Apply IC
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Apply the detector-specific filter and create epochs.
    epochs = mne_tools.prepare_eeg(
        config,
        BIDS,
        raw=raw,
        freq_limits=freq_limits,
        epoch_definition=epoch_definition
    )

    # Estimate the median and the Median Absolute Deviation
    epoch_data = epochs.get_data()
    epoch_data_std = np.std(epoch_data,axis=2)

    median = np.median(epoch_data_std)
    MAD = median_abs_deviation(epoch_data_std, axis=None)

    # Define bad channels using Z-score
    bad_epochs = ((epoch_data_std - median) / MAD > config['preprocess']['badchannel_detection']['component_detection']['threshold'])

    # Get the number of occurrences per channel in percentage
    hits = bad_epochs.sum(axis=0) / bad_epochs.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['preprocess']['badchannel_detection']['component_detection']['percentage_threshold'])
    component_badchannels = [epochs.ch_names[hit] for hit in hits]

    return component_badchannels


def gel_bridge_detection(config, BIDS):
    """
    Detect nearby, highly correlated channels consistent with gel bridging.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    badchannels : list of str
        Channels identified as possible gel bridges.
    """


    # Create empty list to append badchannels
    gel_bridge_badchannels = []

    # Parameters for loading EEG  recordings
    ica_config = {
        'desc': 'badchannels',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    gel_bridge_config = (
        config["preprocess"]["badchannel_detection"]["gel_bridge"]
    )
    freq_limits = [
        gel_bridge_config.get("low_freq", 2),
        gel_bridge_config.get("high_freq", 45),
    ]
    resample_frequency  = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds        = config['preprocess']['badchannel_detection']['crop_seconds']

    # Load the raw EEG
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        freq_limits=freq_limits,
        resample_frequency=resample_frequency,
        notch_filter=True,
        crop_seconds=crop_seconds
    )

    # Apply IC
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Get the data
    raw_data = raw.get_data().copy()

    # Estimate the correlation
    correlation_coefficients = np.corrcoef(raw_data)

    # Find highly correlated channels
    correlation_coefficients = np.triu(correlation_coefficients, k=1)
    correlation_coefficients_mask = correlation_coefficients > config['preprocess']['badchannel_detection']['gel_bridge']['threshold']

    # Find indexes of channels with sequences indicating gel-bridge.
    row_ind, col_ind = np.where(correlation_coefficients_mask)

    # If any gel_bridged badchannel, check if they are neighbours
    if len(row_ind) > 0:

        # Load the montage to assure they are neighbours (5cm for example).
        montage = raw.get_montage()
        channels_position = montage.get_positions()
        channels_position = channels_position['ch_pos']
        channels_position = channels_position.values()
        channels_position = list(channels_position)
        channels_position = np.array(channels_position)

        for ichannel in range(len(row_ind)):

            # Check the distance to assure they are neighbours (5cm for example).
            ch_pos1 = channels_position[row_ind[ichannel], :]
            ch_pos2 = channels_position[col_ind[ichannel], :]
            distance = np.linalg.norm(ch_pos1 - ch_pos2)

            # If the channels are close enough, they are gel-bridged
            if distance < config['preprocess']['badchannel_detection']['gel_bridge']['neighbour_distance']:

                gel_bridge_badchannels.append(montage.ch_names[row_ind[ichannel]])
                gel_bridge_badchannels.append(montage.ch_names[col_ind[ichannel]])

    return gel_bridge_badchannels


def high_deviation_detection(config, BIDS):
    """
    Detect channels whose activity deviates strongly from the channel set.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    badchannels : list of str
        Channels with strongly deviant activity.
    """


    # Parameters for loading EEG  recordings
    ica_config = {
        'desc': 'badchannels',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }
    freq_limits = [
        config['preprocess']['badchannel_detection']['high_deviation']['low_freq'],
        config['preprocess']['badchannel_detection']['high_deviation']['high_freq']
    ]
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    crop_seconds = config['preprocess']['badchannel_detection']['crop_seconds']
    epoch_definition = config['preprocess']['badchannel_detection']['high_deviation']['epoch_definition']

    # Load the raw EEG
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, crop_seconds=crop_seconds)

    # Apply IC
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Filter
    epochs = mne_tools.prepare_eeg(config, BIDS, raw=raw, preload=True, freq_limits=freq_limits, epoch_definition=epoch_definition)

    # Get a new copy of the data and estimate std
    epoch_data = epochs.get_data().copy()
    epoch_data_std = np.std(epoch_data,axis=2)

    # Estimate the median and the Median Absolute Deviation
    median = np.median(epoch_data_std)
    mad = median_abs_deviation(epoch_data_std,axis=None)

    # Define bad channels using Z-score
    hits = ((epoch_data_std - median) / mad >
                       config['preprocess']['badchannel_detection']['high_deviation'][
                           'threshold'])

    # Get the number of occurrences per channel in percentage
    hits = hits.sum(axis=0) / hits.shape[0]

    # Define as badchannel if many epochs are bads
    hits = np.flatnonzero(hits > config['preprocess']['badchannel_detection']['high_deviation']['percentage_threshold'])
    high_deviation_badchannels = [epochs.ch_names[hit] for hit in hits]

    return high_deviation_badchannels
