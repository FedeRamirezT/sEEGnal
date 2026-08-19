# -*- coding: utf-8 -*-
"""

Locate EOG, muscle, sensor and other artifacts in EEG recordings.

Functions
---------
_as_sample_indices
    Normalize and validate an array of integer sample indices.
_get_original_sample_bounds
    Return the inclusive sample bounds of the uncropped recording.
_get_original_n_times
    Return the uncropped recording length in samples.
_local_samples_to_original
    Convert Raw-local sample indices to original-recording offsets.
_absolute_samples_to_original
    Convert absolute MNE samples to original-recording offsets.
EOG_detection
    Locate ocular artifacts from frontal activity in prepared EEG data.
muscle_detection
    Locate high-frequency muscle artifacts in prepared EEG data.
sensor_detection
    Locate abrupt sensor jumps using the selected ICA decomposition.
other_detection
    Locate signal segments with amplitudes outside the configured limits.
create_annotations
    Convert artifact sample indices into an MNE annotations object.
merge_peaks
    Merge nearby or overlapping artifact peaks into continuous intervals.

Federico Ramírez-Toraño
19/06/2023

"""

# Imports
import mne
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mne_tools


def _as_sample_indices(sample_indices, parameter_name):
    """
    Normalize and validate an array of integer sample indices.

    Parameters
    ----------
    sample_indices : array-like of int
        Sample indices to normalize.
    parameter_name : str
        Parameter name included in validation errors.

    Returns
    -------
    indices : numpy.ndarray
        One-dimensional or multidimensional integer array preserving the
        shape of the input.
    """

    indices = np.asarray(sample_indices)

    if indices.size == 0:
        return indices.astype(int)

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f'{parameter_name} must contain integer sample indices.')

    return indices.astype(int, copy=False)


def _get_original_sample_bounds(raw):
    """
    Return the inclusive sample bounds of the uncropped recording.

    Stored ``original_first_samp`` and ``original_last_samp`` attributes are
    used when present. Otherwise, the current MNE bounds are treated as the
    original bounds.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording whose original bounds are requested.

    Returns
    -------
    original_first_samp : int
        Absolute MNE sample at the start of the original recording.
    original_last_samp : int
        Absolute MNE sample at the inclusive end of the original recording.
    """

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('raw must be an MNE Raw object.')

    has_original_first = hasattr(raw, 'original_first_samp')
    has_original_last = hasattr(raw, 'original_last_samp')

    if has_original_first != has_original_last:
        raise ValueError(
            'original_first_samp and original_last_samp must be stored '
            'together.'
        )

    if has_original_first:
        original_first_samp = int(raw.original_first_samp)
        original_last_samp = int(raw.original_last_samp)
    else:
        original_first_samp = int(raw.first_samp)
        original_last_samp = int(raw.last_samp)

    if original_first_samp > raw.first_samp:
        raise ValueError('original_first_samp cannot follow raw.first_samp.')

    if original_last_samp < raw.last_samp:
        raise ValueError('original_last_samp cannot precede raw.last_samp.')

    if original_last_samp < original_first_samp:
        raise ValueError('The original sample bounds are invalid.')

    return original_first_samp, original_last_samp


def _get_original_n_times(raw):
    """
    Return the uncropped recording length in samples.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording whose original length is requested.

    Returns
    -------
    n_times : int
        Number of samples between the inclusive original bounds.
    """

    original_first_samp, original_last_samp = (
        _get_original_sample_bounds(raw)
    )

    return original_last_samp - original_first_samp + 1


def _local_samples_to_original(raw, sample_indices):
    """
    Convert Raw-local sample indices to original-recording offsets.

    A local index of zero identifies the first sample currently available in
    ``raw``. Returned values instead use zero for the first sample of the
    uncropped recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Current continuous recording, which may have been cropped.
    sample_indices : array-like of int
        Sample indices relative to the beginning of the current Raw object.

    Returns
    -------
    original_indices : numpy.ndarray
        Integer sample offsets from the beginning of the original recording.
    """

    original_first_samp, _ = _get_original_sample_bounds(raw)
    indices = _as_sample_indices(sample_indices, 'sample_indices')

    if np.any(indices < 0) or np.any(indices >= raw.n_times):
        raise ValueError(
            'Local sample indices must fall inside the current Raw object.'
        )

    crop_offset = raw.first_samp - original_first_samp

    return indices + crop_offset


def _absolute_samples_to_original(raw, sample_indices):
    """
    Convert absolute MNE samples to original-recording offsets.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Current continuous recording, which may have been cropped.
    sample_indices : array-like of int
        Absolute MNE sample numbers inside the current Raw object.

    Returns
    -------
    original_indices : numpy.ndarray
        Integer sample offsets from the beginning of the original recording.
    """

    original_first_samp, _ = _get_original_sample_bounds(raw)
    indices = _as_sample_indices(sample_indices, 'sample_indices')

    if np.any(indices < raw.first_samp) or np.any(indices > raw.last_samp):
        raise ValueError(
            'Absolute sample indices must fall inside the current Raw object.'
        )

    return indices - original_first_samp


# Modules
def EOG_detection(config, BIDS):
    """
    Locate ocular artifacts from frontal activity in prepared EEG data.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    indices : numpy.ndarray
        Detected artifact sample offsets from the beginning of the original,
        uncropped recording. An empty array is returned when none of the
        configured frontal channels is available.
    n_times : int
        Number of samples in the original recording.
    sfreq : float
        Sampling frequency in hertz.
    """


    # Parameters for loading EEG recordings
    ica_config = {
        'desc': 'cleaning',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['preprocess']['artifact_detection']['EOG']['low_freq'],
        config['preprocess']['artifact_detection']['EOG']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Use the same good-channel subset used to fit the ICA.
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, crop_seconds=crop_seconds, metadata_badchannels=True, exclude_badchannels=True, rereference='average')

    # Apply IC before the detector-specific temporal filter.
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Apply the detector-specific temporal filter.
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits
    )

    # Select frontal channels
    available_channels = raw.ch_names
    frontal_channels = config['preprocess']['artifact_detection']['frontal_channels']
    frontal_channels = [
        current_channel for current_channel in frontal_channels if current_channel in available_channels
    ]
    if len(frontal_channels) > 0:
        frontal_raw = raw.copy()
        frontal_raw.pick(frontal_channels)
    else:
        # If no frontal channels, no EOG artifacts
        return (
            np.array([], dtype=int),
            _get_original_n_times(raw),
            raw.info["sfreq"]
        )

    # Find peaks after averaging the frontal channels
    frontal_data = np.mean(frontal_raw.get_data(),axis=0)
    hits = (np.abs(frontal_data) >
            config['preprocess']['artifact_detection']['EOG']['threshold'] * np.std(frontal_data))
    EOG_index = np.where(hits)[0]

    # Express local detector indices on the original recording time axis.
    EOG_index = _local_samples_to_original(raw, EOG_index)

    # Extra outputs
    n_times = _get_original_n_times(raw)
    sfreq = raw.info['sfreq']

    return EOG_index, n_times, sfreq


def muscle_detection(config, BIDS):
    """
    Locate high-frequency muscle artifacts in prepared EEG data.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    indices : numpy.ndarray
        Detected artifact sample offsets from the beginning of the original,
        uncropped recording.
    n_times : int
        Number of samples in the original recording.
    sfreq : float
        Sampling frequency in hertz.
    """


    # Parameters for loading EEG recordings
    freq_limits = [
        config['preprocess']['artifact_detection']['muscle']['low_freq'],
        config['preprocess']['artifact_detection']['muscle']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]

    # Exclude channels already classified as bad from the detector.
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, freq_limits=freq_limits, crop_seconds=crop_seconds, metadata_badchannels=True, exclude_badchannels=True)

    # Estimate the std across channels
    raw_std = np.std(raw.get_data(),axis=0)

    # Find peaks based on the total height (demeaning the signal first)
    raw_std = raw_std - np.mean(raw_std)
    height = (config['preprocess']['artifact_detection']['muscle']['threshold']
              * np.std(raw_std))
    muscle_index, _ = find_peaks(
        raw_std,
        height=height
    )

    # Express local detector indices on the original recording time axis.
    muscle_index = _local_samples_to_original(raw, muscle_index)

    # Extra outputs
    n_times = _get_original_n_times(raw)
    sfreq = raw.info['sfreq']

    return muscle_index, n_times, sfreq


def sensor_detection(config, BIDS, ica_desc):
    """
    Locate abrupt sensor jumps using the selected ICA decomposition.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica_desc : str
        Description identifying the ICA derivative to use.

    Returns
    -------
    indices : numpy.ndarray
        Detected artifact sample offsets from the beginning of the original,
        uncropped recording.
    n_times : int
        Number of samples in the original recording.
    sfreq : float
        Sampling frequency in hertz.
    """


    # Parameters for loading EEG recordings
    ica_config = {
        'desc': ica_desc,
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['preprocess']['artifact_detection']['sensor']['low_freq'],
        config['preprocess']['artifact_detection']['sensor']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = dict(
        config['preprocess']['artifact_detection']['sensor'][
            'epoch_definition'
        ]
    )
    epoch_definition['overlap'] = 0
    epoch_definition['reject_by_annotation'] = True

    # Load the preceding muscle annotations so their epochs are not
    # reclassified as abrupt sensor artifacts.
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, notch_filter=True, crop_seconds=crop_seconds, metadata_badchannels=True, exclude_badchannels=True, set_annotations=True, rereference='average')

    # Apply IC before the detector-specific temporal filter and epoching.
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Apply the detector-specific temporal filter and create epochs.
    epochs = mne_tools.prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        epoch_definition=epoch_definition
    )

    # Get the clean data
    epoch_data    = epochs.get_data()

    # Get the std for each channel and epoch
    epoch_data_std = np.std(epoch_data, axis=2)
    epoch_data_std = np.max(np.abs(epoch_data), axis=2)

    # For each epoch, estimate the threshold based on the std of the channels
    # in that epoch
    threshold = np.mean(epoch_data_std,axis=1) * config['preprocess']['artifact_detection']['sensor']['threshold']

    # Find peaks based on the threshold
    threshold = np.repeat(threshold[:, np.newaxis], epoch_data_std.shape[1], axis=1)
    hits = np.any(epoch_data_std > threshold, axis=1)
    hits = np.where(hits)[0]

    # Save the peaks index
    sensor_index = epochs.events[hits, :][:, 0]
    sensor_index = _absolute_samples_to_original(raw, sensor_index)

    # Extra outputs
    n_times = _get_original_n_times(raw)
    sfreq = epochs.info['sfreq']

    return sensor_index, n_times, sfreq


def other_detection(config, BIDS, ica_desc):
    """
    Locate signal segments with amplitudes outside the configured limits.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica_desc : str
        Description identifying the ICA derivative to use.

    Returns
    -------
    indices : numpy.ndarray
        Detected artifact sample offsets from the beginning of the original,
        uncropped recording.
    n_times : int
        Number of samples in the original recording.
    sfreq : float
        Sampling frequency in hertz.
    """


    # Parameters for loading EEG recordings
    ica_config = {
        'desc': ica_desc,
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }
    freq_limits         = [
        config['preprocess']['artifact_detection']['other']['low_freq'],
        config['preprocess']['artifact_detection']['other']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = dict(
        config['preprocess']['artifact_detection']['other'][
            'epoch_definition'
        ]
    )
    epoch_definition['reject_by_annotation'] = True

    # Load the preceding muscle annotations so their epochs are not
    # reclassified as remaining high-amplitude artifacts.
    raw = mne_tools.prepare_eeg(config, BIDS, preload=True, channels_to_include=channels_to_include, channels_to_exclude=channels_to_exclude, resample_frequency=resample_frequency, crop_seconds=crop_seconds, metadata_badchannels=True, exclude_badchannels=True, set_annotations=True, rereference='average')

    # Apply IC before the detector-specific temporal filter and epoching.
    raw = mne_tools.apply_ica(
        config,
        BIDS,
        raw,
        ica_config
    )

    # Apply the detector-specific temporal filter and create epochs.
    epochs = mne_tools.prepare_eeg(
        config,
        BIDS,
        raw=raw,
        preload=True,
        freq_limits=freq_limits,
        epoch_definition=epoch_definition
    )

    # De-mean the channels
    epoch_data = epochs.get_data().copy()
    epoch_average = np.mean(epoch_data,axis=2)
    epoch_data_demean = epoch_data - epoch_average[:, :, np.newaxis]

    # Remove the offset
    epoch_data_demean_abs = np.abs(epoch_data_demean)

    # Find impossible peaks
    other_index = epoch_data_demean_abs > config['preprocess']['artifact_detection']['other']['threshold']
    other_index = np.sum(np.sum(other_index,axis=2),axis=1)
    other_index = np.where(other_index)[0]
    other_index = epochs.events[other_index, 0]
    other_index = _absolute_samples_to_original(raw, other_index)

    # Extra outputs
    n_times = _get_original_n_times(raw)
    sfreq = epochs.info['sfreq']

    return other_index, n_times, sfreq


def create_annotations(
    peaks_index,
    n_times,
    sfreq,
    annotation_description,
    fictional_artifact_duration=0.5
):
    """
    Convert artifact sample indices into an MNE annotations object.

    Parameters
    ----------
    peaks_index : array-like of int
        Sample indices at which artifact peaks were detected.
    n_times : int
        Number of samples in the original recording.
    sfreq : float
        Sampling frequency in hertz.
    annotation_description : str
        Description assigned to the annotations.
    fictional_artifact_duration : float
        Nominal artifact duration in seconds.

    Returns
    -------
    annotations : mne.Annotations
        Artifact intervals represented as MNE annotations.
    """


    if (
        isinstance(sfreq, (bool, np.bool_))
        or not isinstance(sfreq, (int, float, np.integer, np.floating))
    ):
        raise TypeError('sfreq must be a positive number.')

    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError('sfreq must be a positive finite number.')

    if (
        isinstance(fictional_artifact_duration, (bool, np.bool_))
        or not isinstance(
            fictional_artifact_duration,
            (int, float, np.integer, np.floating)
        )
    ):
        raise TypeError('fictional_artifact_duration must be a positive number.')

    fictional_artifact_duration = float(fictional_artifact_duration)
    if (
        not np.isfinite(fictional_artifact_duration)
        or fictional_artifact_duration <= 0
    ):
        raise ValueError(
            'fictional_artifact_duration must be a positive finite number.'
        )

    if not isinstance(annotation_description, str):
        raise TypeError('annotation_description must be a string.')

    # Convert the requested duration to the nearest complete sample.
    duration_samples = int(np.round(fictional_artifact_duration * sfreq))
    if duration_samples < 1:
        raise ValueError(
            'fictional_artifact_duration must span at least one sample.'
        )

    onsets, durations = merge_peaks(
        peaks_index,
        n_times,
        duration_samples
    )

    annotations = mne.Annotations(
        onset=onsets / sfreq,
        duration=durations / sfreq,
        description=[annotation_description] * len(onsets)
    )

    return annotations


def merge_peaks(peaks_index, n_times, artifact_duration_samples):
    """
    Merge nearby or overlapping artifact peaks into continuous intervals.

    Parameters
    ----------
    peaks_index : array-like of int
        Sample indices at which artifact peaks were detected.
    n_times : int
        Number of samples in the original recording.
    artifact_duration_samples : int
        Nominal artifact duration in samples.

    Returns
    -------
    onsets : numpy.ndarray
        Onset sample of each merged interval.
    durations : numpy.ndarray
        Duration in samples of each merged interval.
    """

    for value, name in (
        (n_times, 'n_times'),
        (artifact_duration_samples, 'artifact_duration_samples')
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer)
        ):
            raise TypeError(f'{name} must be a positive integer.')
        if value < 1:
            raise ValueError(f'{name} must be greater than zero.')

    n_times = int(n_times)
    artifact_duration_samples = int(artifact_duration_samples)
    peaks = np.unique(_as_sample_indices(peaks_index, 'peaks_index'))

    if np.any(peaks < 0) or np.any(peaks >= n_times):
        raise ValueError('peaks_index must fall inside the original recording.')

    if peaks.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    merged_onsets = []
    merged_ends = []

    for peak in peaks:
        onset = int(peak)
        end = min(onset + artifact_duration_samples, n_times)

        if merged_onsets and onset <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], end)
        else:
            merged_onsets.append(onset)
            merged_ends.append(end)

    onsets = np.asarray(merged_onsets, dtype=int)
    durations = np.asarray(merged_ends, dtype=int) - onsets

    return onsets, durations
