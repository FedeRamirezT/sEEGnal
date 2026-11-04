# -*- coding: utf-8 -*-
"""

Estimate multitaper relative power in sensor or source space.

Functions
---------
estimate_relative_power_spectrum
    Compute and save relative power spectra for sensor and source signals.

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
import numpy
import mne

from sEEGnal.tools.mne_tools import (
    _check_inverse_reference,
    apply_ica,
    prepare_eeg,
)
from sEEGnal.tools.bids_tools import write_relative_power_spectrum, read_inverse_solution
from sEEGnal.tools.psd_tools import multitaper_psd, normalize_psd
from sEEGnal.tools.feature_tools import (
    get_bad_channel_policy,
    get_channel_mapping,
    get_source_metadata,
    require_finite_source_values,
)


def estimate_relative_power_spectrum(config, BIDS):
    """
    Compute and save relative power spectra for sensor and source signals.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    metadata : dict
        Metadata describing the last estimated spectrum.
    """

    # Load cleaned EEG
    ica_config = {
        'desc': 'cleaning',
        'components_to_include': [],
        'components_to_exclude': ['eog', 'ecg']
    }

    freq_limits_components = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    feature_config = config['feature_extraction']['relative_power_spectrum']
    if 'sensor' in feature_config:
        bad_channel_policy = get_bad_channel_policy(feature_config)

    raw = prepare_eeg(config, BIDS, preload=True, channels_to_include=config['global']["channels_to_include"], channels_to_exclude=config['global']["channels_to_exclude"], resample_frequency=config['component_estimation']['resample_frequency'], notch_filter=True, freq_limits=freq_limits_components, crop_seconds=config['component_estimation']['crop_seconds'], metadata_badchannels=True, exclude_badchannels=False, interpolate_badchannels=False, set_annotations=True, rereference='average')
    all_ch_names = list(raw.ch_names)
    bad_channels = [
        ch_name
        for ch_name in all_ch_names
        if ch_name in raw.info['bads']
    ]
    good_ch_names = [
        ch_name
        for ch_name in all_ch_names
        if ch_name not in raw.info['bads']
    ]
    (
        good_channel_indices,
        bad_channel_indices,
        _,
    ) = get_channel_mapping(all_ch_names, good_ch_names)

    raw = apply_ica(config, BIDS, raw, ica_config)

    sensor_epochs = None
    if 'sensor' in feature_config:
        sensor_epochs = prepare_eeg(
            config,
            BIDS,
            raw=raw.copy(),
            metadata_badchannels=True,
            exclude_badchannels=False,
            interpolate_badchannels=(bad_channel_policy == 'interpolate'),
            epoch_definition=feature_config['epoch_definition'],
        )

    source_epochs = None
    if 'source' in feature_config:
        source_epochs = prepare_eeg(
            config,
            BIDS,
            raw=raw.copy(),
            metadata_badchannels=True,
            exclude_badchannels=True,
            interpolate_badchannels=False,
            epoch_definition=feature_config['epoch_definition'],
        )

    # SENSOR LEVEL
    if sensor_epochs is not None:

        params = feature_config['sensor']
        n_channels = len(all_ch_names)
        n_good_channels = len(good_channel_indices)
        sfreq = sensor_epochs.info["sfreq"]

        data = sensor_epochs.get_data()

        psd, freqs = multitaper_psd(
            data=data,
            sfreq=sensor_epochs.info['sfreq'],
            fmin=params['freq_limits'][0],
            fmax=params['freq_limits'][-1],
            bandwidth=params['bandwidth'],
            adaptive=params['adaptive'],
            average_epochs=True,
            dtype=numpy.float32
        )

        relative_power_spectrum = normalize_psd(psd, mode="relative")

        metadata = {
            "method": "custom_multitaper",
            "bandwidth": params['bandwidth'],
            "adaptive": params['adaptive'],
            "fmin": params['freq_limits'][0],
            "fmax": params['freq_limits'][-1],
            "sfreq": sfreq,
            "epoch_length": sensor_epochs.tmax - sensor_epochs.tmin,
            "normalization": "relative_per_channel_after_epoch_average",
            "ch_names": all_ch_names,
            "n_nodes": n_channels,
            "n_good_nodes": n_good_channels,
            "good_channel_indices": good_channel_indices.tolist(),
            "bad_channels": bad_channels,
            "bad_channel_indices": bad_channel_indices.tolist(),
            "nan_channel_indices": [],
            "bad_channel_policy": bad_channel_policy,
            "freqs": freqs,
            "dim": "sensor x freqs",
            "shape": relative_power_spectrum.shape
        }

        # Save the result
        config['current_space'] = 'sensor'
        write_relative_power_spectrum(
            config,
            BIDS,
            relative_power_spectrum=relative_power_spectrum,
            metadata=metadata
        )
        del config['current_space']

    # SOURCE LEVEL
    if source_epochs is not None:

        params = feature_config['source']
        sfreq = source_epochs.info["sfreq"]

        dummy = config['subsystem']
        config['subsystem'] = 'source_reconstruction'
        filters = read_inverse_solution(config, BIDS)
        config['subsystem'] = dummy

        # A beamformer is valid only in the referenced sensor space used to
        # estimate it, so verify provenance before transforming any epoch.
        _check_inverse_reference(filters, source_epochs)
        stcs = mne.beamformer.apply_lcmv_epochs(source_epochs, filters)
        if not stcs:
            raise ValueError('LCMV did not produce any source estimates.')
        for epoch_index, stc in enumerate(stcs):
            require_finite_source_values(
                stc.data,
                f'LCMV source data for epoch {epoch_index}',
            )
        source_metadata = get_source_metadata(
            all_ch_names,
            bad_channels,
            filters,
            stcs[0],
            source_spacing=config['source_reconstruction']['forward'].get(
                'template', {}
            ).get('spacing', ''),
        )

        # Stack (n_epochs, n_vertices, n_times)
        data = numpy.stack([stc.data for stc in stcs])

        # Estimate power
        power_spectrum, freqs = multitaper_psd(
            data=data,
            sfreq=source_epochs.info['sfreq'],
            fmin=params['freq_limits'][0],
            fmax=params['freq_limits'][-1],
            bandwidth=params['bandwidth'],
            adaptive=params['adaptive'],
            average_epochs=True,
            dtype=numpy.float32
        )

        # Get relative power_spectrum
        relative_power_spectrum = normalize_psd(power_spectrum, mode="relative")
        require_finite_source_values(
            relative_power_spectrum,
            'Source relative power spectrum',
        )

        metadata = {
            "method": "custom_multitaper",
            "bandwidth": params['bandwidth'],
            "adaptive": params['adaptive'],
            "fmin": params['freq_limits'][0],
            "fmax": params['freq_limits'][-1],
            "sfreq": sfreq,
            "epoch_length": source_epochs.tmax - source_epochs.tmin,
            "n_epochs_used": len(stcs),
            "normalization": "relative_per_vertex_after_epoch_average",
            "freqs": freqs,
            "dim": "vertices x freqs",
            "shape": relative_power_spectrum.shape,
            **source_metadata,
        }

        # Save the result
        config['current_space'] = 'source'
        write_relative_power_spectrum(
            config,
            BIDS,
            relative_power_spectrum=relative_power_spectrum,
            metadata=metadata
        )
        del config['current_space']

    return metadata
