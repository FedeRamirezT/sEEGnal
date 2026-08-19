# -*- coding: utf-8 -*-
"""

Estimate corrected imaginary PLV in sensor or source space.

Functions
---------
estimate_ciplv
    Compute and save corrected imaginary PLV for the configured signal spaces.

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
import mne
import numpy

from sEEGnal.tools.mne_tools import (
    _check_inverse_reference,
    apply_ica,
    prepare_eeg,
)
from sEEGnal.tools.bids_tools import read_inverse_solution, write_ciplv
from sEEGnal.tools.fc_tools import compute_ciplv
from sEEGnal.tools.feature_tools import (
    expand_connectivity_vector,
    get_channel_mapping,
    get_source_metadata,
    require_finite_source_values,
)


def estimate_ciplv(config, BIDS):
    """
    Compute and save corrected imaginary PLV for the configured signal spaces.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    metadata : dict
        Metadata describing the last estimated ciPLV output.
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

    freq_limits_signal = [
        config['feature_extraction']['ciplv']['freq_limits'][0],
        config['feature_extraction']['ciplv']['freq_limits'][-1]
    ]

    raw = prepare_eeg(config, BIDS, preload=True, channels_to_include=config['global']["channels_to_include"], channels_to_exclude=config['global']["channels_to_exclude"], resample_frequency=config['component_estimation']['resample_frequency'], notch_filter=True, freq_limits=freq_limits_components, crop_seconds=config['component_estimation']['crop_seconds'], metadata_badchannels=True, exclude_badchannels=False, interpolate_badchannels=False, set_annotations=True, rereference='average')
    all_ch_names = list(raw.ch_names)
    if 'source' in config['feature_extraction']['ciplv']:
        source_input_bad_channels = [
            ch_name
            for ch_name in all_ch_names
            if ch_name in raw.info['bads']
        ]

    raw = apply_ica(config, BIDS, raw, ica_config)

    epochs = prepare_eeg(config, BIDS, raw=raw, freq_limits=freq_limits_signal, metadata_badchannels=True, exclude_badchannels=True, interpolate_badchannels=False, epoch_definition=config['feature_extraction']['ciplv']['epoch_definition'])

    # SENSOR LEVEL
    if 'sensor' in config['feature_extraction']['ciplv']:

        params = config['feature_extraction']['ciplv']['sensor']

        # Save memory
        nepochs, n_good_channels, nsamples = epochs.get_data().shape
        (
            good_channel_indices,
            bad_channel_indices,
            bad_channels,
        ) = get_channel_mapping(all_ch_names, epochs.ch_names)
        n_channels = len(all_ch_names)

        for iband,current_band in enumerate(params['freq_bands_name']):

            # Filter the data
            banddata = epochs.copy().load_data()
            banddata.filter(
                params['freq_bands_limits'][iband][0],
                params['freq_bands_limits'][iband][1]
            )

            # ciplv
            good_band_ciplv_vector = compute_ciplv(
                data=banddata.get_data(),
                average_epochs=True
            )
            (
                band_ciplv_vector,
                nan_connection_indices,
            ) = expand_connectivity_vector(
                good_band_ciplv_vector,
                good_channel_indices,
                n_channels,
            )

            # Save the metadata
            metadata = {
                "method": "ciplv",
                "n_nodes": n_channels,
                "n_good_nodes": n_good_channels,
                "ch_names": all_ch_names,
                "good_channel_indices": good_channel_indices.tolist(),
                "bad_channels": bad_channels,
                "bad_channel_indices": bad_channel_indices.tolist(),
                "nan_connection_indices": nan_connection_indices.tolist(),
                "bad_channel_policy": "nan",
                "vectorization": "numpy.triu_indices(n_nodes, k=1)",
                "n_epochs_used":nepochs,
                "band_name": current_band,
                "description": "Upper triangular ciPLV matrix values obtained with numpy.triu_indices(n_nodes, k=1); connections involving bad channels are NaN"
            }

            # Save the result
            config['current_space'] = 'sensor'
            write_ciplv(
                config,
                BIDS,
                ciplv=band_ciplv_vector,
                metadata=metadata
            )
            del config['current_space']

    # SOURCE LEVEL
    if 'source' in config['feature_extraction']['ciplv']:

        params = config['feature_extraction']['ciplv']['source']

        dummy = config['subsystem']
        config['subsystem'] = 'source_reconstruction'
        filters = read_inverse_solution(config, BIDS)
        config['subsystem'] = dummy

        # A beamformer is valid only in the referenced sensor space used to
        # estimate it, so verify provenance before transforming any epoch.
        _check_inverse_reference(filters, epochs)
        stcs = mne.beamformer.apply_lcmv_epochs(epochs, filters)
        if not stcs:
            raise ValueError('LCMV did not produce any source estimates.')
        for epoch_index, stc in enumerate(stcs):
            require_finite_source_values(
                stc.data,
                f'LCMV source data for epoch {epoch_index}',
            )

        # Save memory
        nsources, nsamples = stcs[0].shape
        source_metadata = get_source_metadata(
            all_ch_names,
            source_input_bad_channels,
            filters,
            stcs[0],
            source_spacing=config['source_reconstruction']['forward'].get(
                'template', {}
            ).get('spacing', ''),
        )

        for iband, current_band in enumerate(params['freq_bands_name']):

            # Filter the data
            filtered_stcs = []
            for stc in stcs:
                stc_copy = stc.copy()
                stc_copy.filter(
                    params['freq_bands_limits'][iband][0],
                    params['freq_bands_limits'][iband][1]
                )
                filtered_stcs.append(stc_copy)

            banddata = numpy.stack([stc.data for stc in filtered_stcs])

            # ciplv
            band_ciplv_vector = compute_ciplv(
                data=banddata,
                average_epochs=True
            )
            require_finite_source_values(
                band_ciplv_vector,
                f'Source ciPLV for band {current_band}',
            )

            # Save the metadata
            metadata = {
                "method": "ciplv",
                "n_nodes": nsources,
                "ch_names": '',
                "n_epochs_used": len(stcs),
                "band_name": current_band,
                "vectorization": "numpy.triu_indices(n_nodes, k=1)",
                "description": "Upper triangular source ciPLV matrix values obtained with numpy.triu_indices(n_nodes, k=1)",
                **source_metadata,
            }

            # Save the result
            config['current_space'] = 'source'
            write_ciplv(
                config,
                BIDS,
                ciplv=band_ciplv_vector,
                metadata=metadata
            )
            del config['current_space']


    return metadata
