"""Read Neuroelectrics NEDF recordings and normalize their events."""

from pathlib import Path

import mne
import numpy


NEDF_STIM_CHANNEL = 'STI 014'
NEDF_DATA_LOSS_MARKER = 255
NEDF_ADC_MAX = (1 << 23) - 1
NEDF_ADC_SCALE_VOLTS = 2.4 / (6 * NEDF_ADC_MAX)
NEDF_VALID_MAX_VOLTS = NEDF_ADC_MAX * NEDF_ADC_SCALE_VOLTS
NEDF_MISREAD_MIN_VOLTS = (1 << 23) * NEDF_ADC_SCALE_VOLTS


def _correct_signed_24bit_minimum(raw):
    """Restore the sign of the exact 24-bit minimum misread by MNE.

    Returns the number of corrected EEG samples. NEDF data must be preloaded
    because a lazy Raw would decode the original value again on every read.
    """

    if not raw.preload:
        raise RuntimeError(
            'NEDF recordings must be preloaded so sEEGnal can correct the '
            'signed 24-bit minimum; use preload=True.'
        )

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    correction_count = 0
    for pick in eeg_picks:
        values = raw.get_data(picks=[pick])[0]
        above_valid_max = values > NEDF_VALID_MAX_VOLTS
        unexpected = above_valid_max & (values != NEDF_MISREAD_MIN_VOLTS)
        if numpy.any(unexpected):
            channel = raw.ch_names[pick]
            unexpected_value = values[numpy.flatnonzero(unexpected)[0]]
            raise RuntimeError(
                f'NEDF channel {channel!r} contains an unexpected value '
                f'above the valid 24-bit range: {unexpected_value!r} V.'
            )
        correction_count += int(numpy.count_nonzero(
            values == NEDF_MISREAD_MIN_VOLTS
        ))

    if correction_count:
        def replace_misread_minimum(values):
            values[values == NEDF_MISREAD_MIN_VOLTS] = (
                -NEDF_MISREAD_MIN_VOLTS
            )
            return values

        raw.apply_function(
            replace_misread_minimum,
            picks=eeg_picks,
            channel_wise=True,
            n_jobs=1,
        )

    previous_count = getattr(
        raw,
        '_seegnal_nedf_signed_min_corrections',
        0,
    )
    raw._seegnal_nedf_signed_min_corrections = (
        previous_count + correction_count
    )
    return correction_count


def _add_event_annotations(raw):
    """Copy experimental NEDF triggers from the stim channel to annotations.

    Marker 255 identifies data loss and is deliberately excluded here. It is
    handled separately because it represents an interval rather than an
    experimental event.
    """

    if NEDF_STIM_CHANNEL not in raw.ch_names:
        return raw

    events = mne.find_events(
        raw,
        stim_channel=NEDF_STIM_CHANNEL,
        output='onset',
        consecutive=True,
        shortest_event=1,
        initial_event=True,
        verbose=False,
    )
    if not len(events):
        return raw

    events = events[events[:, 2] != NEDF_DATA_LOSS_MARKER]
    if not len(events):
        return raw

    event_desc = {
        int(code): f'Stimulus/s{int(code)}'
        for code in set(events[:, 2])
    }
    orig_time = raw.annotations.orig_time
    if orig_time is None:
        orig_time = raw.info['meas_date']

    annotations = mne.annotations_from_events(
        events,
        sfreq=raw.info['sfreq'],
        event_desc=event_desc,
        first_samp=raw.first_samp,
        orig_time=orig_time,
    )
    raw.set_annotations(raw.annotations + annotations)
    return raw


def _add_data_loss_annotations(raw):
    """Annotate contiguous marker-255 samples as acquisition gaps."""

    if NEDF_STIM_CHANNEL not in raw.ch_names:
        return raw

    stim = raw.get_data(picks=[NEDF_STIM_CHANNEL])[0]
    is_data_loss = stim == NEDF_DATA_LOSS_MARKER
    if not numpy.any(is_data_loss):
        return raw

    padded = numpy.concatenate(([False], is_data_loss, [False]))
    transitions = numpy.diff(padded.astype(numpy.int8))
    starts = numpy.flatnonzero(transitions == 1)
    stops = numpy.flatnonzero(transitions == -1)

    orig_time = raw.annotations.orig_time
    if orig_time is None:
        orig_time = raw.info['meas_date']
    sfreq = raw.info['sfreq']
    annotations = mne.Annotations(
        onset=starts / sfreq,
        duration=(stops - starts) / sfreq,
        description=['BAD_ACQ_SKIP'] * len(starts),
        orig_time=orig_time,
    )
    raw.set_annotations(raw.annotations + annotations)
    return raw


def read_raw_nedf(source_filepath, preload=True):
    """Read NEDF data and preserve events and acquisition gaps as annotations."""

    raw = mne.io.read_raw_nedf(
        Path(source_filepath),
        preload=preload,
    )
    _correct_signed_24bit_minimum(raw)
    raw = _add_event_annotations(raw)
    return _add_data_loss_annotations(raw)
