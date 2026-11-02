"""Read Neuroelectrics NEDF recordings and normalize their events."""

from pathlib import Path

import mne
import numpy


NEDF_STIM_CHANNEL = 'STI 014'
NEDF_DATA_LOSS_MARKER = 255


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
    raw = _add_event_annotations(raw)
    return _add_data_loss_annotations(raw)
