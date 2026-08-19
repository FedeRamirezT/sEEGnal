"""Shared synthetic MNE data for segmentation tests."""

import mne
import numpy


SAMPLE_SCALE = 1e-9


def make_synthetic_raw(
    sfreq=1000,
    duration=20,
    n_times=None,
    first_samp=1000,
    annotations=(),
    stim_events=()
):
    """Create deterministic continuous EEG data for segmentation tests.

    The EEG channel stores its absolute sample number, scaled to volts. Stim
    events and annotation onsets are specified in seconds relative to the
    beginning of the returned Raw object.
    """

    if n_times is None:
        n_times = int(numpy.round(duration * sfreq))
    absolute_samples = first_samp + numpy.arange(n_times)
    eeg_data = absolute_samples * SAMPLE_SCALE
    stim_data = numpy.zeros(n_times)

    for onset, event_code in stim_events:
        sample = int(numpy.round(onset * sfreq))
        stim_data[sample] = event_code

    info = mne.create_info(
        ['Cz', 'STI 014'],
        sfreq=sfreq,
        ch_types=['eeg', 'stim']
    )
    raw = mne.io.RawArray(
        numpy.vstack((eeg_data, stim_data)),
        info,
        first_samp=first_samp,
        verbose=False
    )

    if annotations:
        raw.set_annotations(
            mne.Annotations(
                onset=[annotation[0] for annotation in annotations],
                duration=[annotation[1] for annotation in annotations],
                description=[annotation[2] for annotation in annotations]
            )
        )

    return raw


def assert_epoch_contains_samples(epochs, epoch_index=0):
    """Assert that an epoch contains the expected absolute Raw samples."""

    sfreq = epochs.info['sfreq']
    event_sample = epochs.events[epoch_index, 0]
    first_offset = int(numpy.round(epochs.times[0] * sfreq))
    first_sample = event_sample + first_offset
    expected_samples = first_sample + numpy.arange(len(epochs.times))
    observed_data = epochs.get_data(picks=['Cz'])[epoch_index, 0]

    numpy.testing.assert_array_equal(
        observed_data,
        expected_samples * SAMPLE_SCALE
    )
