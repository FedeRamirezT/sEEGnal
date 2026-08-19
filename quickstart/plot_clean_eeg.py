"""Open the EEG produced by the sEEGnal core pipeline.

Run ``python -m quickstart.run_sEEGnal`` successfully before using this
example. It loads the detected bad-channel metadata and artifact annotations,
applies the cleaning ICA, and opens the resulting :class:`mne.io.Raw` object.
Advanced users can adapt the ``raw`` object below for their own MNE analyses.

Run this example from the repository with:

``python -m quickstart.plot_clean_eeg``
"""

from pathlib import Path
import sys


if __package__ in (None, ''):
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

import quickstart.init.init as init
import sEEGnal.tools.mne_tools as mne_tools
from sEEGnal.io.recordings import RecordingsValidationError


def main():
    """Load and display every processed recording in the manifest."""

    try:
        config, recordings = init.init()
    except RecordingsValidationError as error:
        raise SystemExit(f'Input validation failed:\n{error}') from None

    for recording in recordings:
        current_sub = recording['subject']
        current_ses = recording['session']
        current_task = recording['task']
        current_run = recording['run']
        BIDS = recording['bids_path']

        print(
            f'Working with sub {current_sub} ses {current_ses} '
            f'task {current_task} run {current_run}'
        )

        ica_config = {
            'desc': 'cleaning',
            'components_to_include': ['brain', 'other'],
            'components_to_exclude': [],
        }
        freq_limits = [
            config['component_estimation']['low_freq'],
            config['component_estimation']['high_freq'],
        ]
        crop_seconds = config['component_estimation']['crop_seconds']
        resample_frequency = config['component_estimation'][
            'resample_frequency'
        ]
        channels_to_include = config['global']['channels_to_include']
        channels_to_exclude = config['global']['channels_to_exclude']

        config['subsystem'] = 'preprocess'
        raw = mne_tools.prepare_eeg(
            config,
            BIDS,
            preload=True,
            channels_to_include=channels_to_include,
            channels_to_exclude=channels_to_exclude,
            resample_frequency=resample_frequency,
            notch_filter=True,
            freq_limits=freq_limits,
            crop_seconds=crop_seconds,
            metadata_badchannels=True,
            exclude_badchannels=True,
            interpolate_badchannels=False,
            set_annotations=True,
            rereference='average',
        )
        raw = mne_tools.apply_ica(config, BIDS, raw, ica_config)
        raw = mne_tools.prepare_eeg(
            config,
            BIDS,
            raw=raw,
            freq_limits=[2, 45],
            metadata_badchannels=True,
            exclude_badchannels=True,
            interpolate_badchannels=False,
        )

        raw.plot(block=True)


if __name__ == '__main__':
    main()
