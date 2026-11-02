"""Tests for source-format dispatch and NEDF event normalization."""

from datetime import UTC, datetime
import unittest
from unittest.mock import patch

import mne
import numpy

from sEEGnal.io.nedf import (
    _add_data_loss_annotations,
    _add_event_annotations,
    read_raw_nedf,
)
from sEEGnal.io.read_source_files import read_source_files


class TestNedfEventAnnotations(unittest.TestCase):
    """Preserve experimental NEDF markers before the stim channel is dropped."""

    @staticmethod
    def make_raw():
        """Create EEG and stim data with one-sample events and a loss interval."""

        sfreq = 500
        data = numpy.zeros((2, 50))
        data[1, 0] = 2
        data[1, 10] = 1
        data[1, 12] = 1
        data[1, 20] = 11
        data[1, 30:34] = 255
        info = mne.create_info(
            ['Cz', 'STI 014'],
            sfreq=sfreq,
            ch_types=['eeg', 'stim'],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        meas_date = datetime(2026, 1, 2, tzinfo=UTC)
        raw.set_meas_date(meas_date)
        raw.set_annotations(mne.Annotations(
            onset=[0.01],
            duration=[0.004],
            description=['existing'],
            orig_time=meas_date,
        ))
        return raw

    def test_experimental_markers_become_sample_aligned_annotations(self):
        raw = _add_event_annotations(self.make_raw())

        annotations = [
            (onset, duration, description)
            for onset, duration, description in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
            )
            if description.startswith('Stimulus/')
        ]

        self.assertEqual(
            annotations,
            [
                (0.0, 0.0, 'Stimulus/s2'),
                (0.02, 0.0, 'Stimulus/s1'),
                (0.024, 0.0, 'Stimulus/s1'),
                (0.04, 0.0, 'Stimulus/s11'),
            ],
        )
        self.assertEqual(raw.annotations.orig_time, raw.info['meas_date'])

    def test_existing_annotations_are_preserved_and_255_is_excluded(self):
        raw = _add_event_annotations(self.make_raw())

        self.assertIn('existing', raw.annotations.description)
        self.assertNotIn('Stimulus/s255', raw.annotations.description)
        self.assertEqual(raw.ch_names, ['Cz', 'STI 014'])

    def test_contiguous_255_samples_become_one_bad_annotation(self):
        raw = _add_event_annotations(self.make_raw())

        returned = _add_data_loss_annotations(raw)

        bad_annotations = [
            (onset, duration, description)
            for onset, duration, description in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
            )
            if description == 'BAD_ACQ_SKIP'
        ]
        self.assertIs(returned, raw)
        self.assertEqual(
            bad_annotations,
            [(0.06, 0.008, 'BAD_ACQ_SKIP')],
        )
        self.assertIn('existing', raw.annotations.description)

    def test_data_loss_runs_at_recording_boundaries_are_supported(self):
        raw = self.make_raw()
        stim = raw.get_data(picks=['STI 014'])[0]
        stim[:] = 0
        stim[:2] = 255
        stim[-3:] = 255
        raw._data[raw.ch_names.index('STI 014')] = stim

        _add_data_loss_annotations(raw)

        bad_picks = raw.annotations.description == 'BAD_ACQ_SKIP'
        self.assertEqual(
            list(raw.annotations.onset[bad_picks]),
            [0.0, 0.094],
        )
        self.assertEqual(
            list(raw.annotations.duration[bad_picks]),
            [0.004, 0.006],
        )

    def test_annotations_survive_eeg_channel_selection(self):
        raw = _add_event_annotations(self.make_raw())
        raw = _add_data_loss_annotations(raw)

        raw.pick('eeg')

        self.assertEqual(raw.ch_names, ['Cz'])
        self.assertEqual(
            sum(desc.startswith('Stimulus/') for desc in raw.annotations.description),
            4,
        )
        self.assertIn('BAD_ACQ_SKIP', raw.annotations.description)

    def test_raw_without_stim_channel_is_unchanged(self):
        raw = self.make_raw().copy().pick('eeg')
        original_annotations = raw.annotations.copy()

        returned = _add_event_annotations(raw)

        self.assertIs(returned, raw)
        self.assertEqual(returned.annotations, original_annotations)


class TestNedfReader(unittest.TestCase):
    """Dispatch NEDF files through the format-specific reader."""

    @patch('sEEGnal.io.nedf._add_data_loss_annotations')
    @patch('sEEGnal.io.nedf._add_event_annotations')
    @patch('sEEGnal.io.nedf.mne.io.read_raw_nedf')
    def test_nedf_reader_forwards_preload_and_normalizes_events(
        self,
        mne_reader,
        add_annotations,
        add_data_loss,
    ):
        source_raw = object()
        event_normalized_raw = object()
        fully_normalized_raw = object()
        mne_reader.return_value = source_raw
        add_annotations.return_value = event_normalized_raw
        add_data_loss.return_value = fully_normalized_raw

        returned = read_raw_nedf('recording.nedf', preload=False)

        mne_reader.assert_called_once_with(
            unittest.mock.ANY,
            preload=False,
        )
        self.assertEqual(mne_reader.call_args.args[0].name, 'recording.nedf')
        add_annotations.assert_called_once_with(source_raw)
        add_data_loss.assert_called_once_with(event_normalized_raw)
        self.assertIs(returned, fully_normalized_raw)

    @patch('sEEGnal.io.read_source_files.read_raw_nedf')
    def test_public_reader_accepts_uppercase_nedf(self, nedf_reader):
        expected = object()
        nedf_reader.return_value = expected

        returned = read_source_files('recording.NEDF', preload=True)

        nedf_reader.assert_called_once_with(
            unittest.mock.ANY,
            preload=True,
        )
        self.assertIs(returned, expected)


if __name__ == '__main__':
    unittest.main()
