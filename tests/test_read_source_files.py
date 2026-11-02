"""Tests for source-format dispatch and NEDF event normalization."""

from datetime import UTC, datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy

from sEEGnal.io.nedf import (
    NEDF_MISREAD_MIN_VOLTS,
    NEDF_VALID_MAX_VOLTS,
    _add_data_loss_annotations,
    _add_event_annotations,
    _correct_signed_24bit_minimum,
    _read_nedf_source_metadata,
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
    @patch('sEEGnal.io.nedf._correct_signed_24bit_minimum')
    @patch('sEEGnal.io.nedf.mne.io.read_raw_nedf')
    @patch('sEEGnal.io.nedf._read_nedf_source_metadata')
    def test_nedf_reader_corrects_samples_before_normalizing_annotations(
        self,
        metadata_reader,
        mne_reader,
        correct_samples,
        add_annotations,
        add_data_loss,
    ):
        source_raw = SimpleNamespace()
        event_normalized_raw = object()
        fully_normalized_raw = object()
        source_metadata = {
            'Manufacturer': 'Neuroelectrics',
            'ManufacturersModelName': 'Enobio 20',
        }
        metadata_reader.return_value = source_metadata
        mne_reader.return_value = source_raw
        add_annotations.return_value = event_normalized_raw
        add_data_loss.return_value = fully_normalized_raw

        returned = read_raw_nedf('recording.nedf', preload=True)

        mne_reader.assert_called_once_with(
            unittest.mock.ANY,
            preload=True,
        )
        self.assertEqual(mne_reader.call_args.args[0].name, 'recording.nedf')
        metadata_reader.assert_called_once_with(mne_reader.call_args.args[0])
        self.assertEqual(
            source_raw._seegnal_source_bids_metadata,
            source_metadata,
        )
        correct_samples.assert_called_once_with(source_raw)
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

    def test_nedf_header_provides_bids_source_metadata(self):
        xml = (
            '<Recording>'
            '<StepDetails>'
            '<DeviceClass>ENOBIO</DeviceClass>'
            '<SoftwareVersion>NIC v2.0.11.7</SoftwareVersion>'
            '<FirmwareVersion>3026</FirmwareVersion>'
            '</StepDetails>'
            '<EEGSettings>'
            '<NumberOfEEGChannels>20</NumberOfEEGChannels>'
            '</EEGSettings>'
            '</Recording>'
        ).encode('utf-8')

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'recording.nedf'
            path.write_bytes(xml + bytes(10_240 - len(xml)))
            metadata = _read_nedf_source_metadata(path)

        self.assertEqual(metadata, {
            'Manufacturer': 'Neuroelectrics',
            'ManufacturersModelName': 'Enobio 20',
            'SoftwareVersions': 'NIC v2.0.11.7; Firmware 3026',
        })


class TestNedfSignedMinimumCorrection(unittest.TestCase):
    """Correct only MNE's exact misinterpretation of the signed minimum."""

    @staticmethod
    def make_raw(eeg_data, stim_data=None):
        eeg_data = numpy.asarray(eeg_data, dtype=float)
        if eeg_data.ndim == 1:
            eeg_data = eeg_data[numpy.newaxis, :]
        names = [f'EEG{index}' for index in range(len(eeg_data))]
        types = ['eeg'] * len(eeg_data)
        data = eeg_data
        if stim_data is not None:
            names.append('STI 014')
            types.append('stim')
            data = numpy.vstack((data, numpy.asarray(stim_data, dtype=float)))
        info = mne.create_info(names, sfreq=500, ch_types=types)
        return mne.io.RawArray(data, info, verbose=False)

    def test_exact_misread_minimum_is_corrected_only_in_eeg(self):
        raw = self.make_raw(
            [[
                NEDF_MISREAD_MIN_VOLTS,
                NEDF_VALID_MAX_VOLTS,
                -NEDF_MISREAD_MIN_VOLTS,
                0,
            ]],
            stim_data=[
                NEDF_MISREAD_MIN_VOLTS,
                NEDF_VALID_MAX_VOLTS,
                255,
                0,
            ],
        )

        count = _correct_signed_24bit_minimum(raw)

        numpy.testing.assert_array_equal(
            raw.get_data(picks='eeg')[0],
            [
                -NEDF_MISREAD_MIN_VOLTS,
                NEDF_VALID_MAX_VOLTS,
                -NEDF_MISREAD_MIN_VOLTS,
                0,
            ],
        )
        numpy.testing.assert_array_equal(
            raw.get_data(picks='stim')[0],
            [
                NEDF_MISREAD_MIN_VOLTS,
                NEDF_VALID_MAX_VOLTS,
                255,
                0,
            ],
        )
        self.assertEqual(count, 1)
        self.assertEqual(raw._seegnal_nedf_signed_min_corrections, 1)

    def test_correction_is_idempotent_and_preserves_provenance_count(self):
        raw = self.make_raw([NEDF_MISREAD_MIN_VOLTS, 0])

        first_count = _correct_signed_24bit_minimum(raw)
        second_count = _correct_signed_24bit_minimum(raw)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        self.assertEqual(raw._seegnal_nedf_signed_min_corrections, 1)

    def test_recording_without_misread_minimum_is_unchanged(self):
        original = numpy.array([
            -NEDF_MISREAD_MIN_VOLTS,
            NEDF_VALID_MAX_VOLTS,
            0,
        ])
        raw = self.make_raw(original)

        count = _correct_signed_24bit_minimum(raw)

        numpy.testing.assert_array_equal(raw.get_data()[0], original)
        self.assertEqual(count, 0)

    def test_unexpected_value_above_valid_range_fails_without_mutation(self):
        original = numpy.array([0.5, NEDF_MISREAD_MIN_VOLTS])
        raw = self.make_raw(original)

        with self.assertRaisesRegex(RuntimeError, 'unexpected value'):
            _correct_signed_24bit_minimum(raw)

        numpy.testing.assert_array_equal(raw.get_data()[0], original)

    def test_lazy_nedf_is_rejected(self):
        raw = SimpleNamespace(preload=False)

        with self.assertRaisesRegex(RuntimeError, 'use preload=True'):
            _correct_signed_24bit_minimum(raw)


if __name__ == '__main__':
    unittest.main()
