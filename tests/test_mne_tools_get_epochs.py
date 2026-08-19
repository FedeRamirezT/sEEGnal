"""Tests for sample counts in fixed-length and event-based epochs."""

import json
import unittest
from pathlib import Path

import mne
import numpy

from sEEGnal.tools.mne_tools import get_epochs
from tests._mne_test_data import (
    SAMPLE_SCALE as _SAMPLE_SCALE,
    assert_epoch_contains_samples as _assert_epoch_contains_absolute_samples,
    make_synthetic_raw as _make_synthetic_raw,
)


def _assert_epochs_equal(test_case, actual, expected):
    """Assert that sEEGnal produced the same Epochs object as direct MNE."""

    test_case.assertEqual(actual.event_id, expected.event_id)
    test_case.assertEqual(actual.drop_log, expected.drop_log)
    numpy.testing.assert_array_equal(actual.events, expected.events)
    numpy.testing.assert_array_equal(actual.selection, expected.selection)
    numpy.testing.assert_allclose(actual.times, expected.times)
    numpy.testing.assert_allclose(actual.get_data(), expected.get_data())


class TestSyntheticRaw(unittest.TestCase):
    """Check the common timing model used by segmentation tests."""

    def test_absolute_samples_annotations_and_stim_events(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=2,
            first_samp=1250,
            annotations=((0.5, 0, 'target'),),
            stim_events=((1, 32),)
        )

        self.assertEqual(raw.first_samp, 1250)
        self.assertEqual(raw.last_samp, 3249)
        numpy.testing.assert_array_equal(
            raw.get_data(picks=['Cz'])[0],
            numpy.arange(1250, 3250) * _SAMPLE_SCALE
        )

        annotation_events, annotation_event_id = (
            mne.events_from_annotations(raw, verbose=False)
        )
        self.assertEqual(annotation_event_id, {'target': 1})
        self.assertEqual(annotation_events[0, 0], 1750)

        stim_events = mne.find_events(raw, verbose=False)
        self.assertEqual(stim_events[0, 0], 2250)
        self.assertEqual(stim_events[0, 2], 32)


class TestEpochSampleCounts(unittest.TestCase):
    """Check that configured durations determine exact sample counts."""

    def test_fixed_epoch_sample_counts(self):
        cases = (
            {
                'sfreq': 1000,
                'duration': 4,
                'expected_samples': 4000,
                'expected_tmin': 0,
                'expected_tmax': 3.999,
            },
            {
                'sfreq': 1000,
                'duration': 1,
                'expected_samples': 1000,
                'expected_tmin': 0,
                'expected_tmax': 0.999,
            },
        )

        for case in cases:
            with self.subTest(case=case):
                raw = _make_synthetic_raw(
                    sfreq=case['sfreq'],
                    first_samp=0
                )
                epoch_definition = {
                    'mode': 'fixed',
                    'duration': case['duration'],
                    'overlap': 0,
                    'reject_by_annotation': False,
                }

                epochs = get_epochs(raw, True, epoch_definition)

                self.assertEqual(len(epochs.times), case['expected_samples'])
                self.assertEqual(
                    epochs.get_data().shape[-1],
                    case['expected_samples']
                )
                self.assertAlmostEqual(
                    epochs.times[0],
                    case['expected_tmin']
                )
                self.assertAlmostEqual(
                    epochs.times[-1],
                    case['expected_tmax']
                )
                _assert_epoch_contains_absolute_samples(epochs)

    def test_event_epoch_sample_counts(self):
        cases = (
            {
                'tmin': 0,
                'tmax': 1,
                'expected_samples': 1001,
                'expected_tmin': 0,
                'expected_tmax': 1,
            },
            {
                'tmin': -0.2,
                'tmax': 0.8,
                'expected_samples': 1001,
                'expected_tmin': -0.2,
                'expected_tmax': 0.8,
            },
        )

        for case in cases:
            with self.subTest(case=case):
                raw = _make_synthetic_raw(
                    sfreq=1000,
                    first_samp=0,
                    annotations=((5, 0, 'target'),)
                )
                epoch_definition = {
                    'mode': 'events',
                    'event_source': 'annotations',
                    'event_id': 1,
                    'tmin': case['tmin'],
                    'tmax': case['tmax'],
                    'baseline': None,
                    'reject_by_annotation': True,
                }

                epochs = get_epochs(
                    raw,
                    True,
                    epoch_definition
                )

                self.assertEqual(len(epochs), 1)
                self.assertEqual(len(epochs.times), case['expected_samples'])
                self.assertEqual(
                    epochs.get_data().shape[-1],
                    case['expected_samples']
                )
                self.assertAlmostEqual(
                    epochs.times[0],
                    case['expected_tmin']
                )
                self.assertAlmostEqual(
                    epochs.times[-1],
                    case['expected_tmax']
                )
                self.assertEqual(
                    epochs.events[0, 0],
                    raw.first_samp + int(5 * raw.info['sfreq'])
                )
                _assert_epoch_contains_absolute_samples(epochs)


class TestEpochConstruction(unittest.TestCase):
    """Check basic fixed-length and event-based segmentation."""

    def test_fixed_segments_cover_exact_recording(self):
        sfreq = 1000
        duration = 2
        epoch_samples = int(duration * sfreq)
        raw = _make_synthetic_raw(
            sfreq=sfreq,
            n_times=6 * epoch_samples,
            first_samp=0
        )
        epoch_definition = {
            'mode': 'fixed',
            'duration': duration,
            'overlap': 0,
            'reject_by_annotation': False,
        }

        epochs = get_epochs(raw, True, epoch_definition)

        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.arange(6) * epoch_samples
        )
        numpy.testing.assert_array_equal(epochs.events[:, 2], 1)
        self.assertEqual(epochs.event_id, {'1': 1})
        self.assertEqual(epochs.get_data().shape, (6, 2, epoch_samples))
        self.assertTrue(all(reason == () for reason in epochs.drop_log))

        for epoch_index in range(6):
            _assert_epoch_contains_absolute_samples(epochs, epoch_index)

    def test_fixed_segments_ignore_incomplete_remainder(self):
        sfreq = 1000
        duration = 2
        epoch_samples = int(duration * sfreq)
        raw = _make_synthetic_raw(
            sfreq=sfreq,
            n_times=6 * epoch_samples + 10,
            first_samp=0
        )
        epoch_definition = {
            'mode': 'fixed',
            'duration': duration,
            'overlap': 0,
            'reject_by_annotation': False,
        }

        epochs = get_epochs(raw, True, epoch_definition)

        self.assertEqual(len(epochs), 6)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.arange(6) * epoch_samples
        )
        self.assertEqual(len(epochs.drop_log), 6)
        self.assertTrue(all(reason == () for reason in epochs.drop_log))
        self.assertNotIn(
            'TOO_SHORT',
            {reason for reasons in epochs.drop_log for reason in reasons}
        )
        _assert_epoch_contains_absolute_samples(epochs, epoch_index=5)
        last_epoch_data = epochs.get_data(picks=['Cz'])[5, 0]
        self.assertEqual(last_epoch_data[-1], 11999 * _SAMPLE_SCALE)

    def test_fixed_rejects_epoch_overlapping_bad_annotation(self):
        sfreq = 1000
        duration = 2
        epoch_samples = int(duration * sfreq)
        raw = _make_synthetic_raw(
            sfreq=sfreq,
            n_times=6 * epoch_samples,
            first_samp=0,
            annotations=((4.5, 0.5, 'BAD_artifact'),)
        )
        epoch_definition = {
            'mode': 'fixed',
            'duration': duration,
            'overlap': 0,
            'reject_by_annotation': True,
        }

        epochs = get_epochs(raw, True, epoch_definition)

        self.assertEqual(len(epochs), 5)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.array([0, 2000, 6000, 8000, 10000])
        )
        numpy.testing.assert_array_equal(
            epochs.selection,
            numpy.array([0, 1, 3, 4, 5])
        )
        self.assertEqual(
            epochs.drop_log,
            ((), (), ('BAD_artifact',), (), (), ())
        )
        self.assertEqual(epochs.get_data().shape, (5, 2, epoch_samples))

    def test_fixed_keeps_bad_annotated_epoch_when_rejection_is_disabled(self):
        sfreq = 1000
        duration = 2
        epoch_samples = int(duration * sfreq)
        raw = _make_synthetic_raw(
            sfreq=sfreq,
            n_times=6 * epoch_samples,
            first_samp=0,
            annotations=((4.5, 0.5, 'BAD_artifact'),)
        )
        epoch_definition = {
            'mode': 'fixed',
            'duration': duration,
            'overlap': 0,
            'reject_by_annotation': False,
        }

        epochs = get_epochs(raw, True, epoch_definition)

        self.assertEqual(len(epochs), 6)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.arange(6) * epoch_samples
        )
        numpy.testing.assert_array_equal(epochs.selection, numpy.arange(6))
        self.assertEqual(epochs.drop_log, ((), (), (), (), (), ()))
        self.assertEqual(epochs.get_data().shape, (6, 2, epoch_samples))

    def test_annotation_and_stim_events_create_same_segments(self):
        sfreq = 1000
        event_onsets = (2, 5, 8)
        annotations_raw = _make_synthetic_raw(
            sfreq=sfreq,
            duration=12,
            first_samp=0,
            annotations=tuple(
                (onset, 0, 'target') for onset in event_onsets
            )
        )
        stim_raw = _make_synthetic_raw(
            sfreq=sfreq,
            duration=12,
            first_samp=0,
            stim_events=tuple((onset, 32) for onset in event_onsets)
        )
        annotations_definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': {'target': 1},
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
            'reject_by_annotation': True,
        }
        stim_definition = {
            'mode': 'events',
            'event_source': 'stim_channel',
            'event_id': 32,
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
            'reject_by_annotation': False,
        }

        annotation_epochs = get_epochs(
            annotations_raw,
            True,
            annotations_definition
        )
        stim_epochs = get_epochs(stim_raw, True, stim_definition)
        expected_onsets = numpy.array(event_onsets) * sfreq

        self.assertEqual(annotation_epochs.event_id, {'target': 1})
        self.assertEqual(stim_epochs.event_id, {'32': 32})
        numpy.testing.assert_array_equal(
            annotation_epochs.events[:, 0],
            expected_onsets
        )
        numpy.testing.assert_array_equal(
            stim_epochs.events[:, 0],
            expected_onsets
        )
        self.assertEqual(annotation_epochs.get_data().shape, (3, 2, 1001))
        self.assertEqual(stim_epochs.get_data().shape, (3, 2, 1001))
        self.assertTrue(
            all(reason == () for reason in annotation_epochs.drop_log)
        )
        self.assertTrue(all(reason == () for reason in stim_epochs.drop_log))
        numpy.testing.assert_array_equal(
            annotation_epochs.get_data(picks=['Cz']),
            stim_epochs.get_data(picks=['Cz'])
        )

        for epoch_index in range(3):
            _assert_epoch_contains_absolute_samples(
                annotation_epochs,
                epoch_index
            )
            _assert_epoch_contains_absolute_samples(stim_epochs, epoch_index)


class TestMneParity(unittest.TestCase):
    """Compare sEEGnal output directly with the corresponding MNE calls."""

    def test_fixed_overlap_matches_mne(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=10,
            first_samp=1250
        )
        definition = {
            'mode': 'fixed',
            'duration': 2,
            'overlap': 0.5,
            'reject_by_annotation': True,
            'proj': False,
            'id': 7,
        }

        actual = get_epochs(raw, True, definition)
        expected = mne.make_fixed_length_epochs(
            raw,
            duration=2,
            overlap=0.5,
            reject_by_annotation=True,
            proj=False,
            id=7,
            preload=True,
            verbose=False
        )

        _assert_epochs_equal(self, actual, expected)
        self.assertEqual(len(actual), 6)
        self.assertEqual(actual.events[0, 0], 1250)
        self.assertEqual(actual.events[-1, 0], 8750)
        self.assertEqual(actual.get_data().shape, (6, 2, 2000))
        self.assertNotIn(
            'NO_DATA',
            {reason for reasons in actual.drop_log for reason in reasons}
        )

    def test_fixed_recording_too_short_matches_mne_error(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=1,
            first_samp=0
        )
        definition = {
            'mode': 'fixed',
            'duration': 2,
        }

        with self.assertRaisesRegex(ValueError, 'No events produced'):
            get_epochs(raw, True, definition)
        with self.assertRaisesRegex(ValueError, 'No events produced'):
            mne.make_fixed_length_epochs(
                raw,
                duration=2,
                preload=True,
                verbose=False
            )

    def test_event_boundaries_and_drop_log_match_mne(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=10,
            first_samp=1000,
            annotations=(
                (0, 0, 'target'),
                (5, 0, 'target'),
                (9.5, 0, 'target'),
            )
        )
        definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': {'target': 1},
            'tmin': -0.2,
            'tmax': 0.8,
            'baseline': None,
            'reject_by_annotation': False,
        }
        events, _ = mne.events_from_annotations(raw, verbose=False)

        actual = get_epochs(raw, True, definition)
        expected = mne.Epochs(
            raw,
            events,
            event_id={'target': 1},
            tmin=-0.2,
            tmax=0.8,
            baseline=None,
            reject_by_annotation=False,
            preload=True,
            verbose=False
        )

        _assert_epochs_equal(self, actual, expected)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.events[0, 0], 6000)
        self.assertEqual(actual.drop_log, (('NO_DATA',), (), ('TOO_SHORT',)))

    def test_event_rejection_by_annotation_matches_mne(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=12,
            first_samp=0,
            annotations=(
                (2, 0, 'target'),
                (5, 0, 'target'),
                (5.25, 0.1, 'BAD_artifact'),
                (8, 0, 'target'),
            )
        )
        definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': {'target': 1},
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
            'reject_by_annotation': True,
        }
        events, _ = mne.events_from_annotations(raw, verbose=False)

        actual = get_epochs(raw, True, definition)
        expected = mne.Epochs(
            raw,
            events,
            event_id={'target': 1},
            tmin=0,
            tmax=1,
            baseline=None,
            reject_by_annotation=True,
            preload=True,
            verbose=False
        )

        _assert_epochs_equal(self, actual, expected)
        self.assertEqual(actual.drop_log, ((), ('BAD_artifact',), ()))
        numpy.testing.assert_array_equal(
            actual.events[:, 0],
            numpy.array([2000, 8000])
        )

    def test_event_id_list_and_dict_match_mne(self):
        raw = _make_synthetic_raw(
            sfreq=1000,
            duration=10,
            first_samp=0,
            annotations=(
                (2, 0, 'target'),
                (4, 0, 'control'),
                (6, 0, 'target'),
            )
        )
        bids_event_id = {'target': 32, 'control': 64}
        events, _ = mne.events_from_annotations(
            raw,
            event_id=bids_event_id,
            verbose=False
        )

        for requested_event_id in (
            32,
            [32, 64],
            {'target': 32, 'control': 64},
        ):
            with self.subTest(event_id=requested_event_id):
                definition = {
                    'mode': 'events',
                    'event_source': 'annotations',
                    'event_id': requested_event_id,
                    'tmin': 0,
                    'tmax': 0.5,
                    'baseline': None,
                }
                actual = get_epochs(
                    raw,
                    True,
                    definition,
                    bids_event_id=bids_event_id
                )
                expected = mne.Epochs(
                    raw,
                    events,
                    event_id=requested_event_id,
                    tmin=0,
                    tmax=0.5,
                    baseline=None,
                    preload=True,
                    verbose=False
                )

                _assert_epochs_equal(self, actual, expected)


class TestRepeatedEvents(unittest.TestCase):
    """Check all MNE policies for events occurring at the same sample."""

    def setUp(self):
        self.raw = _make_synthetic_raw(
            sfreq=1000,
            duration=6,
            first_samp=0,
            annotations=(
                (2, 0, 'left'),
                (2, 0, 'right'),
            )
        )
        self.event_id = {'left': 1, 'right': 2}
        self.base_definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': self.event_id,
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
        }
        self.events, _ = mne.events_from_annotations(
            self.raw,
            verbose=False
        )

    def test_event_repeated_error(self):
        definition = dict(self.base_definition, event_repeated='error')

        with self.assertRaisesRegex(RuntimeError, 'not unique'):
            get_epochs(self.raw, True, definition)

    def test_event_repeated_drop(self):
        definition = dict(self.base_definition, event_repeated='drop')
        actual = get_epochs(self.raw, True, definition)
        expected = mne.Epochs(
            self.raw,
            self.events,
            event_id=self.event_id,
            tmin=0,
            tmax=1,
            baseline=None,
            event_repeated='drop',
            preload=True,
            verbose=False
        )

        _assert_epochs_equal(self, actual, expected)
        self.assertEqual(len(actual), 1)
        self.assertIn('DROP DUPLICATE', actual.drop_log[1])

    def test_event_repeated_merge(self):
        definition = dict(self.base_definition, event_repeated='merge')
        actual = get_epochs(self.raw, True, definition)
        expected = mne.Epochs(
            self.raw,
            self.events,
            event_id=self.event_id,
            tmin=0,
            tmax=1,
            baseline=None,
            event_repeated='merge',
            preload=True,
            verbose=False
        )

        _assert_epochs_equal(self, actual, expected)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.event_id, {'left/right': 3})
        self.assertIn('MERGE DUPLICATE', actual.drop_log[1])


class TestEpochConfiguration(unittest.TestCase):
    """Reject ambiguous legacy and sEEGnal-managed parameters."""

    def test_legacy_parameters_raise_clear_errors(self):
        raw = _make_synthetic_raw(sfreq=1000, duration=5, first_samp=0)

        for parameter, replacement in (
            ('length', 'duration'),
            ('event_code', 'event_id'),
            ('padding', 'do not support'),
        ):
            with self.subTest(parameter=parameter):
                definition = {'mode': 'fixed', parameter: 1}
                with self.assertRaisesRegex(ValueError, replacement):
                    get_epochs(raw, True, definition)

    def test_preload_is_managed_by_prepare_eeg(self):
        raw = _make_synthetic_raw(sfreq=1000, duration=5, first_samp=0)
        definition = {
            'mode': 'fixed',
            'duration': 1,
            'preload': False,
        }

        with self.assertRaisesRegex(ValueError, 'managed by sEEGnal'):
            get_epochs(raw, True, definition)

    def test_event_source_is_only_valid_in_event_mode(self):
        raw = _make_synthetic_raw(sfreq=1000, duration=5, first_samp=0)
        definition = {
            'mode': 'fixed',
            'duration': 1,
            'event_source': 'annotations',
        }

        with self.assertRaisesRegex(ValueError, 'only valid'):
            get_epochs(raw, True, definition)

    def test_repository_configs_use_mne_fixed_parameters(self):
        repository = Path(__file__).resolve().parents[1]

        for relative_path in (
            'quickstart/init/config.json',
            'dev/init/config.json',
        ):
            with self.subTest(config=relative_path):
                config = json.loads(
                    (repository / relative_path).read_text(encoding='utf-8')
                )
                definitions = []

                def collect_epoch_definitions(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if key == 'epoch_definition':
                                definitions.append(child)
                            collect_epoch_definitions(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_epoch_definitions(child)

                collect_epoch_definitions(config)

                self.assertGreater(len(definitions), 0)
                for definition in definitions:
                    self.assertEqual(definition['mode'], 'fixed')
                    self.assertIn('duration', definition)
                    self.assertNotIn('length', definition)
                    self.assertNotIn('padding', definition)
                    self.assertIsInstance(
                        definition['reject_by_annotation'],
                        bool
                    )


if __name__ == '__main__':
    unittest.main()
