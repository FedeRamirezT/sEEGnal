"""Tests for artifact sample-coordinate conversions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from sEEGnal.preprocess.find_artifacts import (
    EOG_detection,
    _absolute_samples_to_original,
    _get_original_n_times,
    _get_original_sample_bounds,
    _local_samples_to_original,
    muscle_detection,
    other_detection,
    sensor_detection,
)
from sEEGnal.tools.mne_tools import prepare_eeg
from tests._mne_test_data import make_synthetic_raw


class TestOriginalSampleCoordinates(unittest.TestCase):
    """Check the common original-recording coordinate system."""

    def test_uncropped_raw_with_zero_first_sample(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=10,
            first_samp=0
        )

        self.assertEqual(_get_original_sample_bounds(raw), (0, 9999))
        self.assertEqual(_get_original_n_times(raw), 10000)
        numpy.testing.assert_array_equal(
            _local_samples_to_original(raw, [0, 500, 9999]),
            numpy.array([0, 500, 9999])
        )
        numpy.testing.assert_array_equal(
            _absolute_samples_to_original(raw, [0, 500, 9999]),
            numpy.array([0, 500, 9999])
        )

    def test_uncropped_raw_with_nonzero_first_sample(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=2,
            first_samp=1250
        )

        self.assertEqual(_get_original_sample_bounds(raw), (1250, 3249))
        self.assertEqual(_get_original_n_times(raw), 2000)
        numpy.testing.assert_array_equal(
            _local_samples_to_original(raw, [0, 500, 1999]),
            numpy.array([0, 500, 1999])
        )
        numpy.testing.assert_array_equal(
            _absolute_samples_to_original(raw, [1250, 1750, 3249]),
            numpy.array([0, 500, 1999])
        )

    def test_cropped_raw_uses_stored_original_bounds(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=1000
        )
        cropped = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=2
        )

        self.assertEqual(cropped.first_samp, 3000)
        self.assertEqual(cropped.last_samp, 18999)
        self.assertEqual(_get_original_sample_bounds(cropped), (1000, 20999))
        self.assertEqual(_get_original_n_times(cropped), 20000)

        expected = numpy.array([2000, 2500, 17999])
        numpy.testing.assert_array_equal(
            _local_samples_to_original(cropped, [0, 500, 15999]),
            expected
        )
        numpy.testing.assert_array_equal(
            _absolute_samples_to_original(cropped, [3000, 3500, 18999]),
            expected
        )

    def test_successive_crops_keep_accumulated_offset(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=1000
        )
        first_crop = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=2
        )
        second_crop = prepare_eeg(
            {},
            None,
            raw=first_crop,
            preload=True,
            crop_seconds=1
        )

        self.assertEqual(second_crop.first_samp, 4000)
        self.assertEqual(second_crop.last_samp, 17999)
        self.assertEqual(
            _get_original_sample_bounds(second_crop),
            (1000, 20999)
        )
        self.assertEqual(_get_original_n_times(second_crop), 20000)

        expected = numpy.array([3000, 3500, 16999])
        numpy.testing.assert_array_equal(
            _local_samples_to_original(second_crop, [0, 500, 13999]),
            expected
        )
        numpy.testing.assert_array_equal(
            _absolute_samples_to_original(
                second_crop,
                [4000, 4500, 17999]
            ),
            expected
        )

    def test_empty_indices_are_supported(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=2,
            first_samp=1000
        )

        local = _local_samples_to_original(raw, [])
        absolute = _absolute_samples_to_original(raw, [])

        self.assertEqual(local.dtype, numpy.dtype(int))
        self.assertEqual(absolute.dtype, numpy.dtype(int))
        self.assertEqual(local.size, 0)
        self.assertEqual(absolute.size, 0)

    def test_invalid_indices_raise_clear_errors(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=2,
            first_samp=1000
        )

        with self.assertRaisesRegex(TypeError, 'integer sample indices'):
            _local_samples_to_original(raw, [0.5])

        for indices in ([-1], [2000]):
            with self.subTest(local_indices=indices):
                with self.assertRaisesRegex(ValueError, 'current Raw'):
                    _local_samples_to_original(raw, indices)

        for indices in ([999], [3000]):
            with self.subTest(absolute_indices=indices):
                with self.assertRaisesRegex(ValueError, 'current Raw'):
                    _absolute_samples_to_original(raw, indices)

    def test_incomplete_original_bounds_are_rejected(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=2,
            first_samp=1000
        )
        raw.original_last_samp = raw.last_samp

        with self.assertRaisesRegex(ValueError, 'must be stored together'):
            _get_original_sample_bounds(raw)

    def test_non_raw_input_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'MNE Raw'):
            _local_samples_to_original(object(), [0])

        with self.assertRaisesRegex(TypeError, 'MNE Raw'):
            _absolute_samples_to_original(object(), [0])


class TestDetectorSampleCoordinates(unittest.TestCase):
    """Check that every detector returns original-recording offsets."""

    def _make_config(self, cropped):
        return {
            'component_estimation': {
                'crop_seconds': 1 if cropped else False,
                'resample_frequency': 1000,
            },
            'global': {
                'channels_to_include': ['all'],
                'channels_to_exclude': [],
            },
            'preprocess': {
                'artifact_detection': {
                    'frontal_channels': ['Cz'],
                    'EOG': {
                        'low_freq': 1,
                        'high_freq': 10,
                        'threshold': 2,
                    },
                    'muscle': {
                        'low_freq': 20,
                        'high_freq': 40,
                        'threshold': 2,
                    },
                    'sensor': {
                        'low_freq': 1,
                        'high_freq': 40,
                        'threshold': 1.5,
                        'epoch_definition': {
                            'mode': 'fixed',
                            'duration': 1,
                        },
                    },
                    'other': {
                        'low_freq': 1,
                        'high_freq': 40,
                        'threshold': 5,
                        'epoch_definition': {
                            'mode': 'fixed',
                            'duration': 1,
                        },
                    },
                },
            },
        }

    def _make_raw(self, cropped):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=1000,
        )

        if cropped:
            raw = prepare_eeg(
                {},
                None,
                raw=raw,
                preload=True,
                crop_seconds=2,
            )
            raw = prepare_eeg(
                {},
                None,
                raw=raw,
                preload=True,
                crop_seconds=1,
            )

        original_first, _ = _get_original_sample_bounds(raw)
        crop_offset = raw.first_samp - original_first

        return raw, crop_offset

    def _assert_detection_result(self, result, expected_index):
        indices, n_times, sfreq = result

        numpy.testing.assert_array_equal(
            indices,
            numpy.array([expected_index])
        )
        self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))
        self.assertEqual(n_times, 20000)
        self.assertEqual(sfreq, 1000)

    def _assert_bad_channels_are_excluded(self, prepare_call):
        parameters = prepare_call.kwargs

        self.assertIs(parameters['metadata_badchannels'], True)
        self.assertIs(parameters['exclude_badchannels'], True)
        self.assertIs(
            parameters.get('interpolate_badchannels', False),
            False,
        )

    def test_eog_indices_are_original_offsets_with_and_without_crop(self):
        for cropped in (False, True):
            with self.subTest(cropped=cropped):
                raw, crop_offset = self._make_raw(cropped)
                raw._data[0] = 0
                raw._data[0, 250] = 10

                with (
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.prepare_eeg',
                        return_value=raw,
                    ) as prepare_eeg_mock,
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.apply_ica',
                        return_value=raw,
                    ),
                ):
                    result = EOG_detection(
                        self._make_config(cropped),
                        BIDS=None,
                    )

                self._assert_bad_channels_are_excluded(
                    prepare_eeg_mock.call_args_list[0],
                )
                self._assert_detection_result(result, crop_offset + 250)

    def test_muscle_indices_are_original_offsets_with_and_without_crop(self):
        for cropped in (False, True):
            with self.subTest(cropped=cropped):
                raw, crop_offset = self._make_raw(cropped)

                with (
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.prepare_eeg',
                        return_value=raw,
                    ) as prepare_eeg_mock,
                    patch(
                        'sEEGnal.preprocess.find_artifacts.find_peaks',
                        return_value=(numpy.array([250]), {}),
                    ),
                ):
                    result = muscle_detection(
                        self._make_config(cropped),
                        BIDS=None,
                    )

                self._assert_bad_channels_are_excluded(
                    prepare_eeg_mock.call_args,
                )
                self._assert_detection_result(result, crop_offset + 250)

    def test_sensor_indices_are_original_offsets_with_and_without_crop(self):
        epoch_data = numpy.array([
            [[10, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ])

        for cropped in (False, True):
            with self.subTest(cropped=cropped):
                raw, crop_offset = self._make_raw(cropped)
                epochs = SimpleNamespace(
                    events=numpy.array([
                        [raw.first_samp + 250, 0, 1],
                        [raw.first_samp + 500, 0, 1],
                    ]),
                    info={'sfreq': 1000},
                    get_data=lambda: epoch_data,
                )

                with (
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.prepare_eeg',
                        side_effect=[raw, epochs],
                    ) as prepare_eeg_mock,
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.apply_ica',
                        return_value=raw,
                    ),
                ):
                    result = sensor_detection(
                        self._make_config(cropped),
                        BIDS=None,
                        ica_desc='cleaning',
                    )

                self._assert_bad_channels_are_excluded(
                    prepare_eeg_mock.call_args_list[0],
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[0].kwargs[
                        'set_annotations'
                    ],
                    True,
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[1].kwargs['raw'],
                    raw,
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[1].kwargs[
                        'epoch_definition'
                    ]['reject_by_annotation'],
                    True,
                )
                self._assert_detection_result(result, crop_offset + 250)

    def test_other_indices_are_original_offsets_with_and_without_crop(self):
        epoch_data = numpy.array([
            [[0, 0, 10, 0]],
            [[0, 0, 0, 0]],
        ])

        for cropped in (False, True):
            with self.subTest(cropped=cropped):
                raw, crop_offset = self._make_raw(cropped)
                epochs = SimpleNamespace(
                    events=numpy.array([
                        [raw.first_samp + 250, 0, 1],
                        [raw.first_samp + 500, 0, 1],
                    ]),
                    info={'sfreq': 1000},
                    get_data=lambda: epoch_data,
                )

                with (
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.prepare_eeg',
                        side_effect=[raw, epochs],
                    ) as prepare_eeg_mock,
                    patch(
                        'sEEGnal.preprocess.find_artifacts.'
                        'mne_tools.apply_ica',
                        return_value=raw,
                    ),
                ):
                    result = other_detection(
                        self._make_config(cropped),
                        BIDS=None,
                        ica_desc='cleaning',
                    )

                self._assert_bad_channels_are_excluded(
                    prepare_eeg_mock.call_args_list[0],
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[0].kwargs[
                        'set_annotations'
                    ],
                    True,
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[1].kwargs['raw'],
                    raw,
                )
                self.assertIs(
                    prepare_eeg_mock.call_args_list[1].kwargs[
                        'epoch_definition'
                    ]['reject_by_annotation'],
                    True,
                )
                self._assert_detection_result(result, crop_offset + 250)


if __name__ == '__main__':
    unittest.main()
