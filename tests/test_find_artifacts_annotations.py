"""Tests for converting detected artifact peaks into annotations."""

import unittest

import numpy

from sEEGnal.preprocess.find_artifacts import (
    create_annotations,
    merge_peaks,
)


class TestMergePeaks(unittest.TestCase):
    """Check half-open artifact intervals and their recording boundaries."""

    def test_empty_peaks_return_integer_arrays(self):
        onsets, durations = merge_peaks([], 100, 10)

        self.assertEqual(onsets.dtype, numpy.dtype(int))
        self.assertEqual(durations.dtype, numpy.dtype(int))
        self.assertEqual(onsets.size, 0)
        self.assertEqual(durations.size, 0)

    def test_single_and_separated_peaks_keep_independent_intervals(self):
        onsets, durations = merge_peaks([10, 40], 100, 10)

        numpy.testing.assert_array_equal(onsets, numpy.array([10, 40]))
        numpy.testing.assert_array_equal(durations, numpy.array([10, 10]))

    def test_unsorted_duplicate_touching_and_overlapping_peaks_merge(self):
        onsets, durations = merge_peaks(
            [22, 10, 20, 15, 10],
            100,
            5
        )

        numpy.testing.assert_array_equal(onsets, numpy.array([10]))
        numpy.testing.assert_array_equal(durations, numpy.array([17]))

    def test_first_and_last_samples_are_preserved(self):
        onsets, durations = merge_peaks([0, 99], 100, 10)

        numpy.testing.assert_array_equal(onsets, numpy.array([0, 99]))
        numpy.testing.assert_array_equal(durations, numpy.array([10, 1]))

    def test_interval_is_clipped_to_recording_end(self):
        onsets, durations = merge_peaks([50], 100, 150)

        numpy.testing.assert_array_equal(onsets, numpy.array([50]))
        numpy.testing.assert_array_equal(durations, numpy.array([50]))

    def test_long_overlapping_intervals_cover_recording_once(self):
        onsets, durations = merge_peaks([0, 50], 100, 150)

        numpy.testing.assert_array_equal(onsets, numpy.array([0]))
        numpy.testing.assert_array_equal(durations, numpy.array([100]))

    def test_invalid_boundaries_and_indices_raise(self):
        for n_times in (0, -1):
            with self.subTest(n_times=n_times):
                with self.assertRaisesRegex(ValueError, 'n_times'):
                    merge_peaks([], n_times, 10)

        with self.assertRaisesRegex(TypeError, 'n_times'):
            merge_peaks([], 100.0, 10)

        for duration in (0, -1):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(
                    ValueError,
                    'artifact_duration_samples'
                ):
                    merge_peaks([], 100, duration)

        with self.assertRaisesRegex(TypeError, 'artifact_duration_samples'):
            merge_peaks([], 100, 10.0)

        for peaks in ([-1], [100]):
            with self.subTest(peaks=peaks):
                with self.assertRaisesRegex(ValueError, 'original recording'):
                    merge_peaks(peaks, 100, 10)

        with self.assertRaisesRegex(TypeError, 'integer sample indices'):
            merge_peaks([1.5], 100, 10)


class TestCreateAnnotations(unittest.TestCase):
    """Check sample-to-second conversion and MNE annotation construction."""

    def test_peaks_are_converted_to_seconds(self):
        annotations = create_annotations(
            peaks_index=[1000, 3000],
            n_times=5000,
            sfreq=1000,
            annotation_description='bad_muscle',
            fictional_artifact_duration=0.5
        )

        numpy.testing.assert_allclose(annotations.onset, [1.0, 3.0])
        numpy.testing.assert_allclose(annotations.duration, [0.5, 0.5])
        numpy.testing.assert_array_equal(
            annotations.description,
            ['bad_muscle', 'bad_muscle']
        )
        self.assertIsNone(annotations.orig_time)

    def test_touching_intervals_are_one_annotation(self):
        annotations = create_annotations(
            peaks_index=[100, 600],
            n_times=2000,
            sfreq=1000,
            annotation_description='bad_EOG',
            fictional_artifact_duration=0.5
        )

        self.assertEqual(len(annotations), 1)
        self.assertAlmostEqual(annotations.onset[0], 0.1)
        self.assertAlmostEqual(annotations.duration[0], 1.0)

    def test_duration_is_rounded_to_nearest_sample(self):
        annotations = create_annotations(
            peaks_index=[10],
            n_times=100,
            sfreq=1000,
            annotation_description='bad_other',
            fictional_artifact_duration=0.0016
        )

        self.assertAlmostEqual(annotations.onset[0], 0.01)
        self.assertAlmostEqual(annotations.duration[0], 0.002)

    def test_annotations_are_clipped_at_both_recording_boundaries(self):
        annotations = create_annotations(
            peaks_index=[0, 999],
            n_times=1000,
            sfreq=1000,
            annotation_description='bad_jump',
            fictional_artifact_duration=0.5
        )

        numpy.testing.assert_allclose(annotations.onset, [0, 0.999])
        numpy.testing.assert_allclose(annotations.duration, [0.5, 0.001])

    def test_empty_peaks_create_empty_annotations(self):
        annotations = create_annotations(
            peaks_index=[],
            n_times=1000,
            sfreq=1000,
            annotation_description='bad_other'
        )

        self.assertEqual(len(annotations), 0)
        self.assertIsNone(annotations.orig_time)

    def test_invalid_sampling_and_duration_parameters_raise(self):
        common = {
            'peaks_index': [10],
            'n_times': 100,
            'annotation_description': 'bad_other',
        }

        for sfreq in (0, -1000, numpy.inf):
            with self.subTest(sfreq=sfreq):
                with self.assertRaisesRegex(ValueError, 'sfreq'):
                    create_annotations(sfreq=sfreq, **common)

        with self.assertRaisesRegex(TypeError, 'sfreq'):
            create_annotations(sfreq='1000', **common)

        for duration in (0, -0.5, numpy.inf):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(
                    ValueError,
                    'fictional_artifact_duration'
                ):
                    create_annotations(
                        sfreq=1000,
                        fictional_artifact_duration=duration,
                        **common
                    )

        with self.assertRaisesRegex(TypeError, 'fictional_artifact_duration'):
            create_annotations(
                sfreq=1000,
                fictional_artifact_duration='0.5',
                **common
            )

        with self.assertRaisesRegex(ValueError, 'at least one sample'):
            create_annotations(
                sfreq=1000,
                fictional_artifact_duration=0.0001,
                **common
            )

        with self.assertRaisesRegex(TypeError, 'annotation_description'):
            create_annotations(
                sfreq=1000,
                annotation_description=1,
                peaks_index=[10],
                n_times=100
            )


if __name__ == '__main__':
    unittest.main()
