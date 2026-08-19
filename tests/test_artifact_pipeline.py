"""End-to-end regression tests for the artifact annotation pipeline."""

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import mne_bids
import numpy

from sEEGnal.preprocess.artifact_detection import (
    estimate_artifact_components,
    muscle_detection as build_muscle_annotations,
)
from sEEGnal.preprocess.badchannel_detection import (
    estimate_badchannel_component,
)
from sEEGnal.preprocess.find_artifacts import (
    _get_original_n_times,
    _local_samples_to_original,
)
from sEEGnal.tools.bids_tools import write_annotations
from sEEGnal.tools.mne_tools import (
    fit_ica,
    get_unannotated_raw,
    prepare_eeg,
)
from sEEGnal.tools.qc_tools import (
    _get_welch_segment_length,
    plot_bad_epochs,
    plot_components_type,
    plot_occipital_power_spectrum,
)
from tests._mne_test_data import (
    assert_epoch_contains_samples,
    make_synthetic_raw,
)


class TestArtifactPipelineEndToEnd(unittest.TestCase):
    """Follow an artifact from cropped detection data into rejected epochs."""

    def test_cropped_artifact_round_trip_rejects_the_expected_fixed_epoch(self):
        original_raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=1000,
        )
        detection_raw = prepare_eeg(
            {},
            None,
            raw=original_raw.copy(),
            preload=True,
            crop_seconds=2,
        )

        # Local sample 5250 of this crop is sample offset 7250 from the
        # original recording zero, independently of MNE's first_samp=1000.
        artifact_indices = _local_samples_to_original(
            detection_raw,
            numpy.array([5250]),
        )
        original_n_times = _get_original_n_times(detection_raw)
        numpy.testing.assert_array_equal(artifact_indices, [7250])
        self.assertEqual(original_n_times, 20000)

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {'subsystem': 'preprocess'}
            bids_path = mne_bids.BIDSPath(
                subject='023',
                session='0',
                task='1EC',
                datatype='eeg',
                suffix='eeg',
                extension='.vhdr',
                root=Path(temporary_directory),
            )

            with patch(
                'sEEGnal.preprocess.artifact_detection.'
                'find_artifacts.muscle_detection',
                return_value=(artifact_indices, original_n_times, 1000),
            ):
                annotations = build_muscle_annotations(config, bids_path)

            numpy.testing.assert_allclose(annotations.onset, [7.25])
            numpy.testing.assert_allclose(annotations.duration, [0.5])
            numpy.testing.assert_array_equal(
                annotations.description,
                ['bad_muscle'],
            )
            write_annotations(config, bids_path, annotations)

            loaded_raw = prepare_eeg(
                config,
                bids_path,
                raw=original_raw.copy(),
                preload=True,
                crop_seconds=5,
                set_annotations=True,
            )
            epochs = prepare_eeg(
                config,
                bids_path,
                raw=original_raw.copy(),
                preload=True,
                crop_seconds=5,
                set_annotations=True,
                epoch_definition={
                    'mode': 'fixed',
                    'duration': 1,
                    'overlap': 0,
                    'reject_by_annotation': True,
                },
            )

        events, event_id = mne.events_from_annotations(
            loaded_raw,
            regexp=None,
            verbose=False,
        )
        self.assertEqual(event_id, {'bad_muscle': 1})
        numpy.testing.assert_array_equal(events[:, 0], [8250])
        numpy.testing.assert_allclose(loaded_raw.annotations.onset, [8.25])
        numpy.testing.assert_allclose(loaded_raw.annotations.duration, [0.5])

        expected_events = numpy.delete(
            numpy.arange(6000, 16000, 1000),
            2,
        )
        self.assertEqual(len(epochs), 9)
        self.assertEqual(epochs.get_data().shape, (9, 2, 1000))
        numpy.testing.assert_array_equal(epochs.events[:, 0], expected_events)
        self.assertEqual(
            epochs.drop_log,
            ((), (), ('bad_muscle',), (), (), (), (), (), (), ()),
        )
        self.assertNotIn(
            'NO_DATA',
            {reason for reasons in epochs.drop_log for reason in reasons},
        )
        self.assertEqual(epochs.original_first_samp, 1000)
        self.assertEqual(epochs.original_last_samp, 20999)
        assert_epoch_contains_samples(epochs, epoch_index=0)
        assert_epoch_contains_samples(epochs, epoch_index=-1)


class TestComponentChannelHandling(unittest.TestCase):
    """Keep bad-channel and artifact ICA channel policies distinct."""

    def _make_config(self):
        return {
            'component_estimation': {
                'low_freq': 1,
                'high_freq': 100,
                'resample_frequency': 250,
                'crop_seconds': 5,
                'unclear_threshold': 0.7,
            },
            'global': {
                'channels_to_include': ['all'],
                'channels_to_exclude': [],
            },
            'preprocess': {
                'badchannel_detection': {
                    'crop_seconds': 5,
                },
            },
        }

    def _make_ica(self):
        return SimpleNamespace(
            labels_={
                'brain': [0],
                'muscle': [1],
                'eog': [],
                'ecg': [],
                'line_noise': [],
                'ch_noise': [],
                'other': [],
            },
        )

    def _make_label_scores(self):
        scores = numpy.full((2, 7), 0.01)
        scores[0, 0] = 0.94
        scores[1, 1] = 0.93
        return scores

    def test_artifact_ica_excludes_bad_channels_for_both_fits(self):
        for ica_desc, expected_annotations in (
            ('artifacts', False),
            ('cleaning', True),
        ):
            with self.subTest(ica_desc=ica_desc):
                raw = object()
                classification_raw = object()
                ica = self._make_ica()

                with (
                    patch(
                        'sEEGnal.preprocess.artifact_detection.'
                        'mne_tools.prepare_eeg',
                        return_value=raw,
                    ) as prepare_eeg_mock,
                    patch(
                        'sEEGnal.preprocess.artifact_detection.'
                        'mne_tools.fit_ica',
                        return_value=ica,
                    ) as fit_ica_mock,
                    patch(
                        'sEEGnal.preprocess.artifact_detection.'
                        'mne_tools.get_unannotated_raw',
                        return_value=classification_raw,
                    ) as get_unannotated_raw_mock,
                    patch(
                        'sEEGnal.preprocess.artifact_detection.'
                        'iclabel.iclabel_label_components',
                        return_value=self._make_label_scores(),
                    ) as iclabel_mock,
                    patch(
                        'sEEGnal.preprocess.artifact_detection.bids.write_ica',
                    ),
                ):
                    estimate_artifact_components(
                        self._make_config(),
                        BIDS=None,
                        ica_desc=ica_desc,
                    )

                parameters = prepare_eeg_mock.call_args.kwargs
                self.assertIs(parameters['metadata_badchannels'], True)
                self.assertIs(parameters['exclude_badchannels'], True)
                self.assertIs(
                    parameters.get('interpolate_badchannels', False),
                    False,
                )
                self.assertIs(
                    parameters['set_annotations'],
                    expected_annotations,
                )
                self.assertNotIn('epoch_definition', parameters)
                fit_ica_mock.assert_called_once_with(
                    raw,
                    reject_by_annotation=True,
                )
                get_unannotated_raw_mock.assert_called_once_with(raw)
                iclabel_mock.assert_called_once_with(
                    classification_raw,
                    ica,
                    inplace=True,
                    backend='onnx',
                )

    def test_badchannel_ica_does_not_use_existing_bad_metadata(self):
        raw = object()
        classification_raw = object()
        ica = self._make_ica()

        with (
            patch(
                'sEEGnal.preprocess.badchannel_detection.'
                'mne_tools.prepare_eeg',
                return_value=raw,
            ) as prepare_eeg_mock,
            patch(
                'sEEGnal.preprocess.badchannel_detection.mne_tools.fit_ica',
                return_value=ica,
            ) as fit_ica_mock,
            patch(
                'sEEGnal.preprocess.badchannel_detection.'
                'mne_tools.get_unannotated_raw',
                return_value=classification_raw,
            ) as get_unannotated_raw_mock,
            patch(
                'sEEGnal.preprocess.badchannel_detection.'
                'iclabel.iclabel_label_components',
                return_value=self._make_label_scores(),
            ) as iclabel_mock,
            patch(
                'sEEGnal.preprocess.badchannel_detection.bids.write_ica',
            ),
        ):
            estimate_badchannel_component(
                self._make_config(),
                BIDS=None,
            )

        parameters = prepare_eeg_mock.call_args.kwargs
        for parameter in (
            'metadata_badchannels',
            'exclude_badchannels',
            'interpolate_badchannels',
            'set_annotations',
            'epoch_definition',
        ):
            self.assertIs(parameters.get(parameter, False), False)
        fit_ica_mock.assert_called_once_with(
            raw,
            reject_by_annotation=True,
        )
        get_unannotated_raw_mock.assert_called_once_with(raw)
        iclabel_mock.assert_called_once_with(
            classification_raw,
            ica,
            inplace=True,
            backend='onnx',
        )


class TestIcaRawHandling(unittest.TestCase):
    """Keep ICA fitting and ICLabel on the same unannotated Raw samples."""

    def test_get_unannotated_raw_omits_only_bad_intervals(self):
        raw = make_synthetic_raw(
            sfreq=100,
            duration=10,
            first_samp=0,
            annotations=(
                (2, 1.5, 'bad_muscle'),
                (6, 1, 'task_event'),
            ),
        )
        raw = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            rereference='average',
        )
        reference_state = raw._seegnal_reference

        expected_data = raw.get_data(reject_by_annotation='omit')
        unannotated_raw = get_unannotated_raw(raw)

        self.assertIsInstance(unannotated_raw, mne.io.BaseRaw)
        self.assertEqual(unannotated_raw.ch_names, raw.ch_names)
        self.assertEqual(unannotated_raw.info['bads'], raw.info['bads'])
        self.assertEqual(len(unannotated_raw.annotations), 0)
        self.assertTrue(unannotated_raw.proj)
        self.assertTrue(all(
            projector['active']
            for projector in unannotated_raw.info['projs']
        ))
        numpy.testing.assert_array_equal(
            unannotated_raw.get_data(),
            expected_data,
        )
        self.assertEqual(
            unannotated_raw._seegnal_reference,
            reference_state,
        )
        self.assertIsNot(
            unannotated_raw._seegnal_reference,
            reference_state,
        )
        self.assertIsNot(
            unannotated_raw._seegnal_reference['channels'],
            reference_state['channels'],
        )
        unannotated_raw._seegnal_reference['channels'].append('Pz')
        self.assertEqual(reference_state['channels'], ['Cz'])

    def test_get_unannotated_raw_reuses_raw_without_bad_annotations(self):
        raw = make_synthetic_raw(
            sfreq=100,
            duration=2,
            first_samp=0,
            annotations=((0.5, 0.1, 'task_event'),),
        )

        self.assertIs(get_unannotated_raw(raw), raw)

    def test_get_unannotated_raw_rejects_fully_annotated_recording(self):
        raw = make_synthetic_raw(
            sfreq=100,
            duration=2,
            first_samp=0,
            annotations=((0, 2, 'BAD_all'),),
        )

        with self.assertRaisesRegex(ValueError, 'complete recording'):
            get_unannotated_raw(raw)

    def test_fit_ica_forwards_annotation_rejection_to_mne(self):
        raw = make_synthetic_raw(
            sfreq=100,
            duration=2,
            first_samp=0,
        )
        raw = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            rereference='average',
        )
        reference_state = raw._seegnal_reference

        with patch(
            'sEEGnal.tools.mne_tools.mne.preprocessing.ICA'
        ) as ica_class_mock:
            fitted = fit_ica(raw, reject_by_annotation=True)

        ica = ica_class_mock.return_value
        self.assertIs(fitted, ica)
        ica.fit.assert_called_once_with(
            raw,
            picks='eeg',
            reject_by_annotation=True,
        )
        self.assertEqual(ica._seegnal_reference, reference_state)
        self.assertIsNot(ica._seegnal_reference, reference_state)
        self.assertIsNot(
            ica._seegnal_reference['channels'],
            reference_state['channels'],
        )


class TestArtifactQcWelch(unittest.TestCase):
    """Use one supported Welch window across all good Raw spans."""

    def test_shortest_good_span_sets_welch_window_without_warning(self):
        raw = make_synthetic_raw(
            sfreq=500,
            duration=10,
            first_samp=0,
            annotations=((2, 1, 'BAD_artifact'),),
        )

        n_fft = 2048
        n_per_seg = _get_welch_segment_length(raw, n_fft)

        self.assertEqual(n_per_seg, 1000)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            raw.compute_psd(
                method='welch',
                fmin=2,
                fmax=45,
                picks='eeg',
                reject_by_annotation=True,
                n_fft=n_fft,
                n_per_seg=n_per_seg,
            )

        nperseg_warnings = [
            warning
            for warning in caught_warnings
            if 'nperseg' in str(warning.message)
        ]
        self.assertEqual(nperseg_warnings, [])


class TestArtifactQcBadChannelHandling(unittest.TestCase):
    """Exclude, rather than interpolate, bad sensors from QC calculations."""

    def setUp(self):
        self.config = {
            'global': {
                'channels_to_include': ['all'],
                'channels_to_exclude': [],
            },
            'component_estimation': {
                'low_freq': 2,
                'high_freq': 45,
                'crop_seconds': 1,
                'resample_frequency': 250,
            },
            'source_reconstruction': {
                'epoch_definition': {
                    'mode': 'fixed',
                    'duration': 2,
                    'overlap': 0,
                    'reject_by_annotation': True,
                },
            },
        }

    def assert_bad_channels_are_excluded(self, call):
        self.assertTrue(call.kwargs['metadata_badchannels'])
        self.assertTrue(call.kwargs['exclude_badchannels'])
        self.assertFalse(call.kwargs['interpolate_badchannels'])

    def test_component_qc_excludes_bad_channels(self):
        with patch(
            'sEEGnal.tools.qc_tools.prepare_eeg',
            side_effect=RuntimeError('stop after prepare_eeg'),
        ) as prepare_mock:
            with self.assertRaisesRegex(RuntimeError, 'stop after prepare_eeg'):
                plot_components_type(self.config, None)

        self.assert_bad_channels_are_excluded(prepare_mock.call_args)

    def test_occipital_qc_excludes_bad_channels_before_ica(self):
        prepared_raw = object()
        with (
            patch(
                'sEEGnal.tools.qc_tools.prepare_eeg',
                side_effect=[
                    prepared_raw,
                    RuntimeError('stop after prepare_eeg'),
                ],
            ) as prepare_mock,
            patch(
                'sEEGnal.tools.qc_tools.apply_ica',
                return_value=prepared_raw,
            ) as apply_ica_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, 'stop after prepare_eeg'):
                plot_occipital_power_spectrum(self.config, None)

        self.assertEqual(prepare_mock.call_count, 2)
        first_call, second_call = prepare_mock.call_args_list
        self.assert_bad_channels_are_excluded(first_call)
        self.assert_bad_channels_are_excluded(second_call)
        self.assertIs(second_call.kwargs['raw'], prepared_raw)
        apply_ica_mock.assert_called_once()
        self.assertIs(apply_ica_mock.call_args.args[2], prepared_raw)

    def test_bad_epoch_qc_excludes_bad_channels(self):
        with patch(
            'sEEGnal.tools.qc_tools.prepare_eeg',
            side_effect=RuntimeError('stop after prepare_eeg'),
        ) as prepare_mock:
            with self.assertRaisesRegex(RuntimeError, 'stop after prepare_eeg'):
                plot_bad_epochs(self.config, None)

        self.assert_bad_channels_are_excluded(prepare_mock.call_args)


if __name__ == '__main__':
    unittest.main()
