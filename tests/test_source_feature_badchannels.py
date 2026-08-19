"""Tests for excluding bad sensors while preserving finite source outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from sEEGnal.feature_extraction.FC.estimate_ciplv import estimate_ciplv
from sEEGnal.feature_extraction.FC.estimate_plv import estimate_plv
from sEEGnal.feature_extraction.PSD.estimate_relative_power_spectrum import (
    estimate_relative_power_spectrum,
)
from sEEGnal.sources_reconstruction.inverse import estimate_lcmv
from sEEGnal.tools.bids_tools import (
    read_ciplv,
    read_inverse_solution,
    read_plv,
    read_relative_power_spectrum,
    write_ciplv,
    write_inverse_solution,
    write_plv,
    write_relative_power_spectrum,
)
from sEEGnal.tools.feature_tools import (
    get_source_metadata,
    require_finite_source_values,
)


def _average_reference_state():
    """Return fresh effective-reference metadata for two good channels."""
    return {
        'schema_version': 1,
        'method': 'average',
        'implementation': 'mne_projection',
        'channels': ['A', 'C'],
        'status': 'effective',
    }


class _FakeBeamformer(dict):
    """Minimal dictionary-like beamformer supporting the save contract."""

    def save(self, path, overwrite=False):
        """Record the requested path without serializing filter arrays."""
        self.saved_path = Path(path)
        self.saved_overwrite = overwrite


class _FakeSourceEstimate:
    def __init__(self, data=None):
        if data is None:
            data = numpy.asarray(
                [
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6],
                ],
                dtype=numpy.float32,
            )
        self.data = numpy.asarray(data)
        self.vertices = [numpy.asarray([10, 11]), numpy.asarray([20])]
        self.subject = 'fsaverage'

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return _FakeSourceEstimate(self.data.copy())

    def filter(self, *args, **kwargs):
        return self


class TestSourceMetadata(unittest.TestCase):
    def setUp(self):
        self.filters = {
            'ch_names': ['A', 'C'],
            'src_type': 'surface',
            'subject': 'fsaverage',
        }

    def test_source_metadata_maps_input_sensors_and_vertices(self):
        metadata = get_source_metadata(
            ['A', 'B', 'C'],
            ['B'],
            self.filters,
            _FakeSourceEstimate(),
            source_spacing='ico2',
        )

        self.assertEqual(metadata['input_good_channels'], ['A', 'C'])
        self.assertEqual(metadata['input_good_channel_indices'], [0, 2])
        self.assertEqual(metadata['input_bad_channels'], ['B'])
        self.assertEqual(metadata['input_bad_channel_indices'], [1])
        self.assertEqual(metadata['input_excluded_channels'], ['B'])
        self.assertEqual(metadata['source_vertex_indices'], [0, 1, 2])
        self.assertEqual(metadata['source_vertex_numbers'], [10, 11, 20])
        self.assertEqual(metadata['source_vertex_spaces'], ['lh', 'lh', 'rh'])
        self.assertEqual(metadata['n_sources'], 3)
        self.assertEqual(metadata['source_subject'], 'fsaverage')
        self.assertEqual(metadata['source_spacing'], 'ico2')
        self.assertEqual(metadata['source_value_policy'], 'finite')

    def test_nonfinite_source_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite source values'):
            require_finite_source_values(
                numpy.asarray([1, numpy.nan]),
                'Test source output',
            )

    def test_source_connectivity_metadata_round_trip(self):
        config = {
            'current_space': 'source',
            'subsystem': 'feature_extraction',
        }
        values = numpy.asarray([0.1, 0.2, 0.3], dtype=numpy.float32)
        metadata = self._connectivity_metadata()

        cases = (
            ('plv', write_plv, read_plv),
            ('ciplv', write_ciplv, read_ciplv),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            for value_name, writer, reader in cases:
                with self.subTest(value_name=value_name):
                    path = Path(temporary_directory) / f'{value_name}.h5'
                    with patch(
                        'sEEGnal.tools.bids_tools.build_derivatives_path',
                        return_value=path,
                    ):
                        writer.__wrapped__(
                            config,
                            BIDS=None,
                            **{
                                value_name: values,
                                'metadata': metadata,
                            },
                        )
                        loaded_values, loaded_metadata = reader(
                            config,
                            BIDS=None,
                            space='source',
                            band_name='alpha',
                        )

                    numpy.testing.assert_array_equal(loaded_values, values)
                    self._assert_loaded_source_metadata(loaded_metadata)

    def test_source_power_metadata_round_trip(self):
        config = {
            'current_space': 'source',
            'subsystem': 'feature_extraction',
            'feature_extraction': {
                'relative_power_spectrum': {'overwrite': True},
            },
        }
        values = numpy.asarray(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
            dtype=numpy.float32,
        )
        frequencies = numpy.asarray([8, 9], dtype=float)
        metadata = {
            **self._source_metadata(),
            'method': 'test',
            'freqs': frequencies,
            'shape': values.shape,
            'dim': 'vertices x freqs',
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'power.h5'
            with patch(
                'sEEGnal.tools.bids_tools.build_derivatives_path',
                return_value=path,
            ):
                write_relative_power_spectrum.__wrapped__(
                    config,
                    BIDS=None,
                    relative_power_spectrum=values,
                    metadata=metadata,
                )
                loaded_values, loaded_freqs, loaded_metadata = (
                    read_relative_power_spectrum(
                        config,
                        BIDS=None,
                        space='source',
                    )
                )

        numpy.testing.assert_array_equal(loaded_values, values)
        numpy.testing.assert_array_equal(loaded_freqs, frequencies)
        self._assert_loaded_source_metadata(loaded_metadata)
        self.assertEqual(loaded_metadata['dim'], 'vertices x freqs')

    def _source_metadata(self):
        return get_source_metadata(
            ['A', 'B', 'C'],
            ['B'],
            self.filters,
            _FakeSourceEstimate(),
            source_spacing='ico2',
        )

    def _connectivity_metadata(self):
        return {
            **self._source_metadata(),
            'method': 'test',
            'n_nodes': 3,
            'ch_names': '',
            'n_epochs_used': 2,
            'band_name': 'alpha',
            'vectorization': 'numpy.triu_indices(n_nodes, k=1)',
            'description': 'test source data',
        }

    def _assert_loaded_source_metadata(self, metadata):
        self.assertEqual(metadata['n_sources'], 3)
        self.assertEqual(metadata['input_ch_names'], ['A', 'B', 'C'])
        self.assertEqual(metadata['input_good_channels'], ['A', 'C'])
        self.assertEqual(metadata['input_bad_channels'], ['B'])
        self.assertEqual(metadata['input_bad_channel_indices'], [1])
        self.assertEqual(metadata['input_excluded_channels'], ['B'])
        self.assertEqual(metadata['input_excluded_channel_indices'], [1])
        self.assertEqual(metadata['source_vertex_numbers'], [10, 11, 20])
        self.assertEqual(metadata['source_vertex_spaces'], ['lh', 'lh', 'rh'])
        self.assertEqual(metadata['source_value_policy'], 'finite')


class TestSourceFeaturePolicies(unittest.TestCase):
    def _make_config(self, measure):
        feature_config = {
            'epoch_definition': {
                'mode': 'fixed',
                'duration': 1,
            },
            'source': {},
        }
        if measure in ('plv', 'ciplv'):
            feature_config.update({
                'freq_limits': [1, 45],
                'source': {
                    'freq_bands_name': ['alpha'],
                    'freq_bands_limits': [[8, 12]],
                },
            })
        else:
            feature_config['source'] = {
                'freq_limits': [1, 45],
                'bandwidth': 2,
                'adaptive': False,
            }

        return {
            'component_estimation': {
                'low_freq': 1,
                'high_freq': 100,
                'resample_frequency': 100,
                'crop_seconds': False,
            },
            'global': {
                'channels_to_include': ['all'],
                'channels_to_exclude': [],
            },
            'source_reconstruction': {
                'forward': {
                    'template': {'spacing': 'ico2'},
                },
                'inverse': {'method': 'lcmv'},
            },
            'feature_extraction': {measure: feature_config},
            'subsystem': 'feature_extraction',
        }

    def _inputs(self):
        raw = SimpleNamespace(
            ch_names=['A', 'B', 'C'],
            info={'bads': ['B']},
        )
        epochs = SimpleNamespace(
            ch_names=['A', 'C'],
            info={'sfreq': 100},
            tmin=0,
            tmax=0.03,
            _seegnal_reference=_average_reference_state(),
        )
        filters = {
            'ch_names': ['A', 'C'],
            'src_type': 'surface',
            'subject': 'fsaverage',
            '_seegnal_reference': _average_reference_state(),
        }
        stcs = [_FakeSourceEstimate(), _FakeSourceEstimate()]
        return raw, epochs, filters, stcs

    def _assert_prepare_policy(self, prepare_eeg_mock):
        first_parameters = prepare_eeg_mock.call_args_list[0].kwargs
        second_parameters = prepare_eeg_mock.call_args_list[1].kwargs

        self.assertIs(first_parameters['metadata_badchannels'], True)
        self.assertIs(first_parameters['exclude_badchannels'], False)
        self.assertIs(first_parameters['interpolate_badchannels'], False)
        self.assertIs(second_parameters['metadata_badchannels'], True)
        self.assertIs(second_parameters['exclude_badchannels'], True)
        self.assertIs(second_parameters['interpolate_badchannels'], False)

    def _assert_source_metadata(self, metadata):
        self.assertEqual(metadata['input_bad_channels'], ['B'])
        self.assertEqual(metadata['input_good_channels'], ['A', 'C'])
        self.assertEqual(metadata['input_bad_channel_policy'], 'exclude')
        self.assertEqual(metadata['source_vertex_numbers'], [10, 11, 20])
        self.assertEqual(metadata['source_vertex_spaces'], ['lh', 'lh', 'rh'])
        self.assertEqual(metadata['source_value_policy'], 'finite')

    def test_source_connectivity_outputs_remain_finite(self):
        cases = (
            (
                'plv',
                estimate_plv,
                'sEEGnal.feature_extraction.FC.estimate_plv',
                'compute_plv',
                'write_plv',
            ),
            (
                'ciplv',
                estimate_ciplv,
                'sEEGnal.feature_extraction.FC.estimate_ciplv',
                'compute_ciplv',
                'write_ciplv',
            ),
        )

        for measure, estimator, module_path, compute_name, writer_name in cases:
            with self.subTest(measure=measure):
                raw, epochs, filters, stcs = self._inputs()
                values = numpy.asarray([0.1, 0.2, 0.3], dtype=numpy.float32)
                with (
                    patch(
                        f'{module_path}.prepare_eeg',
                        side_effect=[raw, epochs],
                    ) as prepare_eeg_mock,
                    patch(
                        f'{module_path}.apply_ica',
                        return_value=raw,
                    ),
                    patch(
                        f'{module_path}.read_inverse_solution',
                        return_value=filters,
                    ),
                    patch(
                        f'{module_path}.mne.beamformer.apply_lcmv_epochs',
                        return_value=stcs,
                    ),
                    patch(f'{module_path}.{compute_name}', return_value=values),
                    patch(f'{module_path}.{writer_name}') as writer_mock,
                ):
                    estimator(self._make_config(measure), BIDS=None)

                self._assert_prepare_policy(prepare_eeg_mock)
                output = writer_mock.call_args.kwargs[measure]
                metadata = writer_mock.call_args.kwargs['metadata']
                self.assertTrue(numpy.isfinite(output).all())
                self._assert_source_metadata(metadata)

    def test_source_relative_power_remains_finite(self):
        raw, epochs, filters, stcs = self._inputs()
        power = numpy.asarray(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
            dtype=numpy.float32,
        )
        module_path = (
            'sEEGnal.feature_extraction.PSD.'
            'estimate_relative_power_spectrum'
        )

        with (
            patch(
                f'{module_path}.prepare_eeg',
                side_effect=[raw, epochs],
            ) as prepare_eeg_mock,
            patch(
                f'{module_path}.apply_ica',
                return_value=raw,
            ),
            patch(
                f'{module_path}.read_inverse_solution',
                return_value=filters,
            ),
            patch(
                f'{module_path}.mne.beamformer.apply_lcmv_epochs',
                return_value=stcs,
            ),
            patch(
                f'{module_path}.multitaper_psd',
                return_value=(power, numpy.asarray([8, 9])),
            ),
            patch(f'{module_path}.normalize_psd', return_value=power),
            patch(
                f'{module_path}.write_relative_power_spectrum',
            ) as writer_mock,
        ):
            estimate_relative_power_spectrum(
                self._make_config('relative_power_spectrum'),
                BIDS=None,
            )

        self._assert_prepare_policy(prepare_eeg_mock)
        output = writer_mock.call_args.kwargs['relative_power_spectrum']
        metadata = writer_mock.call_args.kwargs['metadata']
        self.assertTrue(numpy.isfinite(output).all())
        self.assertEqual(metadata['dim'], 'vertices x freqs')
        self._assert_source_metadata(metadata)

    def test_source_filter_with_different_reference_is_rejected(self):
        """LCMV is not applied to epochs in another referenced space."""
        raw, epochs, filters, _ = self._inputs()
        filters['_seegnal_reference'] = {
            'schema_version': 1,
            'method': 'as_recorded',
            'implementation': 'acquisition',
            'channels': None,
            'status': 'effective',
        }
        module_path = 'sEEGnal.feature_extraction.FC.estimate_plv'

        with (
            patch(
                f'{module_path}.prepare_eeg',
                side_effect=[raw, epochs],
            ),
            patch(
                f'{module_path}.apply_ica',
                return_value=raw,
            ),
            patch(
                f'{module_path}.read_inverse_solution',
                return_value=filters,
            ),
            patch(
                f'{module_path}.mne.beamformer.apply_lcmv_epochs',
            ) as apply_lcmv,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'does not match the effective data reference',
            ):
                estimate_plv(self._make_config('plv'), BIDS=None)

        apply_lcmv.assert_not_called()

    def test_source_filter_without_reference_is_rejected(self):
        """An untraceable LCMV filter fails before source reconstruction."""
        raw, epochs, filters, _ = self._inputs()
        filters.pop('_seegnal_reference')
        module_path = 'sEEGnal.feature_extraction.FC.estimate_ciplv'

        with (
            patch(
                f'{module_path}.prepare_eeg',
                side_effect=[raw, epochs],
            ),
            patch(
                f'{module_path}.apply_ica',
                return_value=raw,
            ),
            patch(
                f'{module_path}.read_inverse_solution',
                return_value=filters,
            ),
            patch(
                f'{module_path}.mne.beamformer.apply_lcmv_epochs',
            ) as apply_lcmv,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'does not record the reference',
            ):
                estimate_ciplv(self._make_config('ciplv'), BIDS=None)

        apply_lcmv.assert_not_called()


class TestInverseReferenceRoundTrip(unittest.TestCase):
    """Persist LCMV reference provenance in its JSON sidecar."""

    def test_reference_metadata_survives_inverse_round_trip(self):
        """Writing and reading restore an independent reference mapping."""
        config = {
            'subsystem': 'source_reconstruction',
            'source_reconstruction': {'inverse': {'method': 'lcmv'}},
        }
        reference_state = _average_reference_state()
        filters = _FakeBeamformer(
            kind='LCMV',
            ch_names=['A', 'C'],
            _seegnal_reference=reference_state,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def derivative_path(BIDS, subsystem, tail):
                return root / tail

            with patch(
                'sEEGnal.tools.bids_tools.build_derivatives_path',
                side_effect=derivative_path,
            ):
                write_inverse_solution.__wrapped__(
                    config,
                    BIDS=None,
                    inverse_solution=filters,
                )

            metadata_path = root / 'desc-lcmv.json'
            metadata = json.loads(
                metadata_path.read_text(encoding='utf-8')
            )
            loaded_filters = _FakeBeamformer(
                kind='LCMV',
                ch_names=['A', 'C'],
            )
            with (
                patch(
                    'sEEGnal.tools.bids_tools.build_derivatives_path',
                    side_effect=derivative_path,
                ),
                patch(
                    'sEEGnal.tools.bids_tools.'
                    'mne.beamformer.read_beamformer',
                    return_value=loaded_filters,
                ),
            ):
                loaded = read_inverse_solution(config, BIDS=None)

        self.assertEqual(metadata['Reference'], reference_state)
        self.assertEqual(loaded['_seegnal_reference'], reference_state)
        self.assertIsNot(loaded['_seegnal_reference'], reference_state)
        self.assertIsNot(
            loaded['_seegnal_reference']['channels'],
            reference_state['channels'],
        )

    def test_writer_rejects_inverse_without_reference_provenance(self):
        """An untraceable inverse solution is not serialized."""
        config = {
            'subsystem': 'source_reconstruction',
            'source_reconstruction': {'inverse': {'method': 'lcmv'}},
        }
        filters = _FakeBeamformer(kind='LCMV', ch_names=['A', 'C'])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch(
                'sEEGnal.tools.bids_tools.build_derivatives_path',
                side_effect=lambda BIDS, subsystem, tail: root / tail,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    'must record the effective estimation reference',
                ):
                    write_inverse_solution.__wrapped__(
                        config,
                        BIDS=None,
                        inverse_solution=filters,
                    )

        self.assertFalse(hasattr(filters, 'saved_path'))


class TestInverseBadChannelPolicy(unittest.TestCase):
    def test_lcmv_marks_bads_before_reference_and_excludes_before_covariance(self):
        config = {
            'component_estimation': {
                'low_freq': 1,
                'high_freq': 45,
                'crop_seconds': False,
                'resample_frequency': 100,
            },
            'global': {
                'channels_to_include': ['all'],
                'channels_to_exclude': [],
            },
            'source_reconstruction': {
                'epoch_definition': {
                    'mode': 'fixed',
                    'duration': 1,
                },
                'covariance': {'method': 'oas', 'rank': 'info'},
                'inverse': {
                    'method': 'lcmv',
                    'reg': 0.05,
                    'pick_ori': 'max-power',
                    'weight_norm': 'nai',
                },
            },
        }
        raw = object()
        epochs = SimpleNamespace(
            info={'sfreq': 100},
            _seegnal_reference=_average_reference_state(),
        )
        covariance = object()
        filters = {'ch_names': ['A', 'C']}

        with (
            patch(
                'sEEGnal.sources_reconstruction.inverse.'
                'mne_tools.prepare_eeg',
                side_effect=[raw, epochs],
            ) as prepare_eeg_mock,
            patch(
                'sEEGnal.sources_reconstruction.inverse.'
                'mne_tools.apply_ica',
                return_value=raw,
            ),
            patch(
                'sEEGnal.sources_reconstruction.inverse.'
                'mne.compute_covariance',
                return_value=covariance,
            ),
            patch(
                'sEEGnal.sources_reconstruction.inverse.mne.compute_rank',
                return_value={'eeg': 2},
            ),
            patch(
                'sEEGnal.sources_reconstruction.inverse.read_forward_model',
                return_value=object(),
            ),
            patch(
                'sEEGnal.sources_reconstruction.inverse.'
                'mne.beamformer.make_lcmv',
                return_value=filters,
            ),
            patch(
                'sEEGnal.sources_reconstruction.inverse.'
                'write_inverse_solution',
            ),
        ):
            result = estimate_lcmv(config, BIDS=None)

        self.assertIs(result, filters)
        first_parameters = prepare_eeg_mock.call_args_list[0].kwargs
        second_parameters = prepare_eeg_mock.call_args_list[1].kwargs
        self.assertIs(first_parameters['metadata_badchannels'], True)
        self.assertIs(first_parameters['exclude_badchannels'], False)
        self.assertIs(first_parameters['interpolate_badchannels'], False)
        self.assertIs(second_parameters['metadata_badchannels'], True)
        self.assertIs(second_parameters['exclude_badchannels'], True)
        self.assertIs(second_parameters['interpolate_badchannels'], False)
        self.assertEqual(
            filters['_seegnal_reference'],
            epochs._seegnal_reference,
        )
        self.assertIsNot(
            filters['_seegnal_reference'],
            epochs._seegnal_reference,
        )


if __name__ == '__main__':
    unittest.main()
