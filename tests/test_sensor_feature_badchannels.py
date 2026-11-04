"""Tests for sensor features computed after bad-channel interpolation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import h5py
import numpy

from sEEGnal.feature_extraction.FC.estimate_ciplv import estimate_ciplv
from sEEGnal.feature_extraction.FC.estimate_plv import estimate_plv
from sEEGnal.feature_extraction.PSD.estimate_relative_power_spectrum import (
    estimate_relative_power_spectrum,
)
from sEEGnal.tools.bids_tools import (
    _read_sensor_metadata,
    _write_sensor_metadata,
    read_ciplv,
    read_plv,
    read_relative_power_spectrum,
    write_ciplv,
    write_plv,
    write_relative_power_spectrum,
)
from sEEGnal.tools.feature_tools import (
    expand_channel_values,
    expand_connectivity_vector,
    get_bad_channel_policy,
    get_channel_mapping,
)


class _FakeRaw:
    def __init__(self):
        self.ch_names = ['A', 'B', 'C']
        self.info = {'bads': ['B']}

    def copy(self):
        return _FakeRaw()


class _FakeEpochs:
    def __init__(self):
        self.ch_names = ['A', 'B', 'C']
        self.info = {'sfreq': 100}
        self.tmin = 0
        self.tmax = 0.03
        self._data = numpy.ones((2, 3, 4), dtype=numpy.float32)

    def get_data(self):
        return self._data

    def copy(self):
        return self

    def load_data(self):
        return self

    def filter(self, *args, **kwargs):
        return self


class TestFeatureChannelMapping(unittest.TestCase):
    def test_bad_channel_policy_defaults_to_interpolation(self):
        self.assertEqual(get_bad_channel_policy({}), 'interpolate')
        self.assertEqual(
            get_bad_channel_policy({'bad_channel_policy': 'interpolate'}),
            'interpolate',
        )

    def test_unsupported_bad_channel_policy_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "bad_channel_policy must be 'interpolate'",
        ):
            get_bad_channel_policy({'bad_channel_policy': 'nan'})

    def test_mapping_preserves_canonical_order(self):
        good, bad, bad_names = get_channel_mapping(
            ['A', 'B', 'C', 'D'],
            ['D', 'A'],
        )

        numpy.testing.assert_array_equal(good, [3, 0])
        numpy.testing.assert_array_equal(bad, [1, 2])
        self.assertEqual(bad_names, ['B', 'C'])

    def test_connectivity_expansion_marks_bad_channel_edges(self):
        expanded, nan_indices = expand_connectivity_vector(
            numpy.asarray([0.25], dtype=numpy.float32),
            good_channel_indices=[0, 2],
            n_channels=3,
        )

        numpy.testing.assert_allclose(expanded[1], 0.25)
        self.assertTrue(numpy.isnan(expanded[[0, 2]]).all())
        numpy.testing.assert_array_equal(nan_indices, [0, 2])

    def test_connectivity_expansion_accepts_reordered_good_channels(self):
        expanded, nan_indices = expand_connectivity_vector(
            numpy.asarray([0.25], dtype=numpy.float32),
            good_channel_indices=[2, 0],
            n_channels=3,
        )

        numpy.testing.assert_allclose(expanded[1], 0.25)
        self.assertTrue(numpy.isnan(expanded[[0, 2]]).all())
        numpy.testing.assert_array_equal(nan_indices, [0, 2])

    def test_channel_expansion_marks_bad_channel_rows(self):
        values = numpy.asarray(
            [[1, 2], [3, 4]],
            dtype=numpy.float32,
        )
        expanded = expand_channel_values(
            values,
            good_channel_indices=[0, 2],
            n_channels=3,
        )

        numpy.testing.assert_allclose(expanded[[0, 2]], values)
        self.assertTrue(numpy.isnan(expanded[1]).all())

    def test_expansion_without_bad_channels_preserves_values(self):
        connectivity = numpy.asarray(
            [0.1, 0.2, 0.3],
            dtype=numpy.float32,
        )
        expanded_connectivity, nan_indices = expand_connectivity_vector(
            connectivity,
            good_channel_indices=[0, 1, 2],
            n_channels=3,
        )
        channel_values = numpy.asarray(
            [[1, 2], [3, 4], [5, 6]],
            dtype=numpy.float32,
        )
        expanded_channels = expand_channel_values(
            channel_values,
            good_channel_indices=[0, 1, 2],
            n_channels=3,
        )

        numpy.testing.assert_array_equal(
            expanded_connectivity,
            connectivity,
        )
        numpy.testing.assert_array_equal(nan_indices, [])
        numpy.testing.assert_array_equal(expanded_channels, channel_values)

    def test_sensor_metadata_round_trip_supports_empty_lists(self):
        metadata = {
            'ch_names': ['A', 'B', 'C'],
            'bad_channels': [],
            'good_channel_indices': [0, 1, 2],
            'bad_channel_indices': [],
            'nan_connection_indices': [],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'metadata.h5'
            with h5py.File(path, 'w') as file:
                group = file.create_group('sensor')
                _write_sensor_metadata(group, metadata)

            loaded = {}
            with h5py.File(path, 'r') as file:
                _read_sensor_metadata(file['sensor'], loaded)

        self.assertEqual(loaded, metadata)

    def test_connectivity_writers_round_trip_nan_metadata(self):
        config = {
            'current_space': 'sensor',
            'subsystem': 'feature_extraction',
        }
        values = numpy.asarray(
            [numpy.nan, 0.25, numpy.nan],
            dtype=numpy.float32,
        )
        metadata = {
            'method': 'test',
            'n_nodes': 3,
            'n_good_nodes': 2,
            'ch_names': ['A', 'B', 'C'],
            'good_channel_indices': [0, 2],
            'bad_channels': ['B'],
            'bad_channel_indices': [1],
            'nan_connection_indices': [0, 2],
            'bad_channel_policy': 'nan',
            'vectorization': 'numpy.triu_indices(n_nodes, k=1)',
            'n_epochs_used': 2,
            'band_name': 'alpha',
            'description': 'test data',
        }

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
                            band_name='alpha',
                        )

                    numpy.testing.assert_allclose(
                        loaded_values,
                        values,
                        equal_nan=True,
                    )
                    self.assertEqual(
                        loaded_metadata['bad_channels'],
                        ['B'],
                    )
                    self.assertEqual(
                        loaded_metadata['bad_channel_indices'],
                        [1],
                    )
                    self.assertEqual(
                        loaded_metadata['nan_connection_indices'],
                        [0, 2],
                    )

    def test_relative_power_writer_round_trips_nan_metadata(self):
        config = {
            'current_space': 'sensor',
            'subsystem': 'feature_extraction',
            'feature_extraction': {
                'relative_power_spectrum': {
                    'overwrite': True,
                },
            },
        }
        values = numpy.asarray(
            [[1, 2], [numpy.nan, numpy.nan], [3, 4]],
            dtype=numpy.float32,
        )
        frequencies = numpy.asarray([8, 9], dtype=float)
        metadata = {
            'method': 'test',
            'n_nodes': 3,
            'n_good_nodes': 2,
            'ch_names': ['A', 'B', 'C'],
            'good_channel_indices': [0, 2],
            'bad_channels': ['B'],
            'bad_channel_indices': [1],
            'nan_channel_indices': [1],
            'bad_channel_policy': 'nan',
            'freqs': frequencies,
            'shape': values.shape,
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
                    )
                )

        numpy.testing.assert_allclose(
            loaded_values,
            values,
            equal_nan=True,
        )
        numpy.testing.assert_array_equal(loaded_freqs, frequencies)
        self.assertEqual(loaded_metadata['bad_channels'], ['B'])
        self.assertEqual(loaded_metadata['bad_channel_indices'], [1])
        self.assertEqual(loaded_metadata['nan_channel_indices'], [1])


class TestSensorFeaturePolicies(unittest.TestCase):
    def _make_config(self, measure):
        feature_config = {
            'epoch_definition': {
                'mode': 'fixed',
                'duration': 1,
            },
            'sensor': {},
        }
        if measure in ('plv', 'ciplv'):
            feature_config.update({
                'freq_limits': [1, 45],
                'sensor': {
                    'freq_bands_name': ['alpha'],
                    'freq_bands_limits': [[8, 12]],
                },
            })
        else:
            feature_config['sensor'] = {
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
            'feature_extraction': {
                measure: feature_config,
            },
            'subsystem': 'feature_extraction',
        }

    def _assert_prepare_policy(self, prepare_eeg_mock):
        first_parameters = prepare_eeg_mock.call_args_list[0].kwargs
        second_parameters = prepare_eeg_mock.call_args_list[1].kwargs

        self.assertIs(first_parameters['metadata_badchannels'], True)
        self.assertIs(
            first_parameters.get('exclude_badchannels', False),
            False,
        )
        self.assertIs(
            first_parameters.get('interpolate_badchannels', False),
            False,
        )
        self.assertIs(second_parameters['metadata_badchannels'], True)
        self.assertIs(second_parameters['exclude_badchannels'], False)
        self.assertIs(
            second_parameters.get('interpolate_badchannels', False),
            True,
        )

    def _assert_connectivity_output(self, writer_mock, value_name):
        values = writer_mock.call_args.kwargs[value_name]
        metadata = writer_mock.call_args.kwargs['metadata']

        self.assertEqual(values.shape, (3,))
        numpy.testing.assert_allclose(values, [0.1, 0.2, 0.3])
        self.assertEqual(metadata['ch_names'], ['A', 'B', 'C'])
        self.assertEqual(metadata['good_channel_indices'], [0, 2])
        self.assertEqual(metadata['bad_channels'], ['B'])
        self.assertEqual(metadata['bad_channel_indices'], [1])
        self.assertEqual(metadata['nan_connection_indices'], [])
        self.assertEqual(metadata['bad_channel_policy'], 'interpolate')

    def test_plv_interpolates_bad_signals_before_estimation(self):
        raw = _FakeRaw()
        epochs = _FakeEpochs()

        with (
            patch(
                'sEEGnal.feature_extraction.FC.estimate_plv.prepare_eeg',
                side_effect=[raw, epochs],
            ) as prepare_eeg_mock,
            patch(
                'sEEGnal.feature_extraction.FC.estimate_plv.apply_ica',
                return_value=raw,
            ) as apply_ica_mock,
            patch(
                'sEEGnal.feature_extraction.FC.estimate_plv.compute_plv',
                return_value=numpy.asarray(
                    [0.1, 0.2, 0.3],
                    dtype=numpy.float32,
                ),
            ),
            patch(
                'sEEGnal.feature_extraction.FC.estimate_plv.write_plv',
            ) as writer_mock,
        ):
            estimate_plv(self._make_config('plv'), BIDS=None)

        self._assert_prepare_policy(prepare_eeg_mock)
        apply_ica_mock.assert_called_once_with(
            self._make_config('plv'),
            None,
            raw,
            ANY,
        )
        self._assert_connectivity_output(writer_mock, 'plv')

    def test_ciplv_interpolates_bad_signals_before_estimation(self):
        raw = _FakeRaw()
        epochs = _FakeEpochs()

        with (
            patch(
                'sEEGnal.feature_extraction.FC.estimate_ciplv.prepare_eeg',
                side_effect=[raw, epochs],
            ) as prepare_eeg_mock,
            patch(
                'sEEGnal.feature_extraction.FC.estimate_ciplv.apply_ica',
                return_value=raw,
            ),
            patch(
                'sEEGnal.feature_extraction.FC.estimate_ciplv.compute_ciplv',
                return_value=numpy.asarray(
                    [0.1, 0.2, 0.3],
                    dtype=numpy.float32,
                ),
            ),
            patch(
                'sEEGnal.feature_extraction.FC.estimate_ciplv.write_ciplv',
            ) as writer_mock,
        ):
            estimate_ciplv(self._make_config('ciplv'), BIDS=None)

        self._assert_prepare_policy(prepare_eeg_mock)
        self._assert_connectivity_output(writer_mock, 'ciplv')

    def test_relative_power_interpolates_bad_signals_before_estimation(self):
        raw = _FakeRaw()
        epochs = _FakeEpochs()
        power = numpy.asarray(
            [[1, 2], [3, 4], [5, 6]],
            dtype=numpy.float32,
        )
        frequencies = numpy.asarray([8, 9], dtype=float)

        with (
            patch(
                'sEEGnal.feature_extraction.PSD.'
                'estimate_relative_power_spectrum.prepare_eeg',
                side_effect=[raw, epochs],
            ) as prepare_eeg_mock,
            patch(
                'sEEGnal.feature_extraction.PSD.'
                'estimate_relative_power_spectrum.apply_ica',
                return_value=raw,
            ),
            patch(
                'sEEGnal.feature_extraction.PSD.'
                'estimate_relative_power_spectrum.multitaper_psd',
                return_value=(power, frequencies),
            ),
            patch(
                'sEEGnal.feature_extraction.PSD.'
                'estimate_relative_power_spectrum.normalize_psd',
                return_value=power,
            ),
            patch(
                'sEEGnal.feature_extraction.PSD.'
                'estimate_relative_power_spectrum.'
                'write_relative_power_spectrum',
            ) as writer_mock,
        ):
            estimate_relative_power_spectrum(
                self._make_config('relative_power_spectrum'),
                BIDS=None,
            )

        self._assert_prepare_policy(prepare_eeg_mock)
        values = writer_mock.call_args.kwargs['relative_power_spectrum']
        metadata = writer_mock.call_args.kwargs['metadata']
        numpy.testing.assert_allclose(values, power)
        self.assertEqual(metadata['ch_names'], ['A', 'B', 'C'])
        self.assertEqual(metadata['good_channel_indices'], [0, 2])
        self.assertEqual(metadata['bad_channels'], ['B'])
        self.assertEqual(metadata['bad_channel_indices'], [1])
        self.assertEqual(metadata['nan_channel_indices'], [])
        self.assertEqual(metadata['bad_channel_policy'], 'interpolate')

    def test_sensor_and_source_use_independent_bad_channel_policies(self):
        config = self._make_config('plv')
        config['feature_extraction']['plv']['source'] = {
            'freq_bands_name': ['alpha'],
            'freq_bands_limits': [[8, 12]],
        }
        config['source_reconstruction'] = {
            'forward': {'template': {'spacing': 'ico2'}},
            'inverse': {'method': 'lcmv'},
        }
        raw = _FakeRaw()
        sensor_epochs = _FakeEpochs()
        source_epochs = SimpleNamespace()

        module_path = 'sEEGnal.feature_extraction.FC.estimate_plv'
        with (
            patch(
                f'{module_path}.prepare_eeg',
                side_effect=[raw, sensor_epochs, source_epochs],
            ) as prepare_eeg_mock,
            patch(f'{module_path}.apply_ica', return_value=raw),
            patch(
                f'{module_path}.compute_plv',
                return_value=numpy.asarray(
                    [0.1, 0.2, 0.3],
                    dtype=numpy.float32,
                ),
            ),
            patch(f'{module_path}.write_plv'),
            patch(f'{module_path}.read_inverse_solution', return_value={}),
            patch(
                f'{module_path}._check_inverse_reference',
                side_effect=RuntimeError('stop after epoch preparation'),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'stop after epoch preparation',
            ):
                estimate_plv(config, BIDS=None)

        self.assertEqual(len(prepare_eeg_mock.call_args_list), 3)
        sensor_parameters = prepare_eeg_mock.call_args_list[1].kwargs
        source_parameters = prepare_eeg_mock.call_args_list[2].kwargs
        self.assertIsNot(sensor_parameters['raw'], source_parameters['raw'])
        self.assertIs(sensor_parameters['exclude_badchannels'], False)
        self.assertIs(sensor_parameters['interpolate_badchannels'], True)
        self.assertIs(source_parameters['exclude_badchannels'], True)
        self.assertIs(source_parameters['interpolate_badchannels'], False)


if __name__ == '__main__':
    unittest.main()
