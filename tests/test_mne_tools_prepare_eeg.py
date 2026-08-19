"""Integration tests for validation, epoching and cropping in prepare_eeg."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mne
import mne_bids
import numpy
import pandas

from sEEGnal.tools.bids_tools import write_annotations
from sEEGnal.tools.mne_tools import (
    _clip_annotation_to_time_window,
    apply_ica,
    build_raw,
    get_reference_info,
    prepare_eeg,
)
from tests._mne_test_data import (
    SAMPLE_SCALE,
    assert_epoch_contains_samples,
    make_synthetic_raw,
)


class TestPrepareEegChannelSelection(unittest.TestCase):
    """Select all typed EEG channels plus explicit name exclusions."""

    def test_eeg_selector_excludes_other_types_and_named_exceptions(self):
        info = mne.create_info(
            ['Cz', 'Pz', 'CLAV', 'HEOG', 'ECG', 'STI 014'],
            sfreq=100,
            ch_types=['eeg', 'eeg', 'eeg', 'eog', 'ecg', 'stim'],
        )
        raw = mne.io.RawArray(
            numpy.zeros((len(info['ch_names']), 200)),
            info,
            verbose=False,
        )

        prepared = prepare_eeg(
            {},
            None,
            raw=raw,
            channels_to_include='eeg',
            channels_to_exclude=['CLAV'],
        )

        self.assertEqual(prepared.ch_names, ['Cz', 'Pz'])


class TestPrepareEegReferenceValidation(unittest.TestCase):
    """Check the closed set of reference options before processing starts."""

    def test_false_and_average_are_accepted(self):
        for rereference in (False, 'average'):
            with self.subTest(rereference=rereference):
                raw = make_synthetic_raw(sfreq=1000, duration=2)

                prepared = prepare_eeg(
                    {},
                    None,
                    raw=raw,
                    preload=True,
                    rereference=rereference,
                )

                self.assertIs(prepared, raw)
                if rereference == 'average':
                    self.assertEqual(len(prepared.info['projs']), 1)
                    self.assertEqual(
                        prepared.info['projs'][0]['desc'],
                        'Average EEG reference',
                    )
                    self.assertTrue(prepared.info['projs'][0]['active'])
                else:
                    self.assertEqual(prepared.info['projs'], [])

    def test_other_reference_values_are_rejected_before_raw_mutation(self):
        invalid_values = (True, None, 'median', '', 0)

        for rereference in invalid_values:
            with self.subTest(rereference=rereference):
                raw = make_synthetic_raw(sfreq=1000, duration=4)
                original_data = raw.get_data().copy()
                original_ch_names = list(raw.ch_names)
                original_first_samp = raw.first_samp
                original_last_samp = raw.last_samp

                with self.assertRaisesRegex(
                    ValueError,
                    "rereference must be False or 'average'",
                ):
                    prepare_eeg(
                        {},
                        None,
                        raw=raw,
                        preload=True,
                        channels_to_include=['Cz'],
                        crop_seconds=1,
                        rereference=rereference,
                    )

                self.assertEqual(raw.ch_names, original_ch_names)
                self.assertEqual(raw.first_samp, original_first_samp)
                self.assertEqual(raw.last_samp, original_last_samp)
                self.assertEqual(raw.info['projs'], [])
                numpy.testing.assert_array_equal(raw.get_data(), original_data)

    def test_invalid_reference_is_rejected_before_loading_bids(self):
        with patch(
            'sEEGnal.tools.mne_tools.read_BIDS_files'
        ) as read_bids_files:
            with self.assertRaisesRegex(
                ValueError,
                "rereference must be False or 'average'",
            ):
                prepare_eeg({}, None, rereference='median')

        read_bids_files.assert_not_called()


class TestPrepareEegProjectorInvariant(unittest.TestCase):
    """Ensure prepare_eeg never absorbs pending input projectors."""

    def test_pending_input_projector_is_rejected_before_mutation(self):
        """Pending projectors fail before other preparation steps run."""
        for rereference in (False, 'average'):
            with self.subTest(rereference=rereference):
                raw = make_synthetic_raw(sfreq=1000, duration=4)
                raw.set_eeg_reference(
                    'average',
                    projection=True,
                    verbose=False,
                )
                original_data = raw.get_data().copy()
                original_ch_names = list(raw.ch_names)
                original_first_samp = raw.first_samp
                original_last_samp = raw.last_samp

                with self.assertRaisesRegex(
                    RuntimeError,
                    "unapplied projector.*'Average EEG reference'",
                ):
                    prepare_eeg(
                        {},
                        None,
                        raw=raw,
                        preload=True,
                        channels_to_include=['Cz'],
                        crop_seconds=1,
                        rereference=rereference,
                    )

                self.assertFalse(hasattr(raw, '_seegnal_reference'))
                self.assertEqual(raw.ch_names, original_ch_names)
                self.assertEqual(raw.first_samp, original_first_samp)
                self.assertEqual(raw.last_samp, original_last_samp)
                self.assertFalse(raw.info['projs'][0]['active'])
                numpy.testing.assert_array_equal(raw.get_data(), original_data)

    def test_applied_input_projector_is_preserved(self):
        """Already effective projectors pass through without reapplication."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)
        projected_data = raw.get_data().copy()

        prepared = prepare_eeg({}, None, raw=raw, rereference=False)

        self.assertIs(prepared, raw)
        self.assertTrue(prepared.info['projs'][0]['active'])
        numpy.testing.assert_array_equal(prepared.get_data(), projected_data)


class TestPrepareEegReferenceState(unittest.TestCase):
    """Check initialization and preservation of reference provenance."""

    def test_false_initializes_independent_as_recorded_states(self):
        """False initializes one fresh effective state per Raw object."""
        raw_1 = make_synthetic_raw(sfreq=1000, duration=2)
        raw_2 = make_synthetic_raw(sfreq=1000, duration=2)
        data_1 = raw_1.get_data().copy()
        data_2 = raw_2.get_data().copy()

        prepared_1 = prepare_eeg({}, None, raw=raw_1, rereference=False)
        prepared_2 = prepare_eeg({}, None, raw=raw_2, rereference=False)

        expected = {
            'schema_version': 1,
            'method': 'as_recorded',
            'implementation': 'acquisition',
            'channels': None,
            'status': 'effective',
        }
        self.assertEqual(prepared_1._seegnal_reference, expected)
        self.assertEqual(prepared_2._seegnal_reference, expected)
        self.assertIsNot(
            prepared_1._seegnal_reference,
            prepared_2._seegnal_reference,
        )
        numpy.testing.assert_array_equal(prepared_1.get_data(), data_1)
        numpy.testing.assert_array_equal(prepared_2.get_data(), data_2)

    def test_false_preserves_existing_reference_state(self):
        """A later call without rereferencing does not reset provenance."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        reference_state = {
            'schema_version': 1,
            'method': 'average',
            'implementation': 'mne_projection',
            'channels': list(raw.ch_names),
            'status': 'effective',
        }
        raw._seegnal_reference = reference_state

        prepared = prepare_eeg({}, None, raw=raw, rereference=False)

        self.assertIs(prepared._seegnal_reference, reference_state)
        self.assertEqual(prepared._seegnal_reference, reference_state)

    def test_average_reference_is_applied_and_recorded(self):
        """Average rereferencing changes EEG data and records provenance."""
        sfreq = 1000
        times = numpy.arange(2000) / sfreq
        original_data = numpy.vstack((
            numpy.sin(2 * numpy.pi * 7 * times),
            0.5 * numpy.cos(2 * numpy.pi * 11 * times) + 0.25,
            numpy.zeros(times.size),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'STI 014'],
            sfreq=sfreq,
            ch_types=['eeg', 'eeg', 'stim'],
        )
        raw = mne.io.RawArray(original_data.copy(), info, verbose=False)
        expected_data = original_data.copy()
        expected_data[:2] -= original_data[:2].mean(axis=0, keepdims=True)

        prepared = prepare_eeg({}, None, raw=raw, rereference='average')

        numpy.testing.assert_allclose(
            prepared.get_data(),
            expected_data,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(
            prepared._seegnal_reference,
            {
                'schema_version': 1,
                'method': 'average',
                'implementation': 'mne_projection',
                'channels': ['Cz', 'Pz'],
                'status': 'effective',
            },
        )
        self.assertTrue(prepared.proj)
        self.assertEqual(len(prepared.info['projs']), 1)
        self.assertTrue(prepared.info['projs'][0]['active'])

    def test_average_reference_excludes_channels_marked_bad(self):
        """Bad EEG channels remain present but do not define the average."""
        sfreq = 1000
        times = numpy.arange(1000) / sfreq
        original_data = numpy.vstack((
            numpy.sin(2 * numpy.pi * 7 * times),
            numpy.cos(2 * numpy.pi * 11 * times),
            0.5 * numpy.sin(2 * numpy.pi * 17 * times),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz'],
            sfreq=sfreq,
            ch_types='eeg',
        )
        raw = mne.io.RawArray(original_data.copy(), info, verbose=False)
        raw.info['bads'] = ['Pz']
        expected_data = original_data.copy()
        good_average = original_data[[0, 2]].mean(axis=0, keepdims=True)
        expected_data[[0, 2]] -= good_average

        prepared = prepare_eeg({}, None, raw=raw, rereference='average')

        self.assertEqual(prepared.ch_names, ['Cz', 'Pz', 'Fz'])
        self.assertEqual(prepared.info['bads'], ['Pz'])
        self.assertEqual(
            prepared._seegnal_reference['channels'],
            ['Cz', 'Fz'],
        )
        numpy.testing.assert_allclose(
            prepared.get_data(),
            expected_data,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_repeated_average_reference_is_idempotent(self):
        """A second average request neither transforms nor duplicates data."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)

        first = prepare_eeg({}, None, raw=raw, rereference='average')
        data_after_first = first.get_data().copy()
        state_after_first = dict(first._seegnal_reference)
        channels_after_first = list(state_after_first['channels'])

        second = prepare_eeg({}, None, raw=first, rereference='average')

        self.assertIs(second, first)
        self.assertEqual(len(second.info['projs']), 1)
        self.assertTrue(second.info['projs'][0]['active'])
        self.assertEqual(second._seegnal_reference, state_after_first)
        self.assertEqual(
            second._seegnal_reference['channels'],
            channels_after_first,
        )
        numpy.testing.assert_array_equal(second.get_data(), data_after_first)

    def test_average_reference_is_recomputed_after_channel_selection(self):
        """A new sensor space receives its own requested average reference."""
        data = numpy.vstack((
            numpy.ones(200),
            numpy.full(200, 2.0),
            numpy.full(200, 6.0),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz'],
            sfreq=100,
            ch_types='eeg',
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        first = prepare_eeg({}, None, raw=raw, rereference='average')
        second = prepare_eeg(
            {},
            None,
            raw=first,
            channels_to_include=['Cz', 'Pz'],
            rereference='average',
        )

        numpy.testing.assert_allclose(
            second.get_data().mean(axis=0),
            0,
            rtol=0,
            atol=1e-12,
        )
        self.assertEqual(
            second._seegnal_reference['channels'],
            ['Cz', 'Pz'],
        )
        self.assertEqual(len(second.info['projs']), 1)
        self.assertTrue(second.info['projs'][0]['active'])

    def test_existing_active_average_projector_is_adopted(self):
        """An external active average projector receives sEEGnal metadata."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)
        projected_data = raw.get_data().copy()

        prepared = prepare_eeg({}, None, raw=raw, rereference='average')

        self.assertEqual(len(prepared.info['projs']), 1)
        self.assertTrue(prepared.info['projs'][0]['active'])
        self.assertEqual(
            prepared._seegnal_reference,
            {
                'schema_version': 1,
                'method': 'average',
                'implementation': 'mne_projection',
                'channels': ['Cz'],
                'status': 'effective',
            },
        )
        numpy.testing.assert_array_equal(prepared.get_data(), projected_data)

    def test_inconsistent_average_projection_metadata_is_rejected(self):
        """Projection metadata cannot claim a missing active projector."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        raw._seegnal_reference = {
            'schema_version': 1,
            'method': 'average',
            'implementation': 'mne_projection',
            'channels': ['Cz'],
            'status': 'effective',
        }

        with self.assertRaisesRegex(
            RuntimeError,
            'no active average projector is stored',
        ):
            prepare_eeg({}, None, raw=raw, rereference='average')

    def test_fixed_and_event_epochs_receive_independent_state(self):
        """Both epoch modes copy provenance without sharing mutable values."""
        definitions = (
            {
                'mode': 'fixed',
                'duration': 1,
                'reject_by_annotation': False,
            },
            {
                'mode': 'events',
                'event_source': 'annotations',
                'event_id': {'target': 1},
                'tmin': 0,
                'tmax': 0.5,
                'baseline': None,
                'reject_by_annotation': False,
            },
        )

        for definition in definitions:
            with self.subTest(mode=definition['mode']):
                raw = make_synthetic_raw(
                    sfreq=1000,
                    duration=3,
                    first_samp=0,
                    annotations=((1, 0, 'target'),),
                )
                raw = prepare_eeg(
                    {},
                    None,
                    raw=raw,
                    preload=True,
                    rereference='average',
                )
                reference_state = raw._seegnal_reference

                epochs = prepare_eeg(
                    {},
                    None,
                    raw=raw,
                    preload=True,
                    rereference=False,
                    epoch_definition=definition,
                )

                self.assertEqual(epochs._seegnal_reference, reference_state)
                self.assertIsNot(epochs._seegnal_reference, reference_state)
                self.assertIsNot(
                    epochs._seegnal_reference['channels'],
                    reference_state['channels'],
                )
                epochs._seegnal_reference['channels'].append('Pz')
                self.assertEqual(reference_state['channels'], ['Cz'])

    def test_build_raw_initializes_as_recorded_state(self):
        """A RawArray built without a source starts as acquired."""
        info = {
            'channels': {'label': numpy.array(['Cz', 'Pz'])},
            'sample_rate': 1000,
            'acquisition_time': None,
            'events': [],
        }
        raw = build_raw(info, numpy.zeros((100, 2)))

        self.assertEqual(
            raw._seegnal_reference,
            {
                'schema_version': 1,
                'method': 'as_recorded',
                'implementation': 'acquisition',
                'channels': None,
                'status': 'effective',
            },
        )


class TestGetReferenceInfo(unittest.TestCase):
    """Check the public, read-safe reference metadata accessor."""

    def test_unprepared_raw_is_initialized_as_recorded(self):
        """A Raw without sEEGnal metadata receives the acquisition default."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)

        reference_info = get_reference_info(raw)

        expected = {
            'schema_version': 1,
            'method': 'as_recorded',
            'implementation': 'acquisition',
            'channels': None,
            'status': 'effective',
        }
        self.assertEqual(reference_info, expected)
        self.assertEqual(raw._seegnal_reference, expected)
        self.assertIsNot(reference_info, raw._seegnal_reference)

    def test_returned_average_state_is_an_independent_copy(self):
        """Editing public information cannot alter the Raw provenance."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        raw = prepare_eeg({}, None, raw=raw, rereference='average')

        reference_info = get_reference_info(raw)
        reference_info['method'] = 'changed-by-caller'
        reference_info['channels'].append('Pz')

        self.assertEqual(raw._seegnal_reference['method'], 'average')
        self.assertEqual(raw._seegnal_reference['channels'], ['Cz'])
        self.assertIsNot(
            reference_info['channels'],
            raw._seegnal_reference['channels'],
        )

    def test_epochs_are_supported(self):
        """The public accessor exposes propagated Epochs provenance."""
        raw = make_synthetic_raw(sfreq=1000, duration=2)
        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            rereference='average',
            epoch_definition={
                'mode': 'fixed',
                'duration': 1,
                'reject_by_annotation': False,
            },
        )

        reference_info = get_reference_info(epochs)

        self.assertEqual(reference_info, epochs._seegnal_reference)
        self.assertIsNot(reference_info, epochs._seegnal_reference)
        self.assertIsNot(
            reference_info['channels'],
            epochs._seegnal_reference['channels'],
        )

    def test_non_mne_input_is_rejected(self):
        """Unsupported objects produce a clear public API error."""
        with self.assertRaisesRegex(
            TypeError,
            'mnedata must be an MNE Raw or Epochs object',
        ):
            get_reference_info({'method': 'average'})


class TestPrepareEegRawEpochReferenceConsistency(unittest.TestCase):
    """Compare effective rereferencing in continuous and epoched outputs."""

    @staticmethod
    def _make_raw():
        """Create deterministic two-channel EEG data with target events."""
        sfreq = 1000
        times = numpy.arange(4000) / sfreq
        data = numpy.vstack((
            numpy.sin(2 * numpy.pi * 7 * times) + 0.1,
            0.4 * numpy.cos(2 * numpy.pi * 13 * times) - 0.2,
            numpy.zeros(times.size),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'STI 014'],
            sfreq=sfreq,
            ch_types=['eeg', 'eeg', 'stim'],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_annotations(mne.Annotations(
            onset=[1, 2.5],
            duration=[0, 0],
            description=['target', 'target'],
        ))

        return raw

    def test_fixed_and_event_epochs_match_referenced_raw(self):
        """Epoch data equal the corresponding slices of referenced Raw."""
        definitions = (
            (
                'fixed',
                {
                    'mode': 'fixed',
                    'duration': 1,
                    'overlap': 0,
                    'reject_by_annotation': False,
                },
                4,
            ),
            (
                'events',
                {
                    'mode': 'events',
                    'event_source': 'annotations',
                    'event_id': {'target': 1},
                    'tmin': -0.2,
                    'tmax': 0.3,
                    'baseline': None,
                    'reject_by_annotation': False,
                },
                2,
            ),
        )

        for mode, definition, expected_epochs in definitions:
            for preload in (False, True):
                with self.subTest(mode=mode, preload=preload):
                    source = self._make_raw()
                    referenced_raw = prepare_eeg(
                        {},
                        None,
                        raw=source.copy(),
                        preload=preload,
                        rereference='average',
                    )
                    epochs = prepare_eeg(
                        {},
                        None,
                        raw=source.copy(),
                        preload=preload,
                        rereference='average',
                        epoch_definition=definition,
                    )

                    self.assertEqual(len(epochs.events), expected_epochs)
                    self.assertTrue(referenced_raw.proj)
                    self.assertTrue(epochs.proj)
                    self.assertTrue(all(
                        projector['active']
                        for projector in referenced_raw.info['projs']
                    ))
                    self.assertTrue(all(
                        projector['active']
                        for projector in epochs.info['projs']
                    ))
                    self.assertEqual(
                        epochs._seegnal_reference,
                        referenced_raw._seegnal_reference,
                    )
                    self.assertIsNot(
                        epochs._seegnal_reference,
                        referenced_raw._seegnal_reference,
                    )
                    self.assertIsNot(
                        epochs._seegnal_reference['channels'],
                        referenced_raw._seegnal_reference['channels'],
                    )

                    raw_data = referenced_raw.get_data(
                        picks=['Cz', 'Pz'],
                    )
                    epoch_data = epochs.get_data(picks=['Cz', 'Pz'])
                    first_time_offset = int(numpy.round(
                        epochs.times[0] * epochs.info['sfreq']
                    ))
                    for epoch_index, event in enumerate(epochs.events):
                        start = (
                            event[0]
                            - referenced_raw.first_samp
                            + first_time_offset
                        )
                        stop = start + len(epochs.times)
                        numpy.testing.assert_allclose(
                            epoch_data[epoch_index],
                            raw_data[:, start:stop],
                            rtol=1e-12,
                            atol=1e-12,
                        )

                    numpy.testing.assert_allclose(
                        epoch_data.mean(axis=1),
                        0,
                        rtol=0,
                        atol=1e-12,
                    )

    def test_epoching_in_a_later_call_does_not_reapply_reference(self):
        """A later epoch-only call preserves already referenced samples."""
        referenced_raw = prepare_eeg(
            {},
            None,
            raw=self._make_raw(),
            preload=True,
            rereference='average',
        )
        referenced_data = referenced_raw.get_data(
            picks=['Cz', 'Pz'],
        ).copy()
        reference_state = referenced_raw._seegnal_reference

        epochs = prepare_eeg(
            {},
            None,
            raw=referenced_raw,
            preload=True,
            rereference=False,
            epoch_definition={
                'mode': 'fixed',
                'duration': 1,
                'overlap': 0,
                'proj': False,
                'reject_by_annotation': False,
            },
        )

        self.assertEqual(len(epochs), 4)
        self.assertEqual(len(referenced_raw.info['projs']), 1)
        self.assertEqual(len(epochs.info['projs']), 1)
        self.assertTrue(referenced_raw.info['projs'][0]['active'])
        self.assertTrue(epochs.info['projs'][0]['active'])
        self.assertEqual(epochs._seegnal_reference, reference_state)
        self.assertIsNot(epochs._seegnal_reference, reference_state)
        numpy.testing.assert_array_equal(
            referenced_raw.get_data(picks=['Cz', 'Pz']),
            referenced_data,
        )

        epoch_data = epochs.get_data(picks=['Cz', 'Pz'])
        for epoch_index, event in enumerate(epochs.events):
            start = event[0] - referenced_raw.first_samp
            stop = start + len(epochs.times)
            numpy.testing.assert_allclose(
                epoch_data[epoch_index],
                referenced_data[:, start:stop],
                rtol=1e-12,
                atol=1e-12,
            )


class TestPrepareEegProcessingContract(unittest.TestCase):
    """Check argument safety and the order of basic EEG operations."""

    @staticmethod
    def _make_three_eeg_raw(sfreq=100, duration=10):
        times = numpy.arange(int(sfreq * duration)) / sfreq
        data = numpy.vstack((
            numpy.sin(2 * numpy.pi * 7 * times),
            numpy.cos(2 * numpy.pi * 11 * times),
            numpy.sin(2 * numpy.pi * 17 * times + 0.2),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz'],
            sfreq=sfreq,
            ch_types='eeg',
        )
        return mne.io.RawArray(data, info, verbose=False)

    def test_invalid_parameters_fail_before_mutating_supplied_raw(self):
        cases = (
            (
                {'exclude_badchannels': True,
                 'interpolate_badchannels': True},
                ValueError,
                'cannot both be True',
            ),
            (
                {'epoch_definition': True},
                TypeError,
                'epoch_definition must be a dictionary',
            ),
            (
                {'crop_seconds': -1},
                ValueError,
                'crop_seconds must be finite and non-negative',
            ),
            (
                {'freq_limits': [40, 20]},
                ValueError,
                'lower frequency limit must be below',
            ),
        )

        for parameters, error_type, message in cases:
            with self.subTest(parameters=parameters):
                raw = self._make_three_eeg_raw()
                original_data = raw.get_data().copy()
                original_names = list(raw.ch_names)

                with self.assertRaisesRegex(error_type, message):
                    prepare_eeg({}, None, raw=raw, **parameters)

                self.assertEqual(raw.ch_names, original_names)
                self.assertIsNone(raw.get_montage())
                numpy.testing.assert_array_equal(raw.get_data(), original_data)

    def test_non_raw_input_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'raw must be an MNE Raw'):
            prepare_eeg({}, None, raw=object())

    def test_supplied_raw_honours_preload_true(self):
        source = self._make_three_eeg_raw()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'test_raw.fif'
            source.save(path, overwrite=True, verbose=False)
            lazy_raw = mne.io.read_raw_fif(
                path,
                preload=False,
                verbose=False,
            )
            self.assertFalse(lazy_raw.preload)

            prepared = prepare_eeg(
                {},
                None,
                raw=lazy_raw,
                preload=True,
            )

        self.assertIs(prepared, lazy_raw)
        self.assertTrue(prepared.preload)

    def test_existing_montage_is_not_replaced(self):
        raw = self._make_three_eeg_raw()
        custom_positions = {
            'Cz': (0.01, 0.02, 0.03),
            'Pz': (0.02, -0.01, 0.04),
            'Fz': (-0.02, 0.03, 0.05),
        }
        raw.set_montage(mne.channels.make_dig_montage(
            ch_pos=custom_positions,
            coord_frame='head',
        ))

        prepared = prepare_eeg({}, None, raw=raw)
        observed = prepared.get_montage().get_positions()['ch_pos']

        for channel, expected_position in custom_positions.items():
            numpy.testing.assert_allclose(
                observed[channel],
                expected_position,
                rtol=0,
                atol=1e-12,
            )

    def test_standard_montage_is_only_used_as_fallback(self):
        raw = self._make_three_eeg_raw()
        self.assertIsNone(raw.get_montage())

        prepared = prepare_eeg({}, None, raw=raw)

        self.assertIsNotNone(prepared.get_montage())
        self.assertIn(
            'Cz',
            prepared.get_montage().get_positions()['ch_pos'],
        )

    def test_resampling_uses_mne_antialiasing_and_same_rate_is_noop(self):
        for source_sfreq, target_sfreq in ((100, 100), (200, 100)):
            with self.subTest(
                source_sfreq=source_sfreq,
                target_sfreq=target_sfreq,
            ):
                raw = self._make_three_eeg_raw(sfreq=source_sfreq)
                with patch.object(raw, 'filter', wraps=raw.filter) as filter_mock:
                    prepared = prepare_eeg(
                        {},
                        None,
                        raw=raw,
                        preload=True,
                        resample_frequency=target_sfreq,
                    )

                filter_mock.assert_not_called()
                self.assertEqual(prepared.info['sfreq'], target_sfreq)

    def test_frequency_limits_accept_array_like_without_mutating_it(self):
        cases = (
            [1.0, 80.0],
            (1.0, 80.0),
            numpy.array([1.0, 80.0]),
        )

        for limits in cases:
            with self.subTest(type=type(limits).__name__):
                raw = self._make_three_eeg_raw(sfreq=100)
                before = numpy.asarray(limits).copy()

                prepared = prepare_eeg(
                    {},
                    None,
                    raw=raw,
                    preload=True,
                    freq_limits=limits,
                )

                numpy.testing.assert_array_equal(limits, before)
                self.assertEqual(prepared.info['highpass'], 1.0)

    def test_crop_must_leave_data_before_raw_is_mutated(self):
        raw = self._make_three_eeg_raw(sfreq=100, duration=10)
        original_first_samp = raw.first_samp
        original_last_samp = raw.last_samp

        with self.assertRaisesRegex(ValueError, 'must leave at least'):
            prepare_eeg({}, None, raw=raw, crop_seconds=5)

        self.assertEqual(raw.first_samp, original_first_samp)
        self.assertEqual(raw.last_samp, original_last_samp)
        self.assertIsNone(raw.get_montage())


class TestPrepareEegBadChannels(unittest.TestCase):
    """Validate deterministic and mutually exclusive bad-channel policies."""

    @staticmethod
    def _make_raw():
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz', 'Oz', 'STI 014'],
            sfreq=100,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'stim'],
        )
        return mne.io.RawArray(
            numpy.zeros((5, 1000)),
            info,
            verbose=False,
        )

    @staticmethod
    def _metadata(bad_channels):
        names = ['Fz', 'STI 014', 'Cz', 'Oz', 'Pz']
        return pandas.DataFrame({
            'name': names,
            'status': [
                'bad' if channel in bad_channels else 'good'
                for channel in names
            ],
        })

    def test_metadata_bad_channels_follow_raw_order(self):
        raw = self._make_raw()
        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_channels',
            return_value=self._metadata({'Fz', 'Cz'}),
        ):
            prepared = prepare_eeg(
                {},
                None,
                raw=raw,
                metadata_badchannels=True,
            )

        self.assertEqual(prepared.info['bads'], ['Cz', 'Fz'])

    def test_all_eeg_bad_is_rejected_even_with_good_stim_channel(self):
        raw = self._make_raw()
        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_channels',
            return_value=self._metadata({'Cz', 'Pz', 'Fz', 'Oz'}),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'All EEG channels are marked as bad',
            ):
                prepare_eeg(
                    {},
                    None,
                    raw=raw,
                    metadata_badchannels=True,
                )

    def test_excluded_bad_channels_are_absent_during_filtering(self):
        raw = self._make_raw()
        observed_channel_sets = []
        original_filter = raw.filter

        def record_channels(*args, **kwargs):
            observed_channel_sets.append(list(raw.ch_names))
            return original_filter(*args, **kwargs)

        with (
            patch(
                'sEEGnal.tools.mne_tools.bids_tool.read_channels',
                return_value=self._metadata({'Pz'}),
            ),
            patch.object(raw, 'filter', side_effect=record_channels),
        ):
            prepared = prepare_eeg(
                {},
                None,
                raw=raw,
                preload=True,
                freq_limits=[1, 40],
                exclude_badchannels=True,
            )

        self.assertNotIn('Pz', prepared.ch_names)
        self.assertEqual(len(observed_channel_sets), 1)
        self.assertNotIn('Pz', observed_channel_sets[0])

    def test_interpolation_retains_channel_and_resets_bad_metadata(self):
        raw = self._make_raw().pick(['Cz', 'Pz', 'Fz', 'Oz'])
        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_channels',
            return_value=self._metadata({'Pz'}),
        ):
            prepared = prepare_eeg(
                {},
                None,
                raw=raw,
                preload=True,
                interpolate_badchannels=True,
            )

        self.assertEqual(prepared.ch_names, ['Cz', 'Pz', 'Fz', 'Oz'])
        self.assertEqual(prepared.info['bads'], [])



class TestApplyIcaReference(unittest.TestCase):
    """Require ICA application and signal data to share one reference."""

    @staticmethod
    def _ica_config():
        """Return a minimal component-selection configuration."""
        return {
            'desc': 'cleaning',
            'components_to_include': ['brain'],
            'components_to_exclude': [],
        }

    @staticmethod
    def _make_three_channel_raw():
        """Return preloaded EEG data with deterministic channel order."""
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz'],
            sfreq=100,
            ch_types=['eeg', 'eeg', 'eeg'],
        )
        raw = mne.io.RawArray(
            numpy.zeros((3, 200)),
            info,
            verbose=False,
        )
        return prepare_eeg({}, None, raw=raw, rereference=False)

    @staticmethod
    def _make_ica(raw, ch_names, labels=None):
        """Build a minimal fitted-ICA double for compatibility checks."""
        ica_info = raw.copy().pick(ch_names).info.copy()
        reference_state = raw._seegnal_reference
        reference_channels = reference_state['channels']
        return SimpleNamespace(
            _seegnal_reference={
                **reference_state,
                'channels': (
                    None
                    if reference_channels is None
                    else list(reference_channels)
                ),
            },
            ch_names=list(ch_names),
            info=ica_info,
            labels_=labels or {'brain': [0]},
            n_components_=1,
            apply=Mock(),
        )

    def test_matching_ica_preserves_effective_reference(self):
        """ICA cleaning neither loses nor reapplies an average reference."""
        raw = prepare_eeg(
            {},
            None,
            raw=make_synthetic_raw(sfreq=1000, duration=2),
            preload=True,
            rereference='average',
        )
        reference_state = raw._seegnal_reference
        data_before = raw.get_data().copy()

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_ica',
            return_value=SimpleNamespace(
                _seegnal_reference={
                    **reference_state,
                    'channels': list(reference_state['channels']),
                },
                ch_names=['Cz'],
                info=raw.copy().pick(['Cz']).info.copy(),
                labels_={'brain': []},
                n_components_=1,
                apply=Mock(),
            ),
        ) as read_ica:
            ica = read_ica.return_value

            prepared = apply_ica(
                {},
                None,
                raw,
                self._ica_config(),
            )

        ica.apply.assert_called_once_with(raw, exclude=[0])
        self.assertIs(prepared, raw)
        self.assertIs(prepared._seegnal_reference, reference_state)
        self.assertTrue(prepared.info['projs'][0]['active'])
        numpy.testing.assert_array_equal(prepared.get_data(), data_before)

    def test_missing_or_different_ica_reference_is_rejected(self):
        """Untraceable and incompatible ICA fits fail before application."""
        raw = prepare_eeg(
            {},
            None,
            raw=make_synthetic_raw(sfreq=1000, duration=2),
            preload=True,
            rereference=False,
        )
        cases = (
            ('missing', None, 'does not record the reference'),
            (
                'different',
                {
                    'schema_version': 1,
                    'method': 'average',
                    'implementation': 'mne_projection',
                    'channels': ['Cz'],
                    'status': 'effective',
                },
                'does not match the effective data reference',
            ),
        )

        for name, reference_state, message in cases:
            with self.subTest(case=name):
                with patch(
                    'sEEGnal.tools.mne_tools.bids_tool.read_ica',
                    return_value=SimpleNamespace(apply=Mock()),
                ) as read_ica:
                    ica = read_ica.return_value
                    if reference_state is not None:
                        ica._seegnal_reference = reference_state

                    with self.assertRaisesRegex(RuntimeError, message):
                        apply_ica(
                            {},
                            None,
                            raw,
                            self._ica_config(),
                        )

                ica.apply.assert_not_called()

    def test_channel_space_incompatibilities_are_rejected(self):
        """Missing, bad, reordered and extra good EEG channels fail early."""
        fitted_raw = self._make_three_channel_raw()
        ica = self._make_ica(fitted_raw, ['Cz', 'Pz'])

        cases = []

        missing = fitted_raw.copy().drop_channels(['Pz'])
        cases.append((missing, 'missing channel'))

        reordered = fitted_raw.copy().reorder_channels(['Pz', 'Cz', 'Fz'])
        cases.append((reordered, 'not in their fitting order'))

        fitted_bad = fitted_raw.copy()
        fitted_bad.info['bads'] = ['Pz']
        cases.append((fitted_bad, 'used to fit the ICA are marked bad'))

        cases.append((fitted_raw.copy(), 'would remain uncleaned'))

        for raw, message in cases:
            with self.subTest(message=message):
                with patch(
                    'sEEGnal.tools.mne_tools.bids_tool.read_ica',
                    return_value=ica,
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        apply_ica({}, None, raw, self._ica_config())

        ica.apply.assert_not_called()

    def test_extra_bad_eeg_channel_is_retained_and_left_untouched(self):
        """Feature workflows may retain an EEG channel outside ica.ch_names."""
        raw = self._make_three_channel_raw()
        raw.info['bads'] = ['Fz']
        ica = self._make_ica(raw, ['Cz', 'Pz'])
        data_before = raw.get_data().copy()

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_ica',
            return_value=ica,
        ):
            prepared = apply_ica({}, None, raw, self._ica_config())

        self.assertIs(prepared, raw)
        ica.apply.assert_not_called()
        numpy.testing.assert_array_equal(raw.get_data(), data_before)

    def test_incompatible_channel_metadata_are_rejected(self):
        """ICA channel units must describe the same physical signals."""
        fitted_raw = self._make_three_channel_raw()
        ica = self._make_ica(fitted_raw, fitted_raw.ch_names)
        application_raw = fitted_raw.copy()
        application_raw.info['chs'][0]['unit_mul'] = -6

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_ica',
            return_value=ica,
        ):
            with self.assertRaisesRegex(RuntimeError, 'metadata field'):
                apply_ica(
                    {},
                    None,
                    application_raw,
                    self._ica_config(),
                )

        ica.apply.assert_not_called()

    def test_storage_calibration_fields_do_not_define_compatibility(self):
        """FIF-normalized cal and range values do not reject valid data."""
        fitted_raw = self._make_three_channel_raw()
        ica = self._make_ica(fitted_raw, fitted_raw.ch_names)
        application_raw = fitted_raw.copy()
        application_raw.info['chs'][0]['cal'] = 0.1
        application_raw.info['chs'][0]['range'] = 1e-6

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_ica',
            return_value=ica,
        ):
            prepared = apply_ica(
                {},
                None,
                application_raw,
                self._ica_config(),
            )

        self.assertIs(prepared, application_raw)


class TestPrepareEegEpoching(unittest.TestCase):
    """Check that prepare_eeg delegates fixed and event epoching correctly."""

    def test_prepare_eeg_creates_fixed_epochs_without_crop(self):
        sfreq = 1000
        duration = 2
        epoch_samples = int(duration * sfreq)
        raw = make_synthetic_raw(
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

        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            epoch_definition=epoch_definition
        )

        self.assertIsInstance(epochs, mne.BaseEpochs)
        self.assertEqual(len(epochs), 6)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.arange(6) * epoch_samples
        )
        self.assertEqual(epochs.get_data().shape, (6, 2, epoch_samples))
        self.assertEqual(epochs.drop_log, ((), (), (), (), (), ()))

        for epoch_index in range(6):
            assert_epoch_contains_samples(epochs, epoch_index)

    def test_prepare_eeg_creates_event_epochs_without_crop(self):
        sfreq = 1000
        event_onsets = (2, 5, 8)
        raw = make_synthetic_raw(
            sfreq=sfreq,
            duration=12,
            first_samp=0,
            annotations=tuple(
                (onset, 0, 'target') for onset in event_onsets
            )
        )
        epoch_definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': {'target': 1},
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
            'reject_by_annotation': True,
        }

        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            epoch_definition=epoch_definition
        )

        self.assertIsInstance(epochs, mne.BaseEpochs)
        self.assertEqual(len(epochs), 3)
        self.assertEqual(epochs.event_id, {'target': 1})
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.array(event_onsets) * sfreq
        )
        self.assertEqual(epochs.get_data().shape, (3, 2, sfreq + 1))
        self.assertEqual(epochs.drop_log, ((), (), ()))

        for epoch_index in range(3):
            assert_epoch_contains_samples(epochs, epoch_index)


class TestAnnotationTimeWindow(unittest.TestCase):
    """Check half-open clipping without unnecessary float reconstruction."""

    def test_fully_contained_annotation_preserves_duration_exactly(self):
        onset = numpy.float64(44.3)
        duration = numpy.float64(0.3)

        clipped = _clip_annotation_to_time_window(
            onset,
            duration,
            tmin=0,
            tmax=100,
        )

        self.assertEqual(clipped[0], onset)
        self.assertEqual(clipped[1], duration)
        reconstructed_duration = (
            float(onset) + float(duration)
        ) - float(onset)
        self.assertNotEqual(reconstructed_duration, duration)

    def test_partially_contained_annotations_are_still_clipped(self):
        self.assertEqual(
            _clip_annotation_to_time_window(4.75, 0.5, 5, 15),
            (5, 0.25),
        )
        self.assertEqual(
            _clip_annotation_to_time_window(14.75, 0.5, 5, 15),
            (14.75, 0.25),
        )


class TestPrepareEegCrop(unittest.TestCase):
    """Check crop alone and followed by fixed or event epoching."""

    def _make_bids_path(self, root):
        return mne_bids.BIDSPath(
            subject='023',
            session='0',
            task='1EC',
            datatype='eeg',
            suffix='eeg',
            extension='.vhdr',
            root=Path(root),
        )

    def test_prepare_eeg_crops_raw_at_both_ends(self):
        sfreq = 1000
        raw = make_synthetic_raw(
            sfreq=sfreq,
            duration=20,
            first_samp=0
        )

        cropped = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=2
        )

        self.assertIsInstance(cropped, mne.io.BaseRaw)
        self.assertEqual(cropped.first_samp, 2000)
        self.assertEqual(cropped.last_samp, 17999)
        self.assertEqual(cropped.n_times, 16000)
        self.assertEqual(cropped.original_first_samp, 0)
        self.assertEqual(cropped.original_last_samp, 19999)
        self.assertAlmostEqual(cropped.times[0], 0)
        self.assertAlmostEqual(cropped.times[-1], 15.999)
        cropped_eeg = cropped.get_data(picks=['Cz'])[0]
        self.assertEqual(cropped_eeg[0], 2000 * SAMPLE_SCALE)
        self.assertEqual(cropped_eeg[-1], 17999 * SAMPLE_SCALE)

    def test_prepare_eeg_crops_before_creating_fixed_epochs(self):
        sfreq = 1000
        raw = make_synthetic_raw(
            sfreq=sfreq,
            duration=20,
            first_samp=0
        )
        epoch_definition = {
            'mode': 'fixed',
            'duration': 2,
            'overlap': 0,
            'reject_by_annotation': False,
        }

        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=2,
            epoch_definition=epoch_definition
        )

        self.assertEqual(len(epochs), 8)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.arange(2000, 18000, 2000)
        )
        self.assertEqual(epochs.drop_log, ((),) * 8)
        self.assertEqual(epochs.original_first_samp, 0)
        self.assertEqual(epochs.original_last_samp, 19999)
        assert_epoch_contains_samples(epochs, epoch_index=0)
        assert_epoch_contains_samples(epochs, epoch_index=7)

    def test_prepare_eeg_crops_annotations_before_event_epoching(self):
        sfreq = 1000
        raw = make_synthetic_raw(
            sfreq=sfreq,
            duration=20,
            first_samp=0,
            annotations=(
                (1, 0, 'target'),
                (3, 0, 'target'),
                (10, 0, 'target'),
                (19, 0, 'target'),
            )
        )
        epoch_definition = {
            'mode': 'events',
            'event_source': 'annotations',
            'event_id': {'target': 1},
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
            'reject_by_annotation': True,
        }

        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=2,
            epoch_definition=epoch_definition
        )

        self.assertEqual(len(epochs), 2)
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.array([3000, 10000])
        )
        self.assertEqual(epochs.event_id, {'target': 1})
        self.assertEqual(epochs.drop_log, ((), ()))
        self.assertEqual(epochs.original_first_samp, 0)
        self.assertEqual(epochs.original_last_samp, 19999)
        assert_epoch_contains_samples(epochs, epoch_index=0)
        assert_epoch_contains_samples(epochs, epoch_index=1)

    def test_prepare_eeg_creates_stim_epochs_without_annotations(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=8,
            first_samp=1000,
            stim_events=((2, 32), (5, 32))
        )
        epoch_definition = {
            'mode': 'events',
            'event_source': 'stim_channel',
            'event_id': 32,
            'tmin': 0,
            'tmax': 1,
            'baseline': None,
        }

        epochs = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            epoch_definition=epoch_definition
        )

        self.assertEqual(epochs.event_id, {'32': 32})
        numpy.testing.assert_array_equal(
            epochs.events[:, 0],
            numpy.array([3000, 6000])
        )
        self.assertEqual(epochs.get_data().shape, (2, 2, 1001))

    def test_successive_crops_keep_the_true_original_bounds(self):
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
            crop_seconds=2
        )

        self.assertEqual(second_crop.first_samp, 5000)
        self.assertEqual(second_crop.last_samp, 16999)
        self.assertEqual(second_crop.original_first_samp, 1000)
        self.assertEqual(second_crop.original_last_samp, 20999)

    def test_real_saved_annotations_load_without_crop(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=1000,
        )
        saved_annotations = mne.Annotations(
            onset=[7.25, 12.5],
            duration=[0.5, 0.25],
            description=['bad_muscle', 'bad_EOG'],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {'subsystem': 'preprocess'}
            bids_path = self._make_bids_path(temporary_directory)
            write_annotations(config, bids_path, saved_annotations)

            loaded = prepare_eeg(
                config,
                bids_path,
                raw=raw,
                preload=True,
                set_annotations=True,
            )

        events, event_id = mne.events_from_annotations(
            loaded,
            regexp=None,
            verbose=False,
        )

        self.assertEqual(loaded.first_samp, 1000)
        self.assertEqual(loaded.original_first_samp, 1000)
        self.assertEqual(loaded.original_last_samp, 20999)
        self.assertEqual(event_id, {'bad_EOG': 1, 'bad_muscle': 2})
        numpy.testing.assert_array_equal(
            events[:, 0],
            numpy.array([8250, 13500]),
        )
        numpy.testing.assert_allclose(
            loaded.annotations.onset,
            numpy.array([8.25, 13.5]),
        )
        numpy.testing.assert_allclose(
            loaded.annotations.duration,
            saved_annotations.duration,
        )
        numpy.testing.assert_array_equal(
            loaded.annotations.duration,
            saved_annotations.duration,
        )

    def test_real_saved_annotations_load_with_crop_in_either_call_order(self):
        saved_annotations = mne.Annotations(
            onset=[7.25],
            duration=[0.5],
            description=['bad_muscle'],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {'subsystem': 'preprocess'}
            bids_path = self._make_bids_path(temporary_directory)
            write_annotations(config, bids_path, saved_annotations)

            for crop_then_load in (False, True):
                with self.subTest(crop_then_load=crop_then_load):
                    raw = make_synthetic_raw(
                        sfreq=1000,
                        duration=20,
                        first_samp=1000,
                    )

                    if crop_then_load:
                        raw = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            crop_seconds=5,
                        )
                        loaded = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            set_annotations=True,
                        )
                    else:
                        loaded = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            crop_seconds=5,
                            set_annotations=True,
                        )

                    events, event_id = mne.events_from_annotations(
                        loaded,
                        regexp=None,
                        verbose=False,
                    )

                    self.assertEqual(loaded.first_samp, 6000)
                    self.assertEqual(loaded.last_samp, 15999)
                    self.assertEqual(loaded.original_first_samp, 1000)
                    self.assertEqual(loaded.original_last_samp, 20999)
                    self.assertEqual(event_id, {'bad_muscle': 1})
                    numpy.testing.assert_array_equal(
                        events[:, 0],
                        numpy.array([8250]),
                    )
                    numpy.testing.assert_allclose(
                        loaded.annotations.onset,
                        numpy.array([8.25]),
                    )
                    numpy.testing.assert_allclose(
                        loaded.annotations.duration,
                        numpy.array([0.5]),
                    )

    def test_real_saved_annotations_follow_half_open_crop_boundaries(self):
        saved_annotations = mne.Annotations(
            onset=[4, 4.5, 4.75, 5, 7.25, 14.75, 14.999, 15, 16],
            duration=[0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.5, 0.5],
            description=[
                'before',
                'touch_left',
                'overlap_left',
                'at_left',
                'inside',
                'overlap_right',
                'last_sample',
                'touch_right',
                'after',
            ],
        )
        expected = {
            'overlap_left': (6.0, 0.25),
            'at_left': (6.0, 0.5),
            'inside': (8.25, 0.5),
            'overlap_right': (15.75, 0.25),
            'last_sample': (15.999, 0.001),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {'subsystem': 'preprocess'}
            bids_path = self._make_bids_path(temporary_directory)
            write_annotations(config, bids_path, saved_annotations)

            for crop_then_load in (False, True):
                with self.subTest(crop_then_load=crop_then_load):
                    raw = make_synthetic_raw(
                        sfreq=1000,
                        duration=20,
                        first_samp=1000,
                    )

                    if crop_then_load:
                        raw = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            crop_seconds=5,
                        )
                        loaded = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            set_annotations=True,
                        )
                    else:
                        loaded = prepare_eeg(
                            config,
                            bids_path,
                            raw=raw,
                            preload=True,
                            crop_seconds=5,
                            set_annotations=True,
                        )

                    loaded_again = prepare_eeg(
                        config,
                        bids_path,
                        raw=loaded,
                        preload=True,
                        set_annotations=True,
                    )

                    self.assertIs(loaded_again, loaded)
                    self.assertEqual(len(loaded.annotations), len(expected))
                    observed = {
                        description: (onset, duration)
                        for onset, duration, description in zip(
                            loaded.annotations.onset,
                            loaded.annotations.duration,
                            loaded.annotations.description,
                        )
                    }
                    self.assertEqual(set(observed), set(expected))

                    for description, expected_values in expected.items():
                        numpy.testing.assert_allclose(
                            observed[description],
                            expected_values,
                            atol=1e-12,
                        )

    def test_saved_annotations_loaded_after_crop_use_original_time_axis(self):
        sfreq = 1000
        raw = make_synthetic_raw(
            sfreq=sfreq,
            duration=20,
            first_samp=1000,
            annotations=((6, 0, 'target'),)
        )
        saved_annotations = mne.Annotations(
            onset=[7.25],
            duration=[0.5],
            description=['bad_muscle']
        )

        cropped = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=5
        )

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_annotations',
            return_value=saved_annotations
        ):
            loaded = prepare_eeg(
                {},
                None,
                raw=cropped,
                preload=True,
                set_annotations=True
            )

        events, event_id = mne.events_from_annotations(
            loaded,
            regexp=None,
            verbose=False
        )

        self.assertEqual(loaded.first_samp, 6000)
        self.assertEqual(loaded.original_first_samp, 1000)
        self.assertEqual(loaded.original_last_samp, 20999)
        self.assertEqual(event_id, {'bad_muscle': 1, 'target': 2})
        numpy.testing.assert_array_equal(
            events[:, 0],
            numpy.array([7000, 8250])
        )
        numpy.testing.assert_allclose(
            loaded.annotations.onset,
            numpy.array([7, 8.25])
        )

    def test_loading_saved_annotations_twice_does_not_duplicate_them(self):
        raw = make_synthetic_raw(
            sfreq=1000,
            duration=20,
            first_samp=0
        )
        saved_annotations = mne.Annotations(
            onset=[7.25],
            duration=[0.5],
            description=['bad_muscle']
        )

        cropped = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            crop_seconds=5
        )

        with patch(
            'sEEGnal.tools.mne_tools.bids_tool.read_annotations',
            return_value=saved_annotations
        ):
            loaded = prepare_eeg(
                {},
                None,
                raw=cropped,
                preload=True,
                set_annotations=True
            )
            loaded_again = prepare_eeg(
                {},
                None,
                raw=loaded,
                preload=True,
                set_annotations=True
            )

        self.assertIs(loaded_again, loaded)
        self.assertEqual(len(loaded_again.annotations), 1)
        self.assertEqual(loaded_again.annotations.description[0], 'bad_muscle')
        self.assertAlmostEqual(loaded_again.annotations.onset[0], 7.25)


if __name__ == '__main__':
    unittest.main()
