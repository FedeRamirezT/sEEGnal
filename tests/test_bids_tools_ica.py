"""Tests for ICA reference provenance across derivative storage."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import mne
import mne_bids
import numpy

from sEEGnal.tools.bids_tools import read_ica, write_ica
from sEEGnal.tools.mne_tools import apply_ica, fit_ica, prepare_eeg


class TestIcaReferenceRoundTrip(unittest.TestCase):
    """Persist the effective fitting reference outside the ICA FIF file."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = {'subsystem': 'preprocess'}
        self.bids_path = mne_bids.BIDSPath(
            subject='023',
            session='0',
            task='1EC',
            datatype='eeg',
            suffix='eeg',
            extension='.vhdr',
            root=self.root,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _fit_average_referenced_ica():
        """Fit a small deterministic ICA with effective average reference."""
        sfreq = 200
        times = numpy.arange(800) / sfreq
        data = 1e-6 * numpy.vstack((
            numpy.sin(2 * numpy.pi * 7 * times),
            numpy.cos(2 * numpy.pi * 11 * times),
            numpy.sin(2 * numpy.pi * 17 * times + 0.3),
        ))
        info = mne.create_info(
            ['Cz', 'Pz', 'Fz'],
            sfreq=sfreq,
            ch_types='eeg',
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        # Reproduce acquisition metadata that are not invariant under FIF
        # serialization: MNE resets range to one and stores cal as float32.
        for channel in raw.info['chs']:
            channel['cal'] = 0.1
            channel['range'] = 1e-6
        raw.filter(1, None, verbose=False)
        raw = prepare_eeg(
            {},
            None,
            raw=raw,
            preload=True,
            rereference='average',
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ica = fit_ica(
                raw,
                n_components=2,
                random_state=42,
                max_iter=200,
            )

        return ica, raw

    def test_reference_metadata_survives_real_ica_round_trip(self):
        """The JSON restores the fitting reference after reading the FIF."""
        ica, raw = self._fit_average_referenced_ica()
        reference_state = raw._seegnal_reference
        ica.labels_ = {'brain': [0]}

        created_files = write_ica(
            self.config,
            self.bids_path,
            ica,
            desc='cleaning',
        )
        metadata_file = next(
            Path(path)
            for path in created_files
            if str(path).endswith('desc-cleaning_ica.json')
        )
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        loaded = read_ica(
            self.config,
            self.bids_path,
            desc='cleaning',
        )

        self.assertEqual(metadata['Reference'], reference_state)
        self.assertEqual(loaded._seegnal_reference, reference_state)
        self.assertIsNot(loaded._seegnal_reference, reference_state)
        self.assertIsNot(
            loaded._seegnal_reference['channels'],
            reference_state['channels'],
        )
        self.assertNotEqual(
            loaded.info['chs'][0]['range'],
            raw.info['chs'][0]['range'],
        )

        application_raw = raw.copy()
        data_before = application_raw.get_data().copy()
        prepared = apply_ica(
            self.config,
            self.bids_path,
            application_raw,
            {
                'desc': 'cleaning',
                'components_to_include': ['brain'],
                'components_to_exclude': [],
            },
        )

        self.assertIs(prepared, application_raw)
        self.assertFalse(
            numpy.array_equal(application_raw.get_data(), data_before)
        )

    def test_writer_rejects_ica_without_reference_provenance(self):
        """A fitted but untraceable ICA cannot enter the derivative store."""
        ica, _ = self._fit_average_referenced_ica()
        del ica._seegnal_reference

        with self.assertRaisesRegex(
            RuntimeError,
            'must record the effective fitting reference',
        ):
            write_ica(
                self.config,
                self.bids_path,
                ica,
                desc='untraceable',
            )


if __name__ == '__main__':
    unittest.main()
