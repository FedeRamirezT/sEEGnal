"""Tests for source-device metadata preserved during BIDS standardization."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import mne_bids
import numpy

from sEEGnal.standardize.standardize import (
    _update_source_metadata_sidecar,
    standardize_eeg_file,
)


class TestSourceMetadataSidecar(unittest.TestCase):
    """Override output-format inference only when the source supplies metadata."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.bids_path = mne_bids.BIDSPath(
            subject='001',
            session='0',
            task='test',
            run='1',
            datatype='eeg',
            suffix='eeg',
            extension='.vhdr',
            root=Path(self.temporary_directory.name),
        )
        self.sidecar = self.bids_path.copy().update(
            suffix='eeg',
            extension='.json',
        ).fpath
        self.sidecar.parent.mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_sidecar(self):
        self.sidecar.write_text(
            json.dumps({
                'TaskName': 'test',
                'Manufacturer': 'Brain Products',
                'SamplingFrequency': 500,
            }),
            encoding='utf-8',
        )

    def test_nedf_metadata_overrides_inferred_manufacturer(self):
        self.write_sidecar()

        _update_source_metadata_sidecar(self.bids_path, {
            'Manufacturer': 'Neuroelectrics',
            'ManufacturersModelName': 'Enobio 20',
            'SoftwareVersions': 'NIC v2.0.11.7; Firmware 3026',
        })

        metadata = json.loads(self.sidecar.read_text(encoding='utf-8'))
        self.assertEqual(metadata['Manufacturer'], 'Neuroelectrics')
        self.assertEqual(metadata['ManufacturersModelName'], 'Enobio 20')
        self.assertEqual(
            metadata['SoftwareVersions'],
            'NIC v2.0.11.7; Firmware 3026',
        )
        self.assertEqual(metadata['TaskName'], 'test')
        self.assertEqual(metadata['SamplingFrequency'], 500)

    def test_missing_source_metadata_leaves_sidecar_unchanged(self):
        self.write_sidecar()
        before = self.sidecar.read_bytes()

        _update_source_metadata_sidecar(self.bids_path, None)

        self.assertEqual(self.sidecar.read_bytes(), before)

    def test_standardization_applies_metadata_supplied_by_source_reader(self):
        raw = mne.io.RawArray(
            numpy.zeros((1, 500)),
            mne.create_info(['Cz'], 500, ['eeg']),
            verbose=False,
        )
        raw._seegnal_source_bids_metadata = {
            'Manufacturer': 'Neuroelectrics',
            'ManufacturersModelName': 'Enobio 20',
            'SoftwareVersions': 'NIC 2.1.3.6; Firmware 3039',
        }
        config = {
            'path': {'sourcedata': str(self.bids_path.root)},
            'global': {
                'channels_to_include': 'eeg',
                'channels_to_exclude': [],
                'line_freq': 50,
            },
            'preprocess': {
                'standardization': {
                    'format': 'BrainVision',
                    'overwrite': 1,
                },
            },
        }

        with patch(
            'sEEGnal.standardize.standardize.read_source_files',
            return_value=raw,
        ):
            standardize_eeg_file(config, 'recording.nedf', self.bids_path)

        metadata = json.loads(self.sidecar.read_text(encoding='utf-8'))
        self.assertEqual(metadata['Manufacturer'], 'Neuroelectrics')
        self.assertEqual(metadata['ManufacturersModelName'], 'Enobio 20')
        self.assertEqual(
            metadata['SoftwareVersions'],
            'NIC 2.1.3.6; Firmware 3039',
        )


if __name__ == '__main__':
    unittest.main()
