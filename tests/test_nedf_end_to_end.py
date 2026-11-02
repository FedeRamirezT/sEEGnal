"""End-to-end regression test for NEDF to BIDS standardization."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mne
import numpy

from sEEGnal.io.nedf import (
    NEDF_ADC_SCALE_VOLTS,
    NEDF_MISREAD_MIN_VOLTS,
)
from sEEGnal.io.recordings import validate_recordings
from sEEGnal.standardize.standardize import standardize


NEDF_HEADER_SIZE = 10_240


def _encode_int24(value):
    """Encode one signed 24-bit EEG count as big-endian bytes."""

    if not -(1 << 23) <= value < (1 << 23):
        raise ValueError(f'{value} is outside the signed 24-bit range')
    return (value & 0xFFFFFF).to_bytes(3, byteorder='big')


def _write_synthetic_nedf(path):
    """Write a small NEDF 1.4 file that MNE reads without test doubles."""

    channel_names = ('Fz', 'Cz')
    eeg_counts = (
        (0, 1, -(1 << 23), (1 << 23) - 1, -1, -7, 2, 0, 0, 0),
        (10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
    )
    triggers = (0, 7, 0, 255, 255, 0, 11, 0, 0, 0)
    montage = ''.join(
        f'<Channel{index}>{name}</Channel{index}>'
        for index, name in enumerate(channel_names, start=1)
    )
    header_xml = (
        '<Recording>'
        '<NEDFversion>1.4</NEDFversion>'
        '<NumberOfChannelsOfAccelerometer>3</NumberOfChannelsOfAccelerometer>'
        '<StepDetails>'
        '<DeviceClass>ENOBIO</DeviceClass>'
        '<SoftwareVersion>NIC v2.0.11.7</SoftwareVersion>'
        '<FirmwareVersion>3026</FirmwareVersion>'
        '<StartDate_firstEEGTimestamp>'
        '1767225600000'
        '</StartDate_firstEEGTimestamp>'
        '</StepDetails>'
        '<EEGSettings>'
        '<TotalNumberOfChannels>2</TotalNumberOfChannels>'
        '<NumberOfEEGChannels>2</NumberOfEEGChannels>'
        '<NumberOfRecordsOfEEG>10</NumberOfRecordsOfEEG>'
        '<EEGSamplingRate>500</EEGSamplingRate>'
        f'<EEGMontage>{montage}</EEGMontage>'
        '</EEGSettings>'
        '</Recording>'
    ).encode('utf-8')
    if len(header_xml) >= NEDF_HEADER_SIZE:
        raise AssertionError('synthetic NEDF header is too large')

    binary = bytearray()
    for block_start in range(0, len(triggers), 5):
        binary.extend(bytes(6))  # One 3-axis uint16 accelerometer sample.
        for sample in range(block_start, block_start + 5):
            for channel in eeg_counts:
                binary.extend(_encode_int24(channel[sample]))
            binary.extend(triggers[sample].to_bytes(4, byteorder='big'))

    path.write_bytes(
        header_xml
        + bytes(NEDF_HEADER_SIZE - len(header_xml))
        + binary
    )
    return numpy.asarray(eeg_counts, dtype=float) * NEDF_ADC_SCALE_VOLTS


class TestNedfToBidsEndToEnd(unittest.TestCase):
    """Exercise the manifest and real NEDF reader through BIDS output."""

    def test_synthetic_nedf_is_standardized_without_losing_semantics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sourcedata = root / 'sourcedata' / 'eeg'
            sourcedata.mkdir(parents=True)
            source = sourcedata / 'synthetic.nedf'
            expected_eeg = _write_synthetic_nedf(source)
            source_digest = hashlib.sha256(source.read_bytes()).digest()

            manifest = root / 'recordings.tsv'
            manifest.write_text(
                'file\tsubject\tsession\ttask\trun\n'
                'synthetic.nedf\t001\t0\tPR\t1\n',
                encoding='utf-8',
            )
            config = {
                'path': {
                    'data_root': str(root),
                    'sourcedata': str(sourcedata),
                    'recordings': str(manifest),
                },
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

            recordings = validate_recordings(config)
            self.assertEqual(len(recordings), 1)
            recording = recordings[0]
            result = standardize(
                config,
                recording['file'],
                recording['bids_path'],
            )
            self.assertEqual(result['result'], 'ok', result.get('details'))

            bids_path = recording['bids_path']
            standardized = mne.io.read_raw_brainvision(
                bids_path.fpath,
                preload=True,
                verbose='ERROR',
            )
            self.assertEqual(standardized.ch_names, ['Fz', 'Cz'])
            self.assertEqual(standardized.n_times, 10)
            self.assertEqual(standardized.info['sfreq'], 500)
            numpy.testing.assert_allclose(
                standardized.get_data(),
                expected_eeg,
                atol=2 * NEDF_ADC_SCALE_VOLTS,
                rtol=1e-6,
            )
            self.assertAlmostEqual(
                standardized.get_data(picks=['Fz'])[0, 2],
                -NEDF_MISREAD_MIN_VOLTS,
                delta=2 * NEDF_ADC_SCALE_VOLTS,
            )

            events_path = bids_path.copy().update(
                suffix='events',
                extension='.tsv',
            ).fpath
            with events_path.open(encoding='utf-8', newline='') as file:
                event_rows = list(csv.DictReader(file, delimiter='\t'))
            events = {row['trial_type']: row for row in event_rows}
            self.assertEqual(
                set(events),
                {'Stimulus/s7', 'Stimulus/s11', 'BAD_ACQ_SKIP'},
            )
            self.assertEqual(int(events['Stimulus/s7']['value']), 7)
            self.assertEqual(int(events['Stimulus/s11']['value']), 11)
            self.assertEqual(int(events['BAD_ACQ_SKIP']['value']), 10_000)
            self.assertNotIn('Stimulus/s255', events)
            self.assertAlmostEqual(
                float(events['Stimulus/s7']['onset']),
                0.002,
            )
            self.assertAlmostEqual(
                float(events['Stimulus/s11']['onset']),
                0.012,
            )
            self.assertAlmostEqual(
                float(events['BAD_ACQ_SKIP']['onset']),
                0.006,
            )
            self.assertAlmostEqual(
                float(events['BAD_ACQ_SKIP']['duration']),
                0.004,
            )

            sidecar = bids_path.copy().update(
                suffix='eeg',
                extension='.json',
            ).fpath
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            self.assertEqual(metadata['Manufacturer'], 'Neuroelectrics')
            self.assertEqual(metadata['ManufacturersModelName'], 'Enobio 2')
            self.assertEqual(
                metadata['SoftwareVersions'],
                'NIC v2.0.11.7; Firmware 3026',
            )
            self.assertEqual(metadata['SamplingFrequency'], 500)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).digest(),
                source_digest,
            )


if __name__ == '__main__':
    unittest.main()
