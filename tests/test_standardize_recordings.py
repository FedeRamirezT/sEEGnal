"""Tests for the public recordings.tsv input contract."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quickstart.init.init import init as quickstart_init
from quickstart.init.init import load_config
from quickstart.run_sEEGnal import main as quickstart_main
from sEEGnal.io.recordings import (
    RecordingsValidationError,
    validate_recordings,
)
from sEEGnal.tools.bids_tools import build_BIDS_object


class TestRecordingsManifest(unittest.TestCase):
    """Validate complete manifests before any processing starts."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.sourcedata = self.root / 'sourcedata' / 'eeg'
        self.sourcedata.mkdir(parents=True)
        self.manifest = self.root / 'recordings.tsv'
        self.config = {
            'path': {
                'data_root': str(self.root),
                'sourcedata': str(self.sourcedata),
                'recordings': str(self.manifest),
            }
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_public_manifest_has_the_required_header(self):
        template = (
            Path(__file__).resolve().parents[1]
            / 'quickstart'
            / 'data'
            / 'recordings.tsv'
        )

        self.assertEqual(
            template.read_text(encoding='utf-8').splitlines()[0],
            'file\tsubject\tsession\ttask\trun',
        )

    def write_manifest(self, *rows, header=None):
        """Write a manifest with the public header and supplied rows."""

        if header is None:
            header = 'file\tsubject\tsession\ttask\trun'
        contents = '\n'.join((header, *rows)) + '\n'
        self.manifest.write_text(contents, encoding='utf-8')

    def write_brainvision(self, stem):
        """Write the minimal structural files required by validation."""

        header = self.sourcedata / f'{stem}.vhdr'
        marker = self.sourcedata / f'{stem}.vmrk'
        data = self.sourcedata / f'{stem}.eeg'
        self.write_brainvision_header(header, data.name, marker.name)
        marker.write_text('Brain Vision marker file', encoding='utf-8')
        data.write_bytes(b'')
        return header

    def write_nedf(
        self,
        stem,
        *,
        suffix='.nedf',
        version='1.4',
        channel_names=('Fz', 'Cz'),
        eeg_channels=None,
        eeg_records=2,
        sampling_rate=500,
        binary_records=None,
    ):
        """Write a small NEDF file sufficient for structural validation."""

        if eeg_channels is None:
            eeg_channels = len(channel_names)
        if binary_records is None:
            binary_records = eeg_records
        montage = ''.join(
            f'<Channel{index}>{name}</Channel{index}>'
            for index, name in enumerate(channel_names, start=1)
        )
        xml = (
            '<Recording>'
            f'<NEDFversion>{version}</NEDFversion>'
            '<EEGSettings>'
            f'<TotalNumberOfChannels>{eeg_channels}</TotalNumberOfChannels>'
            f'<NumberOfEEGChannels>{eeg_channels}</NumberOfEEGChannels>'
            f'<NumberOfRecordsOfEEG>{eeg_records}</NumberOfRecordsOfEEG>'
            f'<EEGSamplingRate>{sampling_rate}</EEGSamplingRate>'
            f'<EEGMontage>{montage}</EEGMontage>'
            '</EEGSettings>'
            '</Recording>'
        ).encode('utf-8')
        self.assertLess(len(xml), 10_240)
        path = self.sourcedata / f'{stem}{suffix}'
        header = xml + bytes(10_240 - len(xml))
        data = bytes(eeg_channels * binary_records * 3)
        path.write_bytes(header + data)
        return path

    @staticmethod
    def write_brainvision_header(header, data_reference, marker_reference):
        """Write a header with explicit portable companion references."""

        header.write_text(
            '\n'.join((
                'Brain Vision Data Exchange Header File Version 1.0',
                '[Common Infos]',
                f'DataFile={data_reference}',
                f'MarkerFile={marker_reference}',
            )),
            encoding='utf-8',
        )

    def test_valid_manifest_preserves_strings_and_builds_run_path(self):
        self.write_brainvision('unrelated-original-name')
        self.write_manifest(
            'unrelated-original-name.vhdr\t001\t01\trest\t01'
        )

        recordings = validate_recordings(self.config)

        self.assertEqual(len(recordings), 1)
        recording = recordings[0]
        self.assertEqual(recording['subject'], '001')
        self.assertEqual(recording['session'], '01')
        self.assertEqual(recording['run'], '01')
        self.assertEqual(recording['bids_path'].run, '01')
        self.assertEqual(
            recording['bids_path'].basename,
            'sub-001_ses-01_task-rest_run-01_eeg.vhdr',
        )

    def test_only_supported_source_formats_are_accepted(self):
        (self.sourcedata / 'original.set').write_bytes(b'EEGLAB')
        self.write_manifest('original.set\t002\t01\trest\t01')

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn('unsupported file extension', message)
        self.assertIn('expected one of: .vhdr, .nedf', message)

    def test_nedf_source_is_accepted(self):
        self.write_nedf('original')
        self.write_manifest('original.nedf\t002\t01\trest\t01')

        recordings = validate_recordings(self.config)

        self.assertEqual(recordings[0]['file'], 'original.nedf')

    def test_mixed_brainvision_and_nedf_sources_are_accepted(self):
        self.write_brainvision('brainvision')
        self.write_nedf('neuroelectrics')
        self.write_manifest(
            'brainvision.vhdr\t001\t01\trest\t01',
            'neuroelectrics.nedf\t002\t01\trest\t01',
        )

        recordings = validate_recordings(self.config)

        self.assertEqual(
            [recording['file'] for recording in recordings],
            ['brainvision.vhdr', 'neuroelectrics.nedf'],
        )

    def test_uppercase_nedf_extension_is_accepted(self):
        self.write_nedf('ORIGINAL', suffix='.NEDF')
        self.write_manifest('ORIGINAL.NEDF\t001\t01\trest\t01')

        recordings = validate_recordings(self.config)

        self.assertEqual(recordings[0]['file'], 'ORIGINAL.NEDF')

    def test_invalid_nedf_structure_is_reported(self):
        (self.sourcedata / 'short.nedf').write_bytes(b'not a NEDF header')
        self.write_nedf('version', version='2.0')
        self.write_nedf('montage', eeg_channels=2, channel_names=('Fz',))
        self.write_nedf('truncated', eeg_records=10, binary_records=1)
        self.write_manifest(
            'short.nedf\t001\t01\trest\t01',
            'version.nedf\t002\t01\trest\t01',
            'montage.nedf\t003\t01\trest\t01',
            'truncated.nedf\t004\t01\trest\t01',
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn('header is shorter than 10240 bytes', message)
        self.assertIn("unsupported NEDF version '2.0'", message)
        self.assertIn('montage contains 1 names', message)
        self.assertIn("'truncated.nedf' is truncated", message)

    def test_missing_manifest_is_an_error(self):
        with self.assertRaisesRegex(
            RecordingsValidationError,
            'manifest not found',
        ):
            validate_recordings(self.config)

    def test_header_only_manifest_is_an_error(self):
        self.write_manifest()

        with self.assertRaisesRegex(
            RecordingsValidationError,
            'contains no recordings',
        ):
            validate_recordings(self.config)

    def test_missing_column_and_value_are_reported_together(self):
        self.write_manifest(
            'original.vhdr\t001\t\trest',
            header='file\tsubject\tsession\ttask',
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn("missing required columns: ['run']", message)
        self.assertIn("missing value for 'session'", message)
        self.assertIn("missing value for 'run'", message)

    def test_duplicated_and_unexpected_columns_are_errors(self):
        self.write_manifest(
            'original.vhdr\t001\t01\trest\t01\tduplicate\textra',
            header=(
                'file\tsubject\tsession\ttask\trun\tsubject\tdescription'
            ),
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn("duplicated columns: ['subject']", message)
        self.assertIn("unexpected columns: ['description']", message)

    def test_paths_and_unsupported_extensions_are_rejected(self):
        self.write_manifest(
            'nested/original.vhdr\t001\t01\trest\t01',
            '..\\original.set\t002\t01\trest\t01',
            'C:\\data\\original.vhdr\t003\t01\trest\t01',
            '/data/original.set\t004\t01\trest\t01',
            'original.edf\t005\t01\trest\t01',
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertEqual(message.count('file must be a name without a path'), 4)
        self.assertIn('unsupported file extension', message)

    def test_uppercase_source_and_companion_extensions_are_accepted(self):
        header = self.sourcedata / 'ORIGINAL.VHDR'
        self.write_brainvision_header(
            header,
            'ORIGINAL.EEG',
            'ORIGINAL.VMRK',
        )
        (self.sourcedata / 'ORIGINAL.EEG').write_bytes(b'')
        (self.sourcedata / 'ORIGINAL.VMRK').write_text(
            'Brain Vision marker file',
            encoding='utf-8',
        )
        self.write_manifest('ORIGINAL.VHDR\t001\t01\trest\t01')

        recordings = validate_recordings(self.config)

        self.assertEqual(recordings[0]['file'], 'ORIGINAL.VHDR')

    def test_brainvision_portable_relative_references_are_accepted(self):
        companions = self.sourcedata / 'companions'
        companions.mkdir()
        (companions / 'original.eeg').write_bytes(b'')
        (companions / 'original.vmrk').write_text(
            'Brain Vision marker file',
            encoding='utf-8',
        )
        windows_header = self.sourcedata / 'windows.vhdr'
        unix_header = self.sourcedata / 'unix.vhdr'
        self.write_brainvision_header(
            windows_header,
            r'companions\original.eeg',
            r'companions\original.vmrk',
        )
        self.write_brainvision_header(
            unix_header,
            'companions/original.eeg',
            'companions/original.vmrk',
        )
        self.write_manifest(
            'windows.vhdr\t001\t01\trest\t01',
            'unix.vhdr\t002\t01\trest\t01',
        )

        recordings = validate_recordings(self.config)

        self.assertEqual(len(recordings), 2)

    def test_brainvision_references_cannot_escape_sourcedata(self):
        outside_data = self.sourcedata.parent / 'outside.eeg'
        outside_marker = self.sourcedata.parent / 'outside.vmrk'
        outside_data.write_bytes(b'')
        outside_marker.write_text('marker', encoding='utf-8')
        header = self.sourcedata / 'unsafe.vhdr'
        self.write_brainvision_header(
            header,
            '../outside.eeg',
            r'..\outside.vmrk',
        )
        self.write_manifest('unsafe.vhdr\t001\t01\trest\t01')

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertEqual(message.count('must not escape sourcedata'), 2)

    def test_missing_file_and_brainvision_companions_are_reported(self):
        (self.sourcedata / 'incomplete.vhdr').write_text(
            '\n'.join((
                '[Common Infos]',
                'DataFile=incomplete.eeg',
                'MarkerFile=incomplete.vmrk',
            )),
            encoding='utf-8',
        )
        self.write_manifest(
            'missing.vhdr\t001\t01\trest\t01',
            'incomplete.vhdr\t002\t01\trest\t01',
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn("source file not found", message)
        self.assertIn("missing DataFile 'incomplete.eeg'", message)
        self.assertIn("missing MarkerFile 'incomplete.vmrk'", message)

    def test_repeated_source_and_bids_destination_are_both_errors(self):
        self.write_brainvision('first')
        self.write_brainvision('second')
        self.write_manifest(
            'first.vhdr\t001\t01\trest\t01',
            'first.vhdr\t002\t01\trest\t01',
            'second.vhdr\t001\t01\trest\t01',
        )

        with self.assertRaises(RecordingsValidationError) as context:
            validate_recordings(self.config)

        message = str(context.exception)
        self.assertIn("source file is repeated: 'first.vhdr'", message)
        self.assertIn('generate the same BIDS destination', message)
        self.assertIn("['first.vhdr', 'second.vhdr']", message)

    def test_different_runs_do_not_collide(self):
        self.write_brainvision('first')
        self.write_brainvision('second')
        self.write_manifest(
            'first.vhdr\t001\t01\trest\t01',
            'second.vhdr\t001\t01\trest\t02',
        )

        recordings = validate_recordings(self.config)

        self.assertEqual(
            [recording['bids_path'].run for recording in recordings],
            ['01', '02'],
        )

    def test_invalid_bids_identifier_is_not_silently_changed(self):
        self.write_brainvision('original')
        self.write_manifest('original.vhdr\t001-bad\t01\trest\t01')

        with self.assertRaisesRegex(
            RecordingsValidationError,
            'invalid BIDS identifiers',
        ):
            validate_recordings(self.config)

    def test_quickstart_launcher_delegates_to_public_runner(self):
        config = {'pipeline': {'standardize': True}}

        with (
            patch(
                'quickstart.run_sEEGnal.init.load_config',
                return_value=config,
            ) as load_config_mock,
            patch('quickstart.run_sEEGnal.sEEGnal.run_sEEGnal') as runner,
        ):
            quickstart_main()

        load_config_mock.assert_called_once_with()
        runner.assert_called_once_with(config)

    def test_quickstart_paths_do_not_depend_on_current_directory(self):
        original_directory = Path.cwd()
        try:
            os.chdir(self.root)
            config = load_config()
        finally:
            os.chdir(original_directory)

        quickstart_root = Path(__file__).resolve().parents[1] / 'quickstart'
        self.assertEqual(
            Path(config['path']['data_root']),
            quickstart_root / 'data',
        )
        self.assertEqual(
            Path(config['path']['recordings']),
            quickstart_root / 'data' / 'recordings.tsv',
        )

    def test_quickstart_config_loads_in_a_subprocess_outside_repository_root(self):
        repository_root = Path(__file__).resolve().parents[1]
        command = (
            'import sys; '
            f'sys.path.insert(0, {str(repository_root)!r}); '
            f'sys.path.insert(0, {str(repository_root / "src")!r}); '
            'from quickstart.init.init import load_config; '
            "print(load_config()['path']['data_root'])"
        )

        completed = subprocess.run(
            [sys.executable, '-B', '-c', command],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            Path(completed.stdout.strip()),
            repository_root / 'quickstart' / 'data',
        )
        self.assertNotIn('ModuleNotFoundError', completed.stderr)

    def test_header_only_input_does_not_generate_source_data(self):
        self.write_manifest()

        with self.assertRaisesRegex(
            RecordingsValidationError,
            'contains no recordings',
        ):
            quickstart_init(self.config)

        self.assertEqual(list(self.sourcedata.iterdir()), [])

    def test_build_bids_object_requires_valid_run(self):
        with self.assertRaisesRegex(ValueError, 'run is not an index'):
            build_BIDS_object(self.config, '001', '01', 'rest', 'first')


if __name__ == '__main__':
    unittest.main()
