"""Tests for the artifact-annotation TSV storage contract."""

import json
import tempfile
import unittest
from pathlib import Path

import mne
import mne_bids
import numpy
import pandas

from sEEGnal.tools.bids_tools import read_annotations, write_annotations


class TestArtifactAnnotationsRoundTrip(unittest.TestCase):
    """Exercise the real writer and reader through files on disk."""

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

        basename = (
            'sub-023_ses-0_task-1EC_desc-artifacts_annotations'
        )
        output_directory = (
            self.root
            / 'derivatives'
            / 'sEEGnal'
            / 'preprocess'
            / 'sub-023'
            / 'ses-0'
            / 'eeg'
        )
        self.tsv_path = output_directory / f'{basename}.tsv'
        self.json_path = output_directory / f'{basename}.json'

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_real_tsv_round_trip_preserves_annotations(self):
        annotations = mne.Annotations(
            onset=[1.25, 8.5],
            duration=[0.5, 0.25],
            description=['bad_muscle', 'bad_EOG'],
        )

        created_files = write_annotations(
            self.config,
            self.bids_path,
            annotations,
        )

        self.assertEqual(
            {Path(path) for path in created_files},
            {self.tsv_path, self.json_path},
        )
        self.assertTrue(self.tsv_path.is_file())
        self.assertTrue(self.json_path.is_file())
        self.assertEqual(
            self.tsv_path.read_text(encoding='utf-8-sig').splitlines()[0],
            'onset\tduration\tlabel',
        )

        stored = pandas.read_table(
            self.tsv_path,
            delimiter='\t',
            encoding='utf-8-sig',
        )
        self.assertEqual(
            list(stored.columns),
            ['onset', 'duration', 'label'],
        )
        numpy.testing.assert_allclose(stored['onset'], [1.25, 8.5])
        numpy.testing.assert_allclose(stored['duration'], [0.5, 0.25])
        self.assertEqual(
            stored['label'].tolist(),
            ['bad_muscle', 'bad_EOG'],
        )

        loaded = read_annotations(self.config, self.bids_path)
        numpy.testing.assert_allclose(loaded.onset, annotations.onset)
        numpy.testing.assert_allclose(loaded.duration, annotations.duration)
        numpy.testing.assert_array_equal(
            loaded.description,
            annotations.description,
        )

        metadata = json.loads(self.json_path.read_text(encoding='utf-8'))
        self.assertEqual(metadata['onset']['Units'], 's')
        self.assertEqual(metadata['duration']['Units'], 's')

    def test_empty_annotations_round_trip_keeps_the_tsv_contract(self):
        write_annotations(
            self.config,
            self.bids_path,
            mne.Annotations([], [], []),
        )

        stored = pandas.read_table(
            self.tsv_path,
            delimiter='\t',
            encoding='utf-8-sig',
        )
        self.assertEqual(
            list(stored.columns),
            ['onset', 'duration', 'label'],
        )
        self.assertTrue(stored.empty)

        loaded = read_annotations(self.config, self.bids_path)
        self.assertEqual(len(loaded), 0)

    def test_second_write_replaces_the_complete_annotation_set(self):
        write_annotations(
            self.config,
            self.bids_path,
            mne.Annotations(
                onset=[1.25, 8.5],
                duration=[0.5, 0.25],
                description=['old_muscle', 'old_EOG'],
            ),
        )
        write_annotations(
            self.config,
            self.bids_path,
            mne.Annotations(
                onset=[3.75],
                duration=[0.125],
                description=['new_jump'],
            ),
        )

        stored = pandas.read_table(
            self.tsv_path,
            delimiter='\t',
            encoding='utf-8-sig',
        )
        self.assertEqual(len(stored), 1)
        numpy.testing.assert_allclose(stored['onset'], [3.75])
        numpy.testing.assert_allclose(stored['duration'], [0.125])
        self.assertEqual(stored['label'].tolist(), ['new_jump'])

        loaded = read_annotations(self.config, self.bids_path)
        numpy.testing.assert_allclose(loaded.onset, [3.75])
        numpy.testing.assert_allclose(loaded.duration, [0.125])
        numpy.testing.assert_array_equal(
            loaded.description,
            ['new_jump'],
        )

    def test_reader_reports_missing_required_columns(self):
        self.tsv_path.parent.mkdir(parents=True)
        pandas.DataFrame({
            'onset': [1.25],
            'duration': [0.5],
        }).to_csv(
            self.tsv_path,
            sep='\t',
            index=False,
            encoding='utf-8-sig',
        )

        with self.assertRaisesRegex(ValueError, r"\['label'\]"):
            read_annotations(self.config, self.bids_path)


if __name__ == '__main__':
    unittest.main()
