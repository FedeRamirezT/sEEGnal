"""Regression tests for bad-channel storage and QC colors."""

import tempfile
import unittest
from pathlib import Path

import mne_bids

from sEEGnal.tools.bids_tools import (
    build_derivatives_path,
    build_standardize_path,
    write_badchannels,
)
from sEEGnal.tools.qc_tools import _badchannel_color


class TestBadChannelStorage(unittest.TestCase):
    """Ensure repeated preprocessing runs replace stale classifications."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = {'subsystem': 'preprocess'}
        self.bids_path = mne_bids.BIDSPath(
            subject='001',
            session='0',
            task='rest',
            datatype='eeg',
            suffix='eeg',
            extension='.vhdr',
            root=self.root,
        )
        standardize_path = Path(
            build_standardize_path(self.bids_path, 'channels.tsv')
        )
        standardize_path.parent.mkdir(parents=True)
        standardize_path.write_text(
            'name\tstatus\tstatus_description\n'
            'Cz\tgood\tn/a\n'
            'Pz\tgood\tn/a\n',
            encoding='utf-8',
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_repeated_run_does_not_append_previous_description(self):
        description = 'bad_impossible_amplitude_badchannels'

        write_badchannels(
            self.config, self.bids_path, ['Cz'], [description]
        )
        write_badchannels(
            self.config, self.bids_path, ['Cz'], [description]
        )

        derivative_path = build_derivatives_path(
            self.bids_path, 'preprocess', 'channels.tsv'
        )
        contents = Path(derivative_path).read_text(encoding='utf-8-sig')
        self.assertEqual(contents.count(description), 1)

    def test_new_run_removes_stale_bad_channel(self):
        write_badchannels(
            self.config, self.bids_path, ['Cz'], ['bad_component']
        )
        write_badchannels(
            self.config, self.bids_path, ['Pz'], ['bad_gel_bridge']
        )

        derivative_path = build_derivatives_path(
            self.bids_path, 'preprocess', 'channels.tsv'
        )
        contents = Path(derivative_path).read_text(encoding='utf-8-sig')
        self.assertIn('Cz\tgood\tn/a', contents)
        self.assertIn('Pz\tbad\tbad_gel_bridge', contents)


class TestBadChannelColors(unittest.TestCase):
    """Exercise descriptions produced by old and current channel tables."""

    def setUp(self):
        self.colors = {
            'bad_impossible_amplitude_badchannels': 'orange',
            'bad_component': 'red',
            'bad_gel_bridge': 'gold',
        }

    def test_repeated_description_keeps_its_category_color(self):
        repeated = ','.join(
            ['bad_impossible_amplitude_badchannels'] * 3
        )
        self.assertEqual(
            _badchannel_color(repeated, self.colors),
            'orange'
        )

    def test_multiple_causes_are_not_shown_as_bad_component(self):
        self.assertEqual(
            _badchannel_color(
                'bad_impossible_amplitude_badchannels,bad_component',
                self.colors,
            ),
            'black'
        )

    def test_unknown_cause_is_not_shown_as_bad_component(self):
        self.assertEqual(
            _badchannel_color('bad_future_method', self.colors),
            'gray'
        )


if __name__ == '__main__':
    unittest.main()
