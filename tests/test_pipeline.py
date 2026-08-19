"""Tests for the public sEEGnal pipeline runner."""

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from sEEGnal.pipeline import PipelineConfigurationError, run_sEEGnal


class TestPipelineConfiguration(unittest.TestCase):
    """Reject invalid stage selections before input or processing begins."""

    def test_invalid_pipeline_does_not_validate_or_process_recordings(self):
        invalid_pipelines = (
            {'artifact_detction': True},
            {'artifact_detection': 1},
            {},
            {'standardize': False, 'artifact_detection': False},
        )

        for pipeline in invalid_pipelines:
            with self.subTest(pipeline=pipeline):
                config = {'pipeline': pipeline}
                with patch(
                    'sEEGnal.pipeline.validate_recordings'
                ) as validate_recordings:
                    with self.assertRaises(PipelineConfigurationError) as error:
                        run_sEEGnal(config)

                validate_recordings.assert_not_called()
                self.assertIn('Accepted stages:', str(error.exception))

    def test_unknown_stage_error_lists_invalid_and_accepted_names(self):
        config = {'pipeline': {'standardise': True}}

        with self.assertRaises(PipelineConfigurationError) as context:
            run_sEEGnal(config)

        message = str(context.exception)
        self.assertIn("['standardise']", message)
        self.assertIn('standardize', message)
        self.assertIn('badchannel_detection', message)
        self.assertIn('artifact_detection', message)
        self.assertIn('source_reconstruction', message)
        self.assertIn('feature_extraction', message)


class TestPipelineExecution(unittest.TestCase):
    """Coordinate stages while isolating failures to one recording."""

    @staticmethod
    def _recording(subject):
        return {
            'file': f'{subject}.vhdr',
            'subject': subject,
            'session': '01',
            'task': 'rest',
            'run': '01',
            'bids_path': SimpleNamespace(basename=f'sub-{subject}_eeg'),
        }

    def test_failed_stage_stops_subject_and_next_subject_continues(self):
        recordings = [self._recording('001'), self._recording('002')]
        config = {
            'pipeline': {
                'standardize': True,
                'artifact_detection': True,
            },
            'global': {'verbose': 'brief'},
        }
        standardize_results = (
            {'result': 'error', 'details': 'first failed'},
            {'result': 'ok'},
        )

        with (
            patch(
                'sEEGnal.pipeline.validate_recordings',
                return_value=recordings,
            ),
            patch(
                'sEEGnal.pipeline.standardize.standardize',
                side_effect=standardize_results,
            ) as standardize,
            patch(
                'sEEGnal.pipeline.artifact_detection.artifact_detection',
                return_value={'result': 'ok'},
            ) as artifacts,
        ):
            run_sEEGnal(config)

        self.assertEqual(standardize.call_count, 2)
        artifacts.assert_called_once_with(config, recordings[1]['bids_path'])
        self.assertEqual(
            standardize.call_args_list,
            [
                call(config, '001.vhdr', recordings[0]['bids_path']),
                call(config, '002.vhdr', recordings[1]['bids_path']),
            ],
        )


if __name__ == '__main__':
    unittest.main()
