"""Run the configured sEEGnal processing stages."""

from sEEGnal.feature_extraction import feature_extraction
from sEEGnal.io.recordings import validate_recordings
from sEEGnal.preprocess import artifact_detection, badchannel_detection
from sEEGnal.sources_reconstruction import forward, inverse
from sEEGnal.standardize import standardize


PIPELINE_STAGES = (
    'standardize',
    'badchannel_detection',
    'artifact_detection',
    'source_reconstruction',
    'feature_extraction',
)


class PipelineConfigurationError(ValueError):
    """Report an invalid pipeline-stage selection."""


def _accepted_stages_message():
    """Return the ordered list of accepted public stage names."""

    return f"Accepted stages: {', '.join(PIPELINE_STAGES)}."


def _validate_pipeline(config):
    """Validate and normalize the public pipeline-stage selection."""

    pipeline = config.get('pipeline') if isinstance(config, dict) else None
    if not isinstance(pipeline, dict):
        raise PipelineConfigurationError(
            "The 'pipeline' configuration must be an object. "
            f'{_accepted_stages_message()}'
        )

    invalid_stages = sorted(set(pipeline) - set(PIPELINE_STAGES))
    if invalid_stages:
        raise PipelineConfigurationError(
            f'Invalid pipeline stages: {invalid_stages}. '
            f'{_accepted_stages_message()}'
        )

    invalid_values = [
        stage for stage, enabled in pipeline.items()
        if not isinstance(enabled, bool)
    ]
    if invalid_values:
        raise PipelineConfigurationError(
            'Pipeline stages must be set to true or false; invalid values '
            f'for: {sorted(invalid_values)}. {_accepted_stages_message()}'
        )

    enabled_stages = {
        stage: pipeline.get(stage, False) for stage in PIPELINE_STAGES
    }
    if not any(enabled_stages.values()):
        raise PipelineConfigurationError(
            'No pipeline stages are enabled. '
            f'{_accepted_stages_message()}'
        )

    return enabled_stages


def _print_error_details(config, results):
    """Print existing stage details when full verbosity is requested."""

    if config.get('global', {}).get('verbose') == 'full':
        print(results['details'])


def _run_standardize(config, recording):
    """Run and report BIDS standardization for one recording."""

    print('   Standardize', end='. ')
    results = standardize.standardize(
        config,
        recording['file'],
        recording['bids_path'],
    )
    print(' Result ' + results['result'])
    if results['result'] == 'error':
        _print_error_details(config, results)
    return results['result'] == 'error'


def _run_badchannel_detection(config, recording):
    """Run and report bad-channel detection for one recording."""

    print('   Badchannel detection', end='. ')
    results = badchannel_detection.badchannel_detection(
        config,
        recording['bids_path'],
    )
    print(' Result ' + results['result'])
    if results['result'] == 'error':
        _print_error_details(config, results)
    return results['result'] == 'error'


def _run_artifact_detection(config, recording):
    """Run and report artifact detection for one recording."""

    print('   Artifact Detection', end='. ')
    results = artifact_detection.artifact_detection(
        config,
        recording['bids_path'],
    )
    print(' Result ' + results['result'])
    if results['result'] == 'error':
        _print_error_details(config, results)
    return results['result'] == 'error'


def _run_source_reconstruction(config, recording):
    """Run and report forward and inverse modelling for one recording."""

    BIDS = recording['bids_path']

    print('   Forward Model', end='. ')
    results = forward.make_forward_model(config, BIDS)
    print(' Result ' + results['result'])
    if results['result'] == 'error':
        _print_error_details(config, results)
        return True

    print('   Inverse Solution', end='. ')
    results = inverse.estimate_inverse_solution(config, BIDS)
    print(' Result ' + results['result'])
    if results['result'] == 'error':
        _print_error_details(config, results)
        return True

    return False


def _run_feature_extraction(config, recording):
    """Run and report configured feature extraction for one recording."""

    print('   Feature extraction', end='. ')
    results = feature_extraction.feature_extraction(
        config,
        recording['bids_path'],
    )
    failed_results = [
        result for result in results if result['result'] == 'error'
    ]
    if not failed_results:
        print(' Result ok')
        return False

    failed_features = [
        result['feature'] for result in failed_results
    ]
    print(f' Result error in: {failed_features}')
    if config.get('global', {}).get('verbose') == 'full':
        for failed_result in failed_results:
            print(failed_result['details'])
    return True


_STAGE_RUNNERS = {
    'standardize': _run_standardize,
    'badchannel_detection': _run_badchannel_detection,
    'artifact_detection': _run_artifact_detection,
    'source_reconstruction': _run_source_reconstruction,
    'feature_extraction': _run_feature_extraction,
}


def run_sEEGnal(config):
    """Run the selected sEEGnal stages for every validated recording.

    Parameters
    ----------
    config : dict
        Complete sEEGnal configuration. ``config['pipeline']`` may contain
        any subset of :data:`PIPELINE_STAGES`; omitted stages are disabled.

    Raises
    ------
    PipelineConfigurationError
        If the public pipeline-stage selection is invalid or empty.
    sEEGnal.io.recordings.RecordingsValidationError
        If any input recording or manifest row is invalid. Validation is
        completed before a processing stage is called.

    Notes
    -----
    A failed stage stops processing for its recording. Other validated
    recordings continue normally.
    """

    enabled_stages = _validate_pipeline(config)
    recordings = validate_recordings(config)

    print(f'Found {len(recordings)} valid recording(s).')

    for recording in recordings:
        print(
            f"Working with sub {recording['subject']} "
            f"ses {recording['session']} task {recording['task']} "
            f"run {recording['run']}"
        )

        for stage in PIPELINE_STAGES:
            if not enabled_stages[stage]:
                continue
            if _STAGE_RUNNERS[stage](config, recording):
                break

        print()
        print()
        print()
