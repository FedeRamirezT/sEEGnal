"""Run the sEEGnal quickstart.

Place BrainVision recordings in ``quickstart/data/sourcedata/eeg``, describe
them in ``quickstart/data/recordings.tsv``, and then run:

``python -m quickstart.run_sEEGnal``
"""

from pathlib import Path
import sys


if __package__ in (None, ''):
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

import quickstart.init.init as init
import sEEGnal
from sEEGnal.io.recordings import RecordingsValidationError
from sEEGnal.pipeline import PipelineConfigurationError


def main():
    """Load the quickstart configuration and run sEEGnal."""

    try:
        config = init.load_config()
        sEEGnal.run_sEEGnal(config)
    except (PipelineConfigurationError, RecordingsValidationError) as error:
        raise SystemExit(f'Validation failed:\n{error}') from None


if __name__ == '__main__':
    main()
