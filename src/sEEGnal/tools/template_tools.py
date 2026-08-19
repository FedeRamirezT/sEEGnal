"""

Prepare the fsaverage template used for source reconstruction.

Functions
---------
get_subjects_dir
    Ensure fsaverage is available locally.

Federico Ramírez-Toraño
17/02/2026

"""

import logging
from pathlib import Path
import warnings

import mne
import pooch


def _fetch_fsaverage() -> Path:
    """Fetch fsaverage quietly and report its first successful download."""

    configured_subjects_dir = mne.get_config("SUBJECTS_DIR")
    if configured_subjects_dir:
        subjects_dir = Path(configured_subjects_dir).expanduser()
    else:
        mne_data_dir = mne.get_config("MNE_DATA")
        if mne_data_dir:
            subjects_dir = Path(mne_data_dir).expanduser()
        else:
            subjects_dir = Path.home() / "mne_data"
        subjects_dir /= "MNE-fsaverage-data"

    subjects_dir = subjects_dir.absolute()
    fsaverage_was_available = (subjects_dir / "fsaverage").is_dir()

    # An existing SUBJECTS_DIR configuration can point to MNE's default cache
    # before its first download. Create it so MNE does not warn about the
    # expected first-run state.
    subjects_dir.mkdir(parents=True, exist_ok=True)

    pooch_logger = pooch.get_logger()
    previous_pooch_level = pooch_logger.level
    pooch_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "SUBJECTS_DIR in your MNE-Python configuration or "
                    "environment does not exist.*"
                ),
                category=RuntimeWarning,
            )
            fsaverage_dir = Path(
                mne.datasets.fetch_fsaverage(
                    subjects_dir=subjects_dir,
                    verbose=False,
                )
            )
    finally:
        pooch_logger.setLevel(previous_pooch_level)

    if not fsaverage_was_available:
        print(f'fsaverage downloaded to "{fsaverage_dir}".', end='')

    return fsaverage_dir


def get_subjects_dir(subject: str) -> tuple[Path, str]:
    """
    Ensure the fsaverage template is available locally.

    Parameters
    ----------
    subject : str
        Template subject name.

    Returns
    -------
    subjects_dir : pathlib.Path
        FreeSurfer subjects directory.
    subject : str
        Normalized template subject name.
    """


    if subject != "fsaverage":
        raise ValueError(
            f"Unsupported template subject: {subject}. "
            "Currently only 'fsaverage' is supported."
        )

    # Fetch or validate fsaverage from MNE.
    fsaverage_dir = _fetch_fsaverage()
    subjects_dir = fsaverage_dir.parent
    subject = "fsaverage"

    return subjects_dir, subject
