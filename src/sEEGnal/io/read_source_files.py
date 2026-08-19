"""

Read an original EEG recording according to its file format.

Functions
---------
read_source_files
    Select the appropriate MNE reader from the source file extension.

Federico Ramírez-Toraño
21/11/2024

"""

# Imports
from pathlib import Path

import mne


def read_source_files(source_filepath, preload=True):
    """
    Select the appropriate MNE reader from the source file extension.

    Parameters
    ----------
    source_filepath : str | pathlib.Path
        Path to the original EEG file.
    preload : bool
        Whether to preload signal data into memory.

    Returns
    -------
    data : mne.io.BaseRaw
        Continuous BrainVision EEG loaded with MNE.
    """

    source_filepath = Path(source_filepath)

    # Detect the EEG file type based on the extension
    extension = source_filepath.suffix.casefold()

    # BrainVision
    if extension == '.vhdr':

        # Read the file
        mnedata = mne.io.read_raw_brainvision(
            source_filepath,
            preload=preload
        )

    else:
        raise ValueError(
            f'Unsupported EEG source format {extension!r}; expected '
            '.vhdr.'
        )

    return mnedata
