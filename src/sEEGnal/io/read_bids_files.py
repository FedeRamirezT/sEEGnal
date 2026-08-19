"""

Read BIDS EEG recordings and their associated events.

Functions
---------
read_BIDS_files
    Load a BIDS EEG recording and return its event identifiers when available.

Federico Ramírez-Toraño
15/04/2026

"""

import mne_bids

def read_BIDS_files(BIDS, preload=True):
    """
    Load a BIDS EEG recording and return its event identifiers when available.

    Parameters
    ----------
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    preload : bool
        Whether to preload signal data into memory.

    Returns
    -------
    raw : mne.io.BaseRaw
        Loaded BIDS EEG recording.
    event_id : dict | None
        Event mapping when an events file is available.
    """

    events_path = BIDS.copy().update(
        suffix="events",
        extension=".tsv"
    ).fpath

    has_events = events_path.exists()

    if has_events:

        raw, event_id = mne_bids.read_raw_bids(
            BIDS,
            return_event_dict=True,
            verbose='ERROR'
        )

    else:

        raw = mne_bids.read_raw_bids(
            BIDS,
            verbose='ERROR'
        )

        event_id = None

    if preload:
        raw.load_data()

    return raw, event_id
