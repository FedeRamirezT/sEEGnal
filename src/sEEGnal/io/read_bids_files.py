"""
Read BIDS EEG and events

Created on 15/04/2026

Federico Ramirez-Toraño
"""

import mne_bids

def read_BIDS_files(BIDS, preload=True):

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