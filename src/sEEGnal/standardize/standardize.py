# -*- coding: utf-8 -*-
"""

Convert original EEG recordings into a standardized BIDS structure.

Functions
---------
standardize
    Select the standardization workflow for the recording datatype.
standardize_eeg_file
    Read an original EEG file and write it into the target BIDS structure.

Federico Ramírez-Toraño
01/05/2023

"""

from pathlib import Path
import re
import traceback
from datetime import datetime as dt, timezone

import mne
import numpy
import mne_bids

from sEEGnal.io.read_source_files import read_source_files


mne.utils.set_log_level(verbose='ERROR')


def standardize(config, current_file, BIDS):
    """Select the standardization workflow for the recording datatype."""

    config['subsystem'] = 'preprocess'

    if BIDS.datatype == 'eeg':
        try:
            bids_basename = standardize_eeg_file(config, current_file, BIDS)
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {
                'result': 'ok',
                'file': current_file,
                'bids_basename': bids_basename,
                'date': formatted_now,
            }
        except Exception as error:
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {
                'result': 'error',
                'file': current_file,
                'details': (
                    f"Exception: {str(error)}, {traceback.format_exc()}"
                ),
                'date': formatted_now,
            }
    else:
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'file': current_file,
            'details': 'Not accepted type of file to process',
            'date': formatted_now,
        }

    return results


def standardize_eeg_file(config, current_file, BIDS):
    """Read an original EEG file and write it into the BIDS structure."""

    source_filepath = Path(config['path']['sourcedata']) / current_file
    mnedata = read_source_files(source_filepath)

    # Keep the original event_ids.
    descriptions = sorted(set(mnedata.annotations.description))
    event_id = {}
    used_codes = set()

    for desc in descriptions:
        match = re.search(r'^Stimulus/s(\d+)$', desc)
        if match:
            code = int(match.group(1))
            event_id[desc] = code
            used_codes.add(code)

    # Add non-experimental annotations using reserved codes.
    next_code = 10000
    for desc in descriptions:
        if desc not in event_id:
            while next_code in used_codes:
                next_code += 1
            event_id[desc] = next_code
            used_codes.add(next_code)
            next_code += 1

    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    mnedata.pick(channels_to_include)
    mnedata.drop_channels(channels_to_exclude, on_missing='ignore')

    mnedata.info['line_freq'] = config['global']['line_freq']
    mnedata.info['subject_info'] = dict()
    mnedata.info['subject_info']['his_id'] = BIDS.subject

    mne_bids.write_raw_bids(
        mnedata,
        BIDS,
        event_id=event_id,
        allow_preload=True,
        format=config['preprocess']['standardization']['format'],
        overwrite=bool(
            config['preprocess']['standardization']['overwrite']
        ),
    )

    ch_tsv = BIDS.copy().update(suffix='channels', extension='.tsv')
    ch_data = mne_bids.tsv_handler._from_tsv(ch_tsv)

    # Set the anti-alias filter to 26% of the sampling rate.
    aaf = 0.26 * mnedata.info['sfreq']
    ch_data['high_cutoff'] = [aaf for _ in ch_data['high_cutoff']]

    ch_imp = mne_bids.dig._get_impedances(mnedata, ch_data['name'])
    if ch_imp[0] != 'n/a':
        ch_imp = [numpy.round(imp, 2) for imp in ch_imp]
    ch_data['impedance'] = ch_imp

    mne_bids.tsv_handler._to_tsv(ch_data, ch_tsv)

    return BIDS.basename
