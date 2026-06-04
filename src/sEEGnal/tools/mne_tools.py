"""
Created on Mon Feb 20 12:40:25 2023

@author: bru
"""

import re
import datetime

import mne
import numpy
import scipy.signal

import sEEGnal.tools.bss as bss
import sEEGnal.tools.tools as tools
import sEEGnal.tools.signal as signal
import sEEGnal.tools.spheres as spheres
import sEEGnal.tools.bids_tools as bids
from sEEGnal.io.read_source_files import read_source_files
from sEEGnal.io.read_bids_files import read_BIDS_files

# Lists the valid MNE objects.
mnevalid = (mne.io.BaseRaw, mne.BaseEpochs)

# Sets the verbosity level for MNE.
mne.set_log_level(verbose='ERROR')


# Function for two-pass filtering on MNE objects.
def filtfilt(mnedata, num=1, den=1, hilbert=False):
    """ Wrapper to apply two-pass filtering to MNE objects."""

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # For IIR filters uses SciPy (faster and more accurate).
    if numpy.array(den).size != 1:

        # Gets the data metadata.
        dshape = rawdata.shape
        nsample = dshape[-1]

        # Reshapes the data into a 2D array.
        rawdata = rawdata.reshape((-1, nsample))

        # Filters the data.
        rawdata = scipy.signal.filtfilt(num, den, rawdata)

        # Restores the original data shape.
        rawdata = rawdata.reshape(dshape)

    # For FIR filters use FFT (much faster, same accuracy).
    else:

        # Filters the data.
        rawdata = signal.filtfilt(rawdata, num=num, den=den, hilbert=hilbert)

    # Replaces the data and marks it as loaded.
    mnedata._data = rawdata
    mnedata.preload = True

    ## Creates a new MNE object with the filtered data.
    #mnedata   = mne.EpochsArray ( rawdata, data.info, events = data.events, verbose = False )

    ## Creates a new MNE object with the filtered data.
    #mnedata    = mne.io.RawArray ( rawdata, data.info, verbose = False )

    # Returns the MNE object.
    return mnedata


def decimate(mnedata, ratio=1):
    """Decimates an MNE object with no filtering."""
    """
    Based on MNE 1.7 functions:
    mne.BaseRaw.resample
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/io/base.py
    mne.Epochs.decimate
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/utils/mixin.py
    """

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Ratio is 1 does nothing.
    if ratio == 1:
        return mnedata

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # Decimates the raw data matrix in the last dimension.
    decdata = rawdata[..., ::ratio]

    # Replaces the data and marks it as loaded.
    mnedata._data = decdata
    mnedata.preload = True

    # Updates the sampling rate.
    with mnedata.info._unlock():
        mnedata.info['sfreq'] = mnedata.info['sfreq'] / ratio

    # Updates the mne.Raw information.
    if isinstance(mnedata, mne.io.BaseRaw):
        n_news = numpy.array(decdata.shape[1:])
        mnedata._cropped_samp = int(numpy.round(mnedata._cropped_samp * ratio))
        mnedata._first_samps = numpy.round(mnedata._first_samps * ratio).astype(int)
        mnedata._last_samps = numpy.array(mnedata._first_samps) + n_news - 1
        mnedata._raw_lengths[:1] = list(n_news)

    # Updates the mne.Epochs information.
    if isinstance(mnedata, mne.BaseEpochs):
        mnedata._decim = 1
        mnedata._set_times(mnedata._raw_times[::ratio])
        mnedata._update_first_last()

    # Returns the MNE object.
    return mnedata


def fixchan(mnedata, elec=None):
    """Wrapper to apply spherical splines channel reconstruction to MNE objects."""

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # If no bad channels does nothing.
    if len(mnedata.info['bads']) == 0:
        return mnedata

    # If no electrode definition provided, loads the 10-05 default montage.
    if elec is None:
        elec = mne.channels.make_standard_montage('standard_1005')
        elec = elec.get_positions()['ch_pos']

    # Generates the reduced montage for the data.
    elec = {ch: elec[ch] for ch in elec.keys() if ch in mnedata.ch_names}

    # Generates the reduced montage for the good channels.
    elec1 = {ch: elec[ch] for ch in elec.keys() if ch not in mnedata.info['bads']}

    # Generates the reduced montage for the bad channels.
    elec2 = {ch: elec[ch] for ch in elec.keys() if ch in mnedata.info['bads']}

    # If no bad channels returns the untouched data.
    if len(elec2) == 0:
        return mnedata

    # Gets the reconstruction matrix.
    wPot = spheres.spline_int(elec1, elec2)

    # Gets the data-to-data transformation matrix.
    d2d = numpy.eye(len(mnedata.ch_names))

    # Gets the indexes of the good and bad channels.
    hits1 = tools.find_matches(list(elec1.keys()), mnedata.ch_names)
    hits2 = tools.find_matches(list(elec2.keys()), mnedata.ch_names)

    # Zeroes the bad channels.
    d2d[hits2, hits2] = 0

    # Adds the reconstruction mapping for the bad channels.
    d2d[numpy.ix_(hits2, hits1)] = wPot

    # Gets the raw data matrix.
    #rawdata = mnedata.get_data ( copy = True )
    rawdata = mnedata.get_data()
    """
    # Rewrites the raw data as epochs * samples * channels.
    shape   = rawdata.shape
    tmpdata = rawdata.reshape ( ( -1, ) + shape [ -2: ] )
    tmpdata = rawdata.swapaxes ( -1, -2 )

    # Fixes all the epochs at once.
    fixdata = numpy.dot ( tmpdata, d2d.T )

    # Restores the original data shape.
    fixdata = fixdata.swapaxes ( -1, -2 )
    fixdata = fixdata.reshape ( shape )
    """

    # Rewrites the raw data as epochs * channels * samples.
    shape = rawdata.shape
    tmpdata = rawdata.reshape((-1, ) + shape[-2:])

    # Fixes the bad channels epoch-wise.
    fixdata = numpy.zeros(tmpdata.shape)
    for i in range(tmpdata.shape[0]):
        fixdata[i] = numpy.dot(d2d, tmpdata[i])

    # Restores the original data shape.
    fixdata = fixdata.reshape(shape)

    # Updates the MNE object.
    mnedata._data = fixdata

    # Returns the fixed MNE object.
    return mnedata


# Perform SOBI on MNE object
def sobi(mnedata, nlag=None, nsource=None):
    ''' Wrapper to estimating SOBI componentes from MNE objects.'''

    # Checks if the data is a valid MNE object.
    if not isinstance(mnedata, mnevalid):

        print('Unsupported data type.')
        return None

    # Creates a copy of the input data.
    mnedata = mnedata.copy()

    # Gets the channel labels.
    chname = mnedata.ch_names

    # Gets the raw data matrix.
    rawdata = mnedata.get_data()

    # Estimates the SOBI mixing matrix.
    mixing, unmixing = bss.sobi(rawdata, nlag=nlag, nsource=nsource)

    # Builds the MNE ICA object.
    mnesobi = build_bss(mixing, unmixing, chname, method='sobi')

    # Returns the MNE ICA object.
    return mnesobi


def build_raw(info, data, montage=None):

    # Lists the channels in the data.
    ch_label = info['channels']['label']

    # If no montage assumes the standard 10-05.
    if montage is None:
        montage = mne.channels.make_standard_montage('standard_1005')

    # Identifies the EEG, EOG, ECG, and EMG channels.
    ind_eeg = numpy.where(numpy.in1d(ch_label, montage.ch_names))
    ind_eog = numpy.where([re.search('EOG', label) != None for label in ch_label])
    ind_ecg = numpy.where([re.search('CLAV', label) != None for label in ch_label])
    ind_emg = numpy.where([re.search('EMG', label) != None for label in ch_label])

    # Marks all the channels as EEG.
    ch_types = numpy.array(['eeg'] * len(ch_label))

    # Sets the channel types.
    ch_types[ind_eeg] = 'eeg'
    ch_types[ind_eog] = 'eog'
    ch_types[ind_ecg] = 'ecg'
    ch_types[ind_emg] = 'emg'

    # Creates the MNE-Python information object.
    mneinfo = mne.create_info(
        ch_names=list(info['channels']['label']), sfreq=info['sample_rate'], ch_types=list(ch_types)
    )

    # Adds the montage, if provided.
    if montage is not None:
        mneinfo.set_montage(montage)

    # Creates the MNE-Python raw data object.
    mneraw = mne.io.RawArray(data.T, mneinfo, verbose=False)

    # Overwrites the default parameters.
    mneraw.set_meas_date(info['acquisition_time'])

    # Adds the calibration factor.
    mneraw._cals = numpy.ones(len(ch_label))

    # Marks the 'active' channels.
    mneraw._read_picks = [numpy.arange(len(ch_label))]

    # Gets the information about the impedances, if any.
    if 'impedances' in info:

        # Takes only the first measurement.
        if len(info['impedances']) > 0:
            impmeta = info['impedances'][0]
            impedances = impmeta['measurement']

            # Fills the extra information for MNE.
            for channel, value in impedances.items():

                impedances[channel] = {
                    'imp': value,
                    'imp_unit': impmeta['unit'],
                    'imp_meas_time': datetime.datetime.fromtimestamp(impmeta['time'])
                }

            # Adds the impedances to the MNE object.
            mneraw.impedances = impedances

    # Gets the annotations, if any.
    annotations = mne.Annotations(
        [annot['onset'] for annot in info['events']], [annot['duration'] for annot in info['events']],
        [annot['description'] for annot in info['events']]
    )

    # Adds the annotations to the MNE object.
    mneraw.set_annotations(annotations)

    # Returns the MNE Raw object.
    return mneraw


def build_bss(mixing, unmixing, chname, icname=None, method='bss'):

    # Gets the number of channels and components.
    nchannel = mixing.shape[0]
    nsource = mixing.shape[1]

    # Generates the labels for the components, if no provided.
    if icname is None:

        # Creates the labels.
        icname = ['IC%03d' % (index + 1) for index in range(nsource)]

    # If the matrices are not square adds some dummy components.
    if nchannel != nsource:

        # Creates the dummy components.
        ndummy = nchannel - nsource
        mdummy = numpy.zeros([nchannel, ndummy])

        # Creates the dummy labels.
        ldummy = ['DUM%03d' % (index + 1) for index in range(ndummy)]

        # Concatenates the matrix and the labels.
        mixing = numpy.append(mixing, mdummy, 1)
        unmixing = numpy.append(unmixing, mdummy.T, 0)
        icname = icname + ldummy

    # Creates a dummy MNE ICA object.
    mnebss = mne.preprocessing.ICA()

    # Fills the object with the SOBI data.
    mnebss.ch_names = chname
    mnebss._ica_names = icname
    mnebss.mixing_matrix_ = mixing
    mnebss.unmixing_matrix_ = unmixing

    # Fills some dummy metadata.
    mnebss.current_fit = 'raw'
    mnebss.method = method
    mnebss.n_components_ = mixing.shape[0]
    mnebss.n_iter_ = 0
    mnebss.n_samples_ = 0
    mnebss.pca_components_ = numpy.eye(nchannel)
    mnebss.pca_explained_variance_ = numpy.ones(nchannel)
    mnebss.pca_mean_ = numpy.zeros(nchannel)
    mnebss.pre_whitener_ = 0.01 * numpy.ones([nchannel, 1])

    # Returns the MNE ICA object.
    return mnebss


# Function to prepare MNE raw data
def prepare_eeg(
    config,
    BIDS,
    raw=False,
    preload=False,
    channels_to_include=['all'],
    channels_to_exclude=[],
    apply_sobi = False,
    resample_frequency=False,
    notch_filter=False,
    freq_limits=False,
    crop_seconds=False,
    metadata_badchannels=False,
    exclude_badchannels=False,
    interpolate_badchannels=False,
    set_annotations=False,
    rereference=False,
    epoch_definition=False
):

    ##################################################################
    # Load EEG and montage
    ##################################################################

    if not raw:
        raw, bids_event_id = read_BIDS_files(BIDS, preload=preload)
        raw._event_id = bids_event_id
    else:
        bids_event_id = getattr(raw, "_event_id", None)

    if epoch_definition and epoch_definition.get("mode", "fixed") == "events":
        if bids_event_id is None:

            if len(raw.annotations) == 0:
                raise ValueError(
                    "Event-based epoching was requested, but no events.tsv or "
                    "annotations were found for this recording."
                )

            _, bids_event_id = mne.events_from_annotations(raw, verbose=False)
            bids_event_id = {str(k): int(v) for k, v in bids_event_id.items()}

    # Set montage
    raw.set_montage('standard_1005', on_missing='ignore')

    # Include and exclude channels explicitly
    raw.pick(channels_to_include)
    raw.drop_channels(channels_to_exclude, on_missing='ignore')

    ##################################################################
    # Include / exclude components
    ##################################################################

    if apply_sobi:

        # Load the SOBI
        # Read the ICA information
        sobi = bids.read_sobi(config,BIDS, apply_sobi['desc'])

        # IClabel recommend to filter between 1 and 100
        raw.filter(1, 100)

        # Select the components and remove
        IClabel_componets = ['brain', 'muscle', 'eog', 'ecg', 'line_noise',
                             'ch_noise', 'other']
        include = set(apply_sobi['components_to_include'] or IClabel_componets)
        exclude = set(apply_sobi['components_to_exclude'] or [])
        final_inclusion_list = list(include - exclude)

        # If any to apply
        if len(IClabel_componets) != len(final_inclusion_list):

            final_inclusion_index = [sobi.labels_[current_label] for
                                     current_label in final_inclusion_list]
            final_inclusion_index = sum(final_inclusion_index, [])

            # Apply
            sobi.apply(raw, include=final_inclusion_index)

    ##################################################################
    # Downsample
    ##################################################################

    if resample_frequency:
        raw.resample(resample_frequency)

    ##################################################################
    # Filters
    ##################################################################

    # Remove 50 Hz noise (and harmonics)
    if notch_filter:

        # Check the maximum frequency I can use
        notch_freqs = [current_freq for current_freq in config['global']['notch_frequencies']
                       if current_freq < resample_frequency/2 ]

        # Apply the filter if any
        if len(notch_freqs) > 0:
            raw.notch_filter(notch_freqs)

    # Filter the data
    if freq_limits:
        if len(freq_limits) != 2:
            raise ValueError(
                "You need to define two frequency limits to filter")
            # print('You need to define two frequency limits to filter')
        else:

            # Check the maximum frequency I can use
            if freq_limits[1] > raw.info['sfreq']/2:
                freq_limits[1] = raw.info['sfreq']/2 - 0.01

            raw.filter(freq_limits[0], freq_limits[1])

    ##################################################################
    # Remove bad channels
    ##################################################################

    # Add bad channels to the info
    if metadata_badchannels:

        chan = bids.read_badchannels(config,BIDS)
        badchannels = list(chan.loc[chan['status'] == 'bad']['name'])

        # To avoid errors, the badchannels has to be among the channels included in the recording
        badchannels = list(set(badchannels) & set(raw.info['ch_names']))

        # If all channels are badchannels, raise an Exception
        if (len(badchannels) == len(raw.info['ch_names'])):
            raise Exception("All channels are marked as bad")

        raw.info['bads'] = badchannels

    # Drop the badchannels
    if exclude_badchannels:
        raw = raw.pick(None, exclude='bads')

    # Interpolate the bad channels
    if interpolate_badchannels == True:
        raw.interpolate_bads()

    ##################################################################
    # Add the annotations if requested
    ##################################################################

    if set_annotations:

        annotations = bids.read_annotations(config, BIDS)

        if len(annotations):

            if annotations.orig_time != raw.annotations.orig_time:
                annotations = mne.Annotations(
                    onset=annotations.onset,
                    duration=annotations.duration,
                    description=annotations.description,
                    orig_time=raw.annotations.orig_time
                )

            raw.set_annotations(raw.annotations + annotations)

    ##################################################################
    # Crop (in seconds)
    ##################################################################

    if crop_seconds:

        # Save the original length
        raw.original_last_samp = raw.last_samp

        # Crop
        raw.crop(tmin=crop_seconds, tmax=raw.times[-1] - crop_seconds)

    ##################################################################
    # Set reference
    ##################################################################

    # Average
    if rereference == 'average':
        raw.set_eeg_reference('average', projection=True)

    # Median
    if rereference == 'median':
        # Get the data (shape: n_epochs × n_channels × n_times)
        data = raw.get_data()

        # Compute the median across channels
        median_ref = numpy.median(data, axis=1, keepdims=True)

        # Subtract the median reference from all channels
        data -= median_ref

        # Put the rereferenced data back into the epochs object
        raw._data = data

    ##################################################################
    # Epoch the data
    ##################################################################

    if epoch_definition:

            # If previously saved the atrribute, keep it
            if hasattr(raw, 'original_last_samp'):
                dummy = raw.original_last_samp

            raw = get_epochs(
                raw,
                preload,
                epoch_definition,
                bids_event_id=bids_event_id
            )

            # Save it again
            if 'dummy' in locals():
                raw.original_last_samp = dummy


    return raw


# Function to segment an MNE raw data object
def get_epochs(raw, preload, epoch_definition, bids_event_id=None):
    """
    Segment an MNE Raw object into epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The MNE Raw object.
    preload : bool
        Whether to preload the epochs data.
    epoch_definition : dict
        Dictionary defining the epoching strategy.
    bids_event_id : dict | None
        Event mapping recovered from BIDS. Used only when mode == 'events'.

    Returns
    -------
    epochs : mne.Epochs
        The epoched data.
    """

    ##################################################################
    # General parameters
    ##################################################################

    mode = epoch_definition.get('mode', 'fixed')

    reject_by_annotation = bool(
        epoch_definition.get('reject_by_annotation', False)
    )

    ##################################################################
    # Fixed-length epochs
    ##################################################################

    if mode == 'fixed':

        length = epoch_definition['length']
        overlap = epoch_definition.get('overlap', 0)
        padding = epoch_definition.get('padding', 0)

        # Get the sample limits
        last_samp = int(numpy.fix(raw.times[-1] * raw.info['sfreq']))
        first_samp = int(numpy.fix(raw.times[0] * raw.info['sfreq']))
        step = int(numpy.fix(raw.info['sfreq'] * (length - overlap)))

        # Create equally spaced events
        onsets = numpy.arange(first_samp, last_samp, step)
        events = numpy.zeros((len(onsets), 3), dtype=int)
        events[:, 0] = onsets
        events[:, 2] = 1

        # Build epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'fixed': 1},
            preload=preload,
            tmin=-padding,
            tmax=length + padding,
            baseline=None,
            reject_by_annotation=reject_by_annotation,
            verbose=False
        )

        return epochs

    ##################################################################
    # Event-based epochs
    ##################################################################

    elif mode == 'events':

        event_source = epoch_definition.get('event_source', 'annotations')
        event_code = epoch_definition['event_code']
        tmin = epoch_definition['tmin']
        tmax = epoch_definition['tmax']
        baseline = epoch_definition.get('baseline', None)

        # Convert JSON-style baseline lists into tuples for MNE
        if baseline is not None and isinstance(baseline, list):
            baseline = tuple(baseline)

        ##############################################################
        # Events from annotations
        ##############################################################

        if event_source == 'annotations':

            # Use the BIDS event mapping if available
            if bids_event_id is not None:
                found_event_id = {
                    str(key): int(value) for key, value in bids_event_id.items()
                }
            else:
                # Fallback: reconstruct from current annotations
                _, found_event_id = mne.events_from_annotations(raw, verbose=False)
                found_event_id = {
                    str(key): int(value) for key, value in found_event_id.items()
                }

            # Keep only event labels that are actually present in the raw
            available_descriptions = set(map(str, raw.annotations.description))

            valid_event_id = {
                key: value
                for key, value in found_event_id.items()
                if key in available_descriptions
            }

            if len(valid_event_id) == 0:
                raise ValueError(
                    'No valid event annotations were found in raw.annotations. '
                    f'Available annotations: {sorted(available_descriptions)}'
                )

            # Create events using only valid annotations
            events, _ = mne.events_from_annotations(
                raw,
                event_id=valid_event_id,
                verbose=False
            )

            # Select only the requested trigger code
            selected_event_id = {
                key: value
                for key, value in valid_event_id.items()
                if value == event_code
            }

            if len(selected_event_id) == 0:
                raise ValueError(
                    f'No annotation found for event_code {event_code}. '
                    f'Available event_id: {valid_event_id}'
                )

        ##############################################################
        # Events from stim channel
        ##############################################################

        elif event_source == 'stim_channel':

            events = mne.find_events(raw, verbose=False)
            selected_event_id = {str(event_code): int(event_code)}

        else:
            raise ValueError(
                "event_source must be 'annotations' or 'stim_channel'"
            )

        # Build epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=selected_event_id,
            preload=preload,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject_by_annotation=reject_by_annotation,
            verbose=False
        )

        return epochs

    ##################################################################
    # Unknown mode
    ##################################################################

    else:
        raise ValueError("epoch_definition['mode'] must be 'fixed' or 'events'")
