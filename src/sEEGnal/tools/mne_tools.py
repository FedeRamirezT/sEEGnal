"""

Prepare, transform and analyze EEG data represented by MNE objects.

Functions
---------
get_reference_info
    Return an independent copy of effective-reference metadata.
_ensure_reference_state
    Initialize or preserve sEEGnal's effective-reference metadata.
_copy_reference_state
    Copy effective-reference metadata between MNE objects.
_check_no_pending_projectors
    Reject MNE objects containing unapplied projectors.
_apply_average_reference
    Apply MNE's average-reference projector and record its effective state.
_check_ica_reference
    Require ICA and signal data to use the same effective reference.
_check_ica_channels
    Require ICA and signal data to use the same ordered EEG sensor space.
_check_inverse_reference
    Require an inverse solution and signal data to use the same reference.
_remember_original_sample_bounds
    Preserve the inclusive sample bounds of an uncropped Raw recording.
_annotation_key
    Build a stable identity key for an MNE annotation.
_clip_annotation_to_time_window
    Intersect an annotation with a half-open temporal window.
_set_saved_annotations
    Attach original-time annotations to a possibly cropped Raw object.
get_unannotated_raw
    Materialize a Raw view without samples covered by BAD annotations.
fit_ica
    Fit a stable Extended Infomax ICA decomposition to Raw or Epochs data.
build_raw
    Build an MNE RawArray and montage from channel metadata and signal data.
prepare_eeg
    Apply the requested loading, cleaning, filtering and epoching operations.
apply_ica
    Apply selected ICA component removal to compatible continuous EEG data.
get_epochs
    Segment continuous data into fixed-length or event-based epochs.

Federico Ramírez-Toraño
20/02/2023

"""

import copy
import datetime
import numbers
import re

import mne
import numpy

import sEEGnal.tools.bids_tools as bids_tool
from sEEGnal.io.read_bids_files import read_BIDS_files

# Lists the valid MNE objects.
mnevalid = (mne.io.BaseRaw, mne.BaseEpochs)

# Sets the verbosity level for MNE.
mne.set_log_level(verbose='ERROR')


def _ensure_reference_state(mnedata):
    """
    Initialize sEEGnal's effective-reference metadata when absent.

    The metadata describe the reference that is already effective in the data,
    rather than a reference operation that has merely been requested. Existing
    metadata are therefore preserved across repeated calls to
    :func:`prepare_eeg`.

    Parameters
    ----------
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        MNE object whose reference state will be initialized.

    Returns
    -------
    reference_state : dict
        Existing or newly initialized reference metadata.
    """

    # Absence of metadata means that sEEGnal has not changed the acquisition
    # reference; it does not mean that the data are physically unreferenced.
    if not hasattr(mnedata, '_seegnal_reference'):
        mnedata._seegnal_reference = {
            'schema_version': 1,
            'method': 'as_recorded',
            'implementation': 'acquisition',
            'channels': None,
            'status': 'effective',
        }

    return mnedata._seegnal_reference


def get_reference_info(mnedata):
    """
    Return an independent copy of the effective EEG reference metadata.

    This is the public accessor for sEEGnal's private
    ``_seegnal_reference`` attribute. If a Raw or Epochs object has not yet
    passed through :func:`prepare_eeg`, its state is initialized as
    ``as_recorded``. Mutating the returned dictionary does not modify the MNE
    object.

    Parameters
    ----------
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        Continuous or epoched EEG data whose effective reference is queried.

    Returns
    -------
    reference_info : dict
        Independent copy of the reference method, implementation, channels,
        status and metadata schema version.

    Raises
    ------
    TypeError
        If ``mnedata`` is not an MNE Raw or Epochs object.
    """

    if not isinstance(mnedata, mnevalid):
        raise TypeError(
            'mnedata must be an MNE Raw or Epochs object, '
            f'not {type(mnedata).__name__}.'
        )

    # Never expose the live private dictionary: callers should be able to
    # inspect or reshape the result without corrupting pipeline provenance.
    return copy.deepcopy(_ensure_reference_state(mnedata))


def _copy_reference_state(source, destination):
    """
    Copy sEEGnal's effective-reference metadata between MNE objects.

    The source state is initialized as ``as_recorded`` when absent. A deep
    copy is attached to the destination so later metadata changes on either
    object remain independent.

    Parameters
    ----------
    source : mne.io.BaseRaw | mne.BaseEpochs
        MNE object providing the effective-reference metadata.
    destination : mne.io.BaseRaw | mne.BaseEpochs | mne.preprocessing.ICA | dict
        MNE object receiving an independent copy of the metadata.

    Returns
    -------
    reference_state : dict
        Reference metadata attached to ``destination``.
    """

    reference_state = _ensure_reference_state(source)

    # ``channels`` is a mutable list. A deep copy prevents later edits to the
    # provenance of one derived object from changing another object silently.
    copied_state = copy.deepcopy(reference_state)

    # Beamformer objects are dictionary-like, whereas Raw, Epochs and ICA keep
    # sEEGnal-specific provenance as a private Python attribute.
    if isinstance(destination, dict):
        destination['_seegnal_reference'] = copied_state
    else:
        destination._seegnal_reference = copied_state

    return copied_state


def _check_no_pending_projectors(mnedata):
    """
    Reject an MNE object containing unapplied projectors.

    MNE applies every inactive projector together when ``apply_proj`` is
    called. Requiring all pre-existing projectors to be active prevents a
    later sEEGnal reference operation from applying unrelated transformations
    implicitly.

    Parameters
    ----------
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        MNE object whose projectors will be inspected.

    Returns
    -------
    None
        The function returns when every stored projector is already active.

    Raises
    ------
    RuntimeError
        If one or more stored projectors have not been applied.
    """

    # Keep the descriptions so an error identifies exactly which projector
    # must be applied or removed by the caller.
    pending_projectors = [
        projector['desc']
        for projector in mnedata.info['projs']
        if not projector['active']
    ]
    if pending_projectors:
        descriptions = ', '.join(
            repr(description) for description in pending_projectors
        )
        raise RuntimeError(
            'prepare_eeg received unapplied projector(s): '
            f'{descriptions}. Apply or remove them before calling '
            'prepare_eeg.'
        )


def _apply_average_reference(raw):
    """
    Apply the average EEG reference and record its effective state.

    MNE's average-reference projector is retained as provenance but is applied
    immediately. Repeated calls are idempotent when sEEGnal metadata already
    identify the average reference as effective.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG data to rereference in place.

    Returns
    -------
    reference_state : dict
        Effective average-reference metadata attached to ``raw``.

    Raises
    ------
    RuntimeError
        If effective projection metadata lack an active projector, or MNE does
        not create an identifiable average-reference projector.
    """

    # An active projector may already exist after a previous prepare_eeg call.
    # Reuse it only while it still describes the current good EEG sensor
    # space. Picking or dropping channels after it was applied changes the
    # average that a new request is asking for.
    average_projector = next(
        (
            projector
            for projector in raw.info['projs']
            if projector['desc'] == 'Average EEG reference'
            and projector['active']
        ),
        None,
    )

    reference_state = _ensure_reference_state(raw)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    current_reference_channels = [
        raw.ch_names[pick] for pick in eeg_picks
    ]

    if not current_reference_channels:
        raise RuntimeError(
            'Average reference requires at least one good EEG channel.'
        )

    # Effective metadata are sufficient for an early return only when their
    # declared MNE implementation is also present and active in Raw.info and
    # the participating sensor space has not changed.
    if (
        reference_state.get('method') == 'average'
        and reference_state.get('status') == 'effective'
    ):
        if (
            reference_state.get('implementation') == 'mne_projection'
            and average_projector is None
        ):
            raise RuntimeError(
                'Reference metadata identify an effective MNE average '
                'projector, but no active average projector is stored.'
            )

        if reference_state.get('channels') == current_reference_channels:
            return reference_state

        # The samples already contain the former reference. Subtracting the
        # mean of the current channels produces the requested new average
        # reference. MNE's direct operation also removes the obsolete average
        # projector. A fresh projector is then created below and applied; its
        # second subtraction is numerically null and restores truthful MNE
        # provenance for source modelling and later serialization.
        raw.load_data()
        raw.set_eeg_reference(
            'average',
            projection=False,
            verbose=False,
        )
        average_projector = None

    if average_projector is None:
        # ``projection=True`` records the operation in MNE provenance first;
        # apply_proj below then makes the reference effective in the samples.
        raw.set_eeg_reference(
            'average',
            projection=True,
            verbose=False,
        )
        # Search backwards because MNE appends the newly created projector.
        average_projector = next(
            (
                projector
                for projector in reversed(raw.info['projs'])
                if projector['desc'] == 'Average EEG reference'
            ),
            None,
        )
        if average_projector is None:
            raise RuntimeError(
                'MNE did not create an average EEG reference projector.'
            )

        # prepare_eeg rejected unrelated inactive projectors before reaching
        # this helper, so applying all pending projectors is safe here.
        raw.apply_proj(verbose=False)

    _check_no_pending_projectors(raw)

    # Record MNE's actual projector columns rather than all Raw channels:
    # non-EEG and excluded channels do not participate in the average.
    raw._seegnal_reference = {
        'schema_version': 1,
        'method': 'average',
        'implementation': 'mne_projection',
        'channels': list(average_projector['data']['col_names']),
        'status': 'effective',
    }

    return raw._seegnal_reference


def _check_ica_reference(ica, mnedata):
    """
    Require an ICA and signal data to use the same effective reference.

    ICA weights are fitted in the sensor space defined by the input reference.
    Applying them to a differently referenced object would therefore combine
    incompatible transformations.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA whose reference provenance will be checked.
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        Signal data to which the ICA will be applied.

    Returns
    -------
    None
        The function returns when both reference states are equal.

    Raises
    ------
    RuntimeError
        If the ICA has no reference metadata or its state differs from the
        signal data state.
    """

    ica_reference = getattr(ica, '_seegnal_reference', None)
    if ica_reference is None:
        raise RuntimeError(
            'The ICA does not record the reference used during fitting. '
            'Fit it with sEEGnal before applying it.'
        )

    data_reference = _ensure_reference_state(mnedata)

    # Compare the complete record, including participating channels. Matching
    # only ``method`` would accept references computed from different sensors.
    if ica_reference != data_reference:
        raise RuntimeError(
            'The ICA reference does not match the effective data reference: '
            f'{ica_reference!r} != {data_reference!r}.'
        )


def _check_ica_channels(ica, raw):
    """
    Require the Raw data to preserve the ICA fitting sensor space.

    Extra non-EEG channels are harmless because MNE does not pass them through
    an EEG ICA. Extra EEG channels are accepted only when they are marked bad:
    this supports sensor-level feature workflows that retain those channels
    temporarily so their final positions can be filled with NaN. Every EEG
    channel used to fit the ICA must remain present, good and in the original
    order, with the same type, coil and unit metadata.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA decomposition.
    raw : mne.io.BaseRaw
        Continuous data to which the ICA will be applied.

    Returns
    -------
    None
        The function returns when the sensor spaces are compatible.

    Raises
    ------
    RuntimeError
        If fitted channels are missing, bad, reordered or have incompatible
        channel metadata, or if an additional good EEG channel is present.
    """

    ica_ch_names = list(getattr(ica, 'ch_names', None) or [])
    if not ica_ch_names:
        raise RuntimeError('The ICA does not record its fitted channel names.')

    raw_ch_names = list(raw.ch_names)
    raw_ch_set = set(raw_ch_names)
    missing_channels = [
        ch_name for ch_name in ica_ch_names if ch_name not in raw_ch_set
    ]
    if missing_channels:
        raise RuntimeError(
            'The Raw data are missing channel(s) used to fit the ICA: '
            f'{missing_channels}.'
        )

    fitted_channels_marked_bad = [
        ch_name for ch_name in ica_ch_names if ch_name in raw.info['bads']
    ]
    if fitted_channels_marked_bad:
        raise RuntimeError(
            'Channel(s) used to fit the ICA are marked bad in the Raw data: '
            f'{fitted_channels_marked_bad}.'
        )

    ica_ch_set = set(ica_ch_names)
    raw_fitted_order = [
        ch_name for ch_name in raw_ch_names if ch_name in ica_ch_set
    ]
    if raw_fitted_order != ica_ch_names:
        raise RuntimeError(
            'The ICA channels are not in their fitting order: '
            f'{ica_ch_names!r} != {raw_fitted_order!r}.'
        )

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    extra_good_eeg = [
        raw_ch_names[pick]
        for pick in eeg_picks
        if raw_ch_names[pick] not in ica_ch_set
        and raw_ch_names[pick] not in raw.info['bads']
    ]
    if extra_good_eeg:
        raise RuntimeError(
            'The Raw data contain good EEG channel(s) that were not used to '
            f'fit the ICA and would remain uncleaned: {extra_good_eeg}.'
        )

    ica_info = getattr(ica, 'info', None)
    if ica_info is None:
        raise RuntimeError('The ICA does not record its fitting channel metadata.')

    metadata_fields = ('kind', 'coil_type', 'unit', 'unit_mul')
    for ch_name in ica_ch_names:
        ica_index = ica_info['ch_names'].index(ch_name)
        raw_index = raw_ch_names.index(ch_name)
        ica_channel = ica_info['chs'][ica_index]
        raw_channel = raw.info['chs'][raw_index]

        mismatches = [
            field
            for field in metadata_fields
            if ica_channel[field] != raw_channel[field]
        ]
        if mismatches:
            raise RuntimeError(
                f'Channel {ch_name!r} has ICA-incompatible metadata field(s): '
                f'{mismatches}.'
            )


def _check_inverse_reference(inverse_solution, mnedata):
    """
    Require an inverse solution and data to use the same effective reference.

    LCMV filters are estimated from a sensor covariance matrix and must be
    applied to data represented in that same referenced sensor space.

    Parameters
    ----------
    inverse_solution : mne.beamformer.Beamformer | dict
        Inverse solution whose reference provenance will be checked.
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        Signal data to which the inverse solution will be applied.

    Returns
    -------
    None
        The function returns when both reference states are equal.

    Raises
    ------
    RuntimeError
        If the inverse solution has no reference metadata or its state differs
        from the signal data state.
    """

    inverse_reference = inverse_solution.get('_seegnal_reference')
    if inverse_reference is None:
        raise RuntimeError(
            'The inverse solution does not record the reference used during '
            'estimation. Re-estimate it with sEEGnal before applying it.'
        )

    data_reference = _ensure_reference_state(mnedata)

    # LCMV weights depend on the exact referenced sensor space, so method and
    # channel membership must both agree with the data being transformed.
    if inverse_reference != data_reference:
        raise RuntimeError(
            'The inverse-solution reference does not match the effective '
            f'data reference: {inverse_reference!r} != '
            f'{data_reference!r}.'
        )


def _remember_original_sample_bounds(raw):
    """
    Preserve the inclusive sample bounds of an uncropped Raw recording.

    Existing ``original_first_samp`` and ``original_last_samp`` attributes
    are retained so successive calls and crops continue to reference the
    same original recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording whose current bounds are stored when no original
        bounds are available.

    Returns
    -------
    None
        The function modifies ``raw`` in place.
    """

    # MNE updates first_samp/last_samp during crop. These custom attributes
    # deliberately retain the bounds seen before the first crop instead.
    if not hasattr(raw, 'original_first_samp'):
        raw.original_first_samp = raw.first_samp

    if not hasattr(raw, 'original_last_samp'):
        raw.original_last_samp = raw.last_samp


def _annotation_key(onset, duration, description, ch_names):
    """
    Build a stable identity key for an MNE annotation.

    Parameters
    ----------
    onset : float
        Annotation onset in seconds relative to the original recording.
    duration : float
        Annotation duration in seconds.
    description : str
        Annotation description.
    ch_names : array-like of str
        Channels associated with the annotation.

    Returns
    -------
    key : tuple
        Hashable annotation identity with temporal values rounded to twelve
        decimal places.
    """

    # TSV round-trips can change the last binary digits of a float. Rounding
    # only the identity key avoids false duplicates without altering onsets.
    return (
        round(float(onset), 12),
        round(float(duration), 12),
        str(description),
        tuple(ch_names)
    )


def _clip_annotation_to_time_window(onset, duration, tmin, tmax):
    """
    Intersect an annotation with the half-open interval ``[tmin, tmax)``.

    Temporal values are expressed in seconds. Fully contained annotations
    retain their original floating-point onset and duration; values are
    reconstructed only when an interval is actually clipped.

    Parameters
    ----------
    onset : float
        Annotation onset.
    duration : float
        Non-negative annotation duration.
    tmin : float
        Inclusive start of the target time window.
    tmax : float
        Exclusive end of the target time window.

    Returns
    -------
    clipped : tuple of float | None
        Clipped ``(onset, duration)`` pair, or ``None`` when the annotation
        does not intersect the target window.
    """

    onset = float(onset)
    duration = float(duration)
    end = onset + duration

    if duration == 0:
        if tmin <= onset < tmax:
            return onset, duration
        return None

    # Preserve the original floating-point values when no clipping is needed.
    # Reconstructing duration as ``(onset + duration) - onset`` can introduce
    # avoidable rounding differences, for example 0.3 -> 0.3000000000000007.
    if onset >= tmin and end <= tmax:
        return onset, duration

    clipped_onset = max(onset, tmin)
    clipped_end = min(end, tmax)

    if clipped_end <= clipped_onset:
        return None

    return clipped_onset, clipped_end - clipped_onset


def _set_saved_annotations(raw, annotations):
    """
    Attach original-time annotations to a possibly cropped Raw object.

    Saved onsets are interpreted as seconds from the beginning of the
    original recording. The function maps them onto the current MNE time
    axis, clips them to the available half-open sample interval and avoids
    duplicating annotations already attached to ``raw``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Current continuous recording, which may have been cropped.
    annotations : mne.Annotations
        Saved annotations expressed on the original recording time axis.

    Returns
    -------
    None
        The function updates ``raw.annotations`` in place.
    """

    if not len(annotations):
        return

    _remember_original_sample_bounds(raw)

    sfreq = raw.info['sfreq']

    # Saved TSV onsets use the original recording origin. crop_offset maps the
    # first sample still present in Raw onto that original time axis.
    crop_offset = (
        raw.first_samp - raw.original_first_samp
    ) / sfreq

    # Use n_times / sfreq to describe the exclusive boundary after the last
    # sample; raw.times[-1] is the timestamp of the last sample itself.
    current_end = crop_offset + raw.n_times / sfreq

    # MNE stores annotations on the Raw time axis, including first_samp. To
    # call set_annotations safely, convert existing annotations back to times
    # relative to the beginning of the current (possibly cropped) Raw.
    existing = raw.annotations
    existing_local_onsets = existing.onset - raw.first_time
    existing_original_onsets = existing_local_onsets + crop_offset

    # Deduplicate on the original time axis so existing and loaded annotations
    # are comparable even when Raw has been cropped.
    seen = {
        _annotation_key(onset, duration, description, ch_names)
        for onset, duration, description, ch_names in zip(
            existing_original_onsets,
            existing.duration,
            existing.description,
            existing.ch_names
        )
    }

    loaded_local_onsets = []
    loaded_durations = []
    loaded_descriptions = []
    loaded_ch_names = []

    for onset, duration, description, ch_names in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        annotations.ch_names
    ):
        # Discard annotations outside the crop and trim only intervals that
        # cross one of its boundaries.
        clipped = _clip_annotation_to_time_window(
            onset,
            duration,
            crop_offset,
            current_end
        )
        if clipped is None:
            continue

        clipped_onset, clipped_duration = clipped
        key = _annotation_key(
            clipped_onset,
            clipped_duration,
            description,
            ch_names
        )
        if key in seen:
            continue

        seen.add(key)
        # MNE expects local onsets when orig_time is None; set_annotations will
        # add Raw.first_time internally when attaching the combined object.
        loaded_local_onsets.append(clipped_onset - crop_offset)
        loaded_durations.append(clipped_duration)
        loaded_descriptions.append(description)
        loaded_ch_names.append(ch_names)

    if not loaded_local_onsets:
        return

    # Rebuild one annotation object so existing annotations are preserved and
    # all new entries go through a single MNE time-axis conversion.
    combined = mne.Annotations(
        onset=numpy.concatenate((
            existing_local_onsets,
            numpy.asarray(loaded_local_onsets)
        )),
        duration=numpy.concatenate((
            existing.duration,
            numpy.asarray(loaded_durations)
        )),
        description=numpy.concatenate((
            existing.description,
            numpy.asarray(loaded_descriptions)
        )),
        orig_time=None,
        ch_names=list(existing.ch_names) + loaded_ch_names
    )

    raw.set_annotations(combined)


def get_unannotated_raw(raw):
    """
    Materialize a Raw view without samples covered by BAD annotations.

    MNE can omit these samples directly while fitting an ICA, but ICLabel
    reads a Raw object without honoring its annotations. This helper provides
    both stages with the same usable samples. The returned time axis is
    compacted and must therefore not be used for event or annotation timing.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG data whose annotations are inspected.

    Returns
    -------
    unannotated_raw : mne.io.BaseRaw
        The original object when no samples are marked BAD, otherwise a new
        in-memory Raw containing only the samples outside BAD annotations.
    """


    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            'raw must be an MNE Raw object, '
            f'not {type(raw).__name__}.'
        )

    has_bad_annotations = any(
        description.lower().startswith('bad')
        for description in raw.annotations.description
    )
    if not has_bad_annotations:
        return raw

    # ``omit`` concatenates the good intervals. This compacted Raw is intended
    # only for ICA/ICLabel computations, never for temporal bookkeeping.
    data = raw.get_data(reject_by_annotation='omit')
    if data.shape[1] == 0:
        raise ValueError(
            'Cannot classify ICA components because BAD annotations cover '
            'the complete recording.'
        )

    if data.shape[1] == raw.n_times:
        return raw

    unannotated_raw = mne.io.RawArray(
        data,
        raw.info.copy(),
        first_samp=raw.first_samp,
        verbose=False
    )
    # Compaction changes time but not the sensor-space reference.
    _copy_reference_state(raw, unannotated_raw)

    return unannotated_raw


def fit_ica(
    mnedata,
    n_components=None,
    random_state=42,
    max_iter='auto',
    reject_by_annotation=True
):
    """
    Fit a stable Extended Infomax ICA decomposition to Raw or Epochs data.

    Parameters
    ----------
    mnedata : mne.io.BaseRaw | mne.BaseEpochs
        EEG data used to fit the decomposition.
    n_components : int | float | None
        Number of ICA components, or variance fraction to retain.
    random_state : int | None
        Random seed used for reproducibility.
    max_iter : int | 'auto'
        Maximum number of ICA iterations.
    reject_by_annotation : bool
        Whether samples covered by annotations whose description starts with
        ``BAD`` are omitted. This option only affects Raw input.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted Extended Infomax decomposition.
    """


    if not isinstance(mnedata, mnevalid):
        raise TypeError(
            'mnedata must be an MNE Raw or Epochs object, '
            f'not {type(mnedata).__name__}.'
        )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='infomax',
        fit_params={'extended': True},
        random_state=random_state,
        max_iter=max_iter
    )

    ica.fit(
        mnedata,
        picks='eeg',
        reject_by_annotation=reject_by_annotation
    )
    # ICA weights are only valid for data with this exact reference state.
    _copy_reference_state(mnedata, ica)

    return ica


def build_raw(info, data, montage=None):
    """
    Build an MNE RawArray and montage from channel metadata and signal data.

    Parameters
    ----------
    info : dict
        Channel metadata used to construct the MNE object.
    data : numpy.ndarray
        Signal data supplied to the computation.
    montage : mne.channels.DigMontage | None
        Electrode montage; the standard 10-05 montage is used by default.

    Returns
    -------
    raw : mne.io.RawArray
        Constructed MNE Raw object with channel types and montage.
    """

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

    # A newly built Raw still uses the reference present at acquisition.
    _ensure_reference_state(mneraw)

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


def prepare_eeg(
    config,
    BIDS,
    raw=None,
    preload=False,
    channels_to_include='all',
    channels_to_exclude=(),
    resample_frequency=False,
    notch_filter=False,
    freq_limits=False,
    crop_seconds=False,
    metadata_badchannels=False,
    exclude_badchannels=False,
    interpolate_badchannels=False,
    set_annotations=False,
    rereference=False,
    epoch_definition=False,
):
    """
    Apply the requested loading, cleaning, filtering and epoching operations.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    raw : mne.io.BaseRaw | None | bool
        Raw EEG object, or ``None``/``False`` to load it from BIDS. ``False``
        remains accepted for backwards compatibility.
    preload : bool
        Whether to preload signal data into memory. A supplied Raw object is
        loaded too when this is true. Operations that require loaded samples
        also load them automatically.
    channels_to_include : str | array-like | slice | None
        MNE channel selector passed to ``raw.pick()``. It may identify channel
        names, indices or channel types, or use selectors such as ``"all"``
        and ``"data"``.
    channels_to_exclude : list of str
        Exact channel names to remove after applying ``channels_to_include``.
        A channel present in both settings is excluded. Names that are not
        present in the selected data are ignored.
    resample_frequency : float | bool
        Target sampling frequency, or false to keep the original rate.
    notch_filter : bool
        Whether to apply the configured notch filters.
    freq_limits : array-like | bool
        Lower and upper filter frequencies.
    crop_seconds : float | bool
        Duration removed from both the beginning and end of the recording.
    metadata_badchannels : bool
        Whether to load bad-channel metadata.
    exclude_badchannels : bool
        Whether to remove channels marked as bad.
    interpolate_badchannels : bool
        Whether to interpolate channels marked as bad.
    set_annotations : bool
        Whether to load saved artifact annotations.
    rereference : str | bool
        Reference operation to perform. Accepted values are ``False`` to keep
        the current reference and ``'average'`` to use the average reference.
    epoch_definition : dict | bool
        Epoch mode and MNE parameters. Fixed definitions are passed to
        ``mne.make_fixed_length_epochs``; event definitions are passed to
        ``mne.Epochs`` after sEEGnal obtains the events from annotations or a
        stim channel. ``preload`` and ``verbose`` remain managed by sEEGnal.

    Returns
    -------
    data : mne.io.BaseRaw | mne.BaseEpochs
        Prepared continuous or epoched EEG data.

    Raises
    ------
    TypeError
        If parameter types do not follow the documented API.
    ValueError
        If parameter values or combinations are invalid.
    RuntimeError
        If the input data contain unapplied projectors or no good EEG channel
        remains for a requested operation.
    """

    ##################################################################
    # Validate and normalize parameters before mutating a supplied Raw
    ##################################################################

    load_from_bids = raw is None or raw is False
    if not load_from_bids and not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            'raw must be an MNE Raw object, None or False, '
            f'not {type(raw).__name__}.'
        )

    boolean_parameters = {
        'preload': preload,
        'notch_filter': notch_filter,
        'metadata_badchannels': metadata_badchannels,
        'exclude_badchannels': exclude_badchannels,
        'interpolate_badchannels': interpolate_badchannels,
        'set_annotations': set_annotations,
    }
    for parameter, value in boolean_parameters.items():
        if not isinstance(value, (bool, numpy.bool_)):
            raise TypeError(f'{parameter} must be a boolean.')

    if exclude_badchannels and interpolate_badchannels:
        raise ValueError(
            'exclude_badchannels and interpolate_badchannels cannot both '
            'be True.'
        )

    # Accept the boolean False exactly: values such as None, 0 or arbitrary
    # strings should not silently select a reference policy.
    if rereference is not False and not (
        isinstance(rereference, str) and rereference == 'average'
    ):
        raise ValueError(
            "rereference must be False or 'average', "
            f"got {rereference!r}."
        )

    if channels_to_exclude is None:
        channels_to_exclude = ()
    elif isinstance(channels_to_exclude, str):
        channels_to_exclude = (channels_to_exclude,)
    else:
        try:
            channels_to_exclude = tuple(channels_to_exclude)
        except TypeError as error:
            raise TypeError(
                'channels_to_exclude must be an iterable of channel names.'
            ) from error

    if not all(isinstance(channel, str) for channel in channels_to_exclude):
        raise TypeError(
            'channels_to_exclude must contain only channel names.'
        )

    if resample_frequency is False or resample_frequency is None:
        target_sfreq = None
    elif (
        isinstance(resample_frequency, bool)
        or not isinstance(resample_frequency, numbers.Real)
    ):
        raise TypeError('resample_frequency must be a positive number or False.')
    else:
        target_sfreq = float(resample_frequency)
        if not numpy.isfinite(target_sfreq) or target_sfreq <= 0:
            raise ValueError(
                'resample_frequency must be finite and greater than zero.'
            )

    if freq_limits is False or freq_limits is None:
        normalized_freq_limits = None
    else:
        if isinstance(freq_limits, (str, bytes)):
            raise TypeError(
                'freq_limits must contain a lower and an upper frequency.'
            )
        try:
            normalized_freq_limits = tuple(freq_limits)
        except TypeError as error:
            raise TypeError(
                'freq_limits must contain a lower and an upper frequency.'
            ) from error

        if len(normalized_freq_limits) != 2:
            raise ValueError(
                'You need to define two frequency limits to filter.'
            )

        for frequency in normalized_freq_limits:
            if frequency is None:
                continue
            if (
                isinstance(frequency, bool)
                or not isinstance(frequency, numbers.Real)
            ):
                raise TypeError(
                    'Each frequency limit must be a positive number or None.'
                )
            if not numpy.isfinite(frequency) or frequency <= 0:
                raise ValueError(
                    'Each frequency limit must be finite and greater than zero.'
                )

        if normalized_freq_limits == (None, None):
            raise ValueError('At least one frequency limit must be defined.')

        if (
            normalized_freq_limits[0] is not None
            and normalized_freq_limits[1] is not None
            and normalized_freq_limits[0] >= normalized_freq_limits[1]
        ):
            raise ValueError(
                'The lower frequency limit must be below the upper limit.'
            )

    if crop_seconds is False or crop_seconds is None:
        crop_seconds_value = 0.0
    elif (
        isinstance(crop_seconds, bool)
        or not isinstance(crop_seconds, numbers.Real)
    ):
        raise TypeError('crop_seconds must be a non-negative number or False.')
    else:
        crop_seconds_value = float(crop_seconds)
        if not numpy.isfinite(crop_seconds_value) or crop_seconds_value < 0:
            raise ValueError(
                'crop_seconds must be finite and non-negative.'
            )

    if epoch_definition is False or epoch_definition is None:
        normalized_epoch_definition = None
    elif not isinstance(epoch_definition, dict):
        raise TypeError('epoch_definition must be a dictionary or False.')
    else:
        normalized_epoch_definition = epoch_definition

    if notch_filter:
        if not isinstance(config, dict):
            raise TypeError('config must be a dictionary.')
        configured_notch_frequencies = tuple(
            config.get('global', {}).get('notch_frequencies', ())
        )
        for frequency in configured_notch_frequencies:
            if (
                isinstance(frequency, bool)
                or not isinstance(frequency, numbers.Real)
            ):
                raise TypeError(
                    'Configured notch frequencies must be positive numbers.'
                )
            if not numpy.isfinite(frequency) or frequency <= 0:
                raise ValueError(
                    'Configured notch frequencies must be finite and '
                    'greater than zero.'
                )
    else:
        configured_notch_frequencies = ()

    ##################################################################
    # Load and validate EEG
    ##################################################################

    if load_from_bids:
        raw, bids_event_id = read_BIDS_files(BIDS, preload=preload)
        raw._event_id = bids_event_id
    else:
        bids_event_id = getattr(raw, "_event_id", None)

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            'read_BIDS_files must return an MNE Raw object, '
            f'not {type(raw).__name__}.'
        )

    # MNE applies all inactive projectors together. Establish this invariant
    # before any later average-reference projector can be applied.
    _check_no_pending_projectors(raw)

    current_sfreq = float(raw.info['sfreq'])
    effective_sfreq = (
        current_sfreq if target_sfreq is None else target_sfreq
    )
    nyquist = effective_sfreq / 2

    if normalized_freq_limits is not None:
        l_freq, h_freq = normalized_freq_limits
        if l_freq is not None and l_freq >= nyquist:
            raise ValueError(
                'The lower frequency limit must be below the Nyquist '
                f'frequency ({nyquist:g} Hz).'
            )
        # Frequencies at or above Nyquist are not represented after
        # resampling. Omitting that low-pass is preferable to silently
        # mutating the caller's configuration or creating an unstable filter.
        if h_freq is not None and h_freq >= nyquist:
            h_freq = None
        normalized_freq_limits = (
            None if (l_freq, h_freq) == (None, None) else (l_freq, h_freq)
        )

    recording_duration = float(raw.times[-1])
    if (
        crop_seconds_value > 0
        and 2 * crop_seconds_value >= recording_duration
    ):
        raise ValueError(
            'crop_seconds must leave at least two samples in the recording.'
        )

    requires_loaded_samples = (
        notch_filter
        or normalized_freq_limits is not None
        or interpolate_badchannels
    )
    if (preload or requires_loaded_samples) and not raw.preload:
        raw.load_data()

    # Loaded legacy Raw objects have no sEEGnal provenance yet; initialize
    # them as effectively using their acquisition reference.
    _ensure_reference_state(raw)

    # Include and exclude channels explicitly
    raw.pick(channels_to_include)
    raw.drop_channels(channels_to_exclude, on_missing='ignore')

    # Preserve individual/BIDS electrode coordinates. The standard montage
    # is only a fallback for recordings that carry no montage at all.
    if raw.get_montage() is None:
        raw.set_montage('standard_1005', on_missing='ignore')

    ##################################################################
    # Load and optionally remove bad channels
    ##################################################################

    if any((
        metadata_badchannels,
        exclude_badchannels,
        interpolate_badchannels,
    )):
        channels_metadata = bids_tool.read_channels(config, BIDS)
        metadata_bad_names = set(
            channels_metadata.loc[
                channels_metadata['status'] == 'bad',
                'name',
            ]
        )

        # Preserve Raw channel order; set intersection made this metadata
        # nondeterministic between processes.
        badchannels = [
            channel
            for channel in raw.ch_names
            if channel in metadata_bad_names
        ]

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eeg_channels = [raw.ch_names[pick] for pick in eeg_picks]
        if eeg_channels and all(
            channel in metadata_bad_names for channel in eeg_channels
        ):
            raise RuntimeError('All EEG channels are marked as bad.')

        raw.info['bads'] = badchannels

    # Dropping before temporal operations avoids processing sensors that no
    # downstream calculation will use.
    if exclude_badchannels:
        raw.pick(None, exclude='bads')

    ##################################################################
    # Resample
    ##################################################################

    if target_sfreq is not None and not numpy.isclose(
        target_sfreq,
        current_sfreq,
        rtol=1e-12,
        atol=0,
    ):
        # MNE performs the appropriate anti-aliasing internally.
        raw.resample(target_sfreq)

    ##################################################################
    # Filters
    ##################################################################

    # Remove 50 Hz noise (and harmonics)
    if notch_filter:

        lowpass = raw.info['lowpass']
        notch_ceiling = raw.info['sfreq'] / 2
        if lowpass is not None:
            notch_ceiling = min(notch_ceiling, lowpass)

        notch_freqs = [
            frequency
            for frequency in configured_notch_frequencies
            if frequency < notch_ceiling
        ]

        if notch_freqs:
            raw.notch_filter(notch_freqs)

    # Filter the data
    if normalized_freq_limits is not None:
        raw.filter(*normalized_freq_limits)

    # Interpolate the bad channels
    if interpolate_badchannels:
        raw.interpolate_bads(reset_bads=True)

    ##################################################################
    # Crop (in seconds)
    ##################################################################

    if crop_seconds_value > 0:

        # Preserve the true original limits across successive calls/crops.
        _remember_original_sample_bounds(raw)

        # Crop
        raw.crop(
            tmin=crop_seconds_value,
            tmax=raw.times[-1] - crop_seconds_value,
        )

    ##################################################################
    # Add the annotations if requested
    ##################################################################

    if set_annotations:

        annotations = bids_tool.read_annotations(config, BIDS)
        _set_saved_annotations(raw, annotations)

    ##################################################################
    # Set reference
    ##################################################################

    if rereference == 'average':
        # The helper retains an active MNE projector for provenance while also
        # applying it immediately, so Raw and later Epochs see the same data.
        _apply_average_reference(raw)

    ##################################################################
    # Epoch the data
    ##################################################################

    if normalized_epoch_definition is not None:

        epochs = get_epochs(
            raw,
            preload,
            normalized_epoch_definition,
            bids_event_id=bids_event_id
        )

        for attribute in ('original_first_samp', 'original_last_samp'):
            if hasattr(raw, attribute):
                setattr(epochs, attribute, getattr(raw, attribute))

        return epochs


    return raw


def apply_ica(config, BIDS, raw, ica_config):
    """
    Apply category-based ICA component removal to continuous EEG data.

    The ICA is loaded from the derivative identified by ``ica_config['desc']``.
    Before any samples are changed, the function requires the Raw data to use
    the same effective reference and ordered EEG sensor space used during ICA
    fitting. Extra non-EEG channels and extra EEG channels marked bad are left
    untouched.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the ICA derivative and recording.
    raw : mne.io.BaseRaw
        Preloaded continuous EEG data. The object is modified in place.
    ica_config : dict
        ICA selection configuration. ``desc`` identifies the derivative;
        ``components_to_include`` and ``components_to_exclude`` contain
        ICLabel category names. An absent or empty include list means all
        supported categories.

    Returns
    -------
    raw : mne.io.BaseRaw
        The same Raw object after ICA component removal.

    Raises
    ------
    TypeError
        If ``raw`` is not an MNE Raw object or ``ica_config`` is not a dict.
    RuntimeError
        If data are not preloaded or their reference or sensor space is
        incompatible with the fitted ICA.
    """

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            'raw must be an MNE Raw object, '
            f'not {type(raw).__name__}.'
        )

    if not raw.preload:
        raise RuntimeError('raw must be preloaded before applying ICA.')

    if not isinstance(ica_config, dict):
        raise TypeError('ica_config must be a dictionary.')

    if 'desc' not in ica_config:
        raise KeyError('ica_config must contain a "desc" entry.')

    # Load the complete MNE ICA object.
    ica = bids_tool.read_ica(
        config,
        BIDS,
        desc=ica_config['desc']
    )
    # Check before modifying samples: ICA weights depend on the reference
    # used during fitting, not just on matching channel names.
    _check_no_pending_projectors(raw)
    _check_ica_reference(ica, raw)
    _check_ica_channels(ica, raw)

    iclabel_categories = {
        'brain',
        'muscle',
        'eog',
        'ecg',
        'line_noise',
        'ch_noise',
        'other'
    }

    categories_to_include = set(
        ica_config.get('components_to_include') or iclabel_categories
    )
    categories_to_exclude = set(
        ica_config.get('components_to_exclude') or []
    )

    unknown_categories = (
                                 categories_to_include | categories_to_exclude
                         ) - iclabel_categories

    if unknown_categories:
        raise ValueError(
            'Unknown ICLabel categories: '
            f'{sorted(unknown_categories)}.'
        )

    categories_to_keep = (
            categories_to_include - categories_to_exclude
    )

    missing_labels = categories_to_keep - set(ica.labels_)

    if missing_labels:
        raise ValueError(
            'The ICA object does not contain the requested ICLabel '
            f'categories: {sorted(missing_labels)}.'
        )

    # Find the components belonging to categories that should be kept.
    components_to_keep = {
        component
        for category in categories_to_keep
        for component in ica.labels_[category]
    }

    # MNE applies ICA removal through excluded component indices.
    all_components = set(range(ica.n_components_))
    components_to_remove = sorted(
        all_components - components_to_keep
    )

    if components_to_remove:
        ica.apply(
            raw,
            exclude=components_to_remove
        )

    return raw


def get_epochs(raw, preload, epoch_definition, bids_event_id=None):
    """
    Segment continuous data into fixed-length or event-based epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG data to segment.
    preload : bool
        Whether to preload signal data into memory.
    epoch_definition : dict
        Epoch mode and MNE parameters. Fixed definitions are passed to
        ``mne.make_fixed_length_epochs``; event definitions are passed to
        ``mne.Epochs`` after events are obtained from annotations or a stim
        channel. ``preload`` and ``verbose`` are managed by sEEGnal.
    bids_event_id : dict | None
        Event mapping recovered from BIDS.

    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data.
    """


    mode = epoch_definition.get('mode', 'fixed')

    # Work on a copy because event_source and other sEEGnal-only keys are
    # removed before forwarding the remaining arguments to MNE.
    epoch_parameters = dict(epoch_definition)
    epoch_parameters.pop('mode', None)

    # Fail loudly for the former sEEGnal vocabulary instead of letting MNE
    # raise an opaque unexpected-keyword error.
    legacy_parameters = {
        'length': 'use MNE\'s "duration" instead',
        'event_code': 'use MNE\'s "event_id" instead',
        'padding': 'MNE fixed-length epochs do not support "padding"'
    }
    unsupported_legacy = [
        parameter
        for parameter in legacy_parameters
        if parameter in epoch_parameters
    ]
    if unsupported_legacy:
        details = '; '.join(
            f'"{parameter}": {legacy_parameters[parameter]}'
            for parameter in unsupported_legacy
        )
        raise ValueError(f'Unsupported epoch parameter(s): {details}.')

    # These arguments are supplied explicitly below and must have one source
    # of truth to avoid duplicate or contradictory keyword values.
    reserved_parameters = {
        parameter
        for parameter in ('raw', 'events', 'preload', 'verbose')
        if parameter in epoch_parameters
    }
    if reserved_parameters:
        raise ValueError(
            'The following epoch parameters are managed by sEEGnal and must '
            f'not be set in epoch_definition: {sorted(reserved_parameters)}.'
        )

    ##################################################################
    # Fixed-length epochs
    ##################################################################

    if mode == 'fixed':
        if 'event_source' in epoch_parameters:
            raise ValueError(
                '"event_source" is only valid when mode is "events".'
            )
        # MNE owns onset placement, overlap and sample-count rounding here.
        epochs = mne.make_fixed_length_epochs(
            raw,
            preload=preload,
            **epoch_parameters,
            verbose=False
        )
        # Epoch construction preserves the effective sensor-space reference.
        _copy_reference_state(raw, epochs)

        return epochs

    ##################################################################
    # Event-based epochs
    ##################################################################

    elif mode == 'events':
        # event_source selects how the events array is built; it is not an
        # mne.Epochs argument and therefore must not be forwarded.
        event_source = epoch_parameters.pop('event_source', 'annotations')

        # Convert JSON-style baseline lists into tuples for MNE
        baseline = epoch_parameters.get('baseline')
        if isinstance(baseline, list):
            epoch_parameters['baseline'] = tuple(baseline)

        ##############################################################
        # Events from annotations
        ##############################################################

        if event_source == 'annotations':
            if bids_event_id is not None:
                # JSON/BIDS mappings may contain non-native scalar types;
                # normalize them to the string/int contract expected by MNE.
                annotation_event_id = {
                    str(key): int(value) for key, value in bids_event_id.items()
                }
            else:
                annotation_event_id = 'auto'

            events, _ = mne.events_from_annotations(
                raw,
                event_id=annotation_event_id,
                verbose=False
            )

        ##############################################################
        # Events from stim channel
        ##############################################################

        elif event_source == 'stim_channel':
            events = mne.find_events(raw, verbose=False)

        else:
            raise ValueError(
                "event_source must be 'annotations' or 'stim_channel'"
            )

        # MNE owns boundary rejection, event_repeated and drop_log semantics.
        epochs = mne.Epochs(
            raw,
            events,
            preload=preload,
            **epoch_parameters,
            verbose=False
        )
        # Keep provenance available for ICA and source-space compatibility
        # checks performed after segmentation.
        _copy_reference_state(raw, epochs)

        return epochs

    ##################################################################
    # Unknown mode
    ##################################################################

    else:
        raise ValueError("epoch_definition['mode'] must be 'fixed' or 'events'")
