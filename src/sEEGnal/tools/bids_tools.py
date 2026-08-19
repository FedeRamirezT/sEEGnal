# -*- coding: utf-8 -*-
"""

Build BIDS paths and read or write sEEGnal derivative files.

Functions
---------
_artifact_annotation_paths
    Build the shared TSV and JSON paths for artifact annotations.
init_derivatives
    Decorate writers so their target derivative directories exist first.
write_annotations
    Save MNE annotations and their metadata as derivative files.
read_annotations
    Load artifact annotations or return an empty annotations object.
read_channels
    Load the derivative channel-status table.
write_badchannels
    Update and save channel status and bad-channel descriptions.
write_ica
    Save an ICA decomposition, reference provenance and ICLabel scores.
read_ica
    Load an ICA decomposition, reference provenance and ICLabel scores.
write_forward_model
    Save an EEG forward solution in the derivatives structure.
read_forward_model
    Load a previously saved EEG forward solution.
_coerce_real_lcmv_arrays
    Convert numerically real LCMV arrays stored with a complex dtype.
write_inverse_solution
    Save an inverse solution together with its reference provenance.
read_inverse_solution
    Load an inverse solution and restore its reference provenance.
_write_sensor_metadata
    Store variable-length sensor metadata as HDF5 datasets.
_read_sensor_metadata
    Load variable-length sensor metadata from HDF5 datasets.
_write_source_metadata
    Store source-input and vertex metadata as HDF5 datasets.
_read_source_metadata
    Load source-input and vertex metadata from HDF5 datasets.
write_relative_power_spectrum
    Save relative power values, frequencies and metadata in HDF5 format.
read_relative_power_spectrum
    Load relative power values, frequencies and metadata from HDF5.
write_plv
    Save PLV values and connectivity metadata for a frequency band.
read_plv
    Load PLV values and metadata for a frequency band.
write_ciplv
    Save corrected imaginary PLV values and connectivity metadata.
read_ciplv
    Load corrected imaginary PLV values and metadata.
build_standardize_path
    Construct a file path inside a recording's standardized BIDS directory.
build_derivatives_path
    Construct and create a path inside the sEEGnal derivatives tree.
build_BIDS_object
    Build an EEG BIDSPath from the configured recording identifiers.
find_matches
    Find the index of each requested value in a target array.

Federico Ramírez-Toraño
09/06/2022

"""

from functools import wraps
import shutil
import inspect
import pandas
import json
import os

import mne
import h5py
import numpy
import mne_bids

import sEEGnal.tools.tsv_tools as tsv


_ARTIFACT_ANNOTATIONS_BASENAME = 'desc-artifacts_annotations'
_ARTIFACT_ANNOTATION_COLUMNS = ('onset', 'duration', 'label')


def _artifact_annotation_paths(BIDS):
    """
    Build the shared TSV and JSON paths for artifact annotations.

    Both paths use the plural ``desc-artifacts_annotations`` basename in the
    preprocessing derivatives directory.

    Parameters
    ----------
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording.

    Returns
    -------
    tsv_file : pathlib.Path
        Artifact-annotation table path.
    json_file : pathlib.Path
        Artifact-annotation metadata path.
    """

    # Derive both files from one basename so the singular/plural convention
    # cannot drift independently between reader and writer.
    tsv_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'{_ARTIFACT_ANNOTATIONS_BASENAME}.tsv'
    )
    json_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'{_ARTIFACT_ANNOTATIONS_BASENAME}.json'
    )

    return tsv_file, json_file


def init_derivatives(func):
    """
    Decorate writers so their target derivative directories exist first.

    Parameters
    ----------
    func : callable
        Writer function to decorate.

    Returns
    -------
    decorated : callable
        Wrapped writer that initializes derivative directories.
    """


    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Call the wrapped writer after preparing its derivative directories.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the wrapped function.
        **kwargs : dict
            Keyword arguments forwarded to the wrapped function.

        Returns
        -------
        created_files : list
            Files reported by the wrapped writer.
        """

        # Get the parameteres
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        config = bound.arguments.get('config')
        BIDS = bound.arguments.get('BIDS')

        # Check if the derivatives exists
        # Creates the paths to the raw and derivative datas.
        derivatives_file_path = build_derivatives_path(BIDS, config['subsystem'], 'channels.tsv')
        derivatives_folder_path = os.path.dirname(derivatives_file_path)

        # Creates the derivatives folder, if required.
        if not os.path.isdir(derivatives_folder_path):
            os.makedirs(derivatives_folder_path)

        # Continue with the call
        created_files = func(*args, **kwargs)

        return created_files

    return wrapper


@init_derivatives
def write_annotations(config,BIDS, annotations=None):
    """
    Save MNE annotations and their metadata as derivative files.

    Existing artifact-annotation files for the same recording are replaced.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    annotations : mne.Annotations | None
        Annotations to save; an empty object is used when omitted.

    Returns
    -------
    created_files : list
        Annotation derivative files created by the function.
    """

    # Initializes the list of created files.
    created_files = []

    # If no annotation, creates an empty one.
    if annotations is None:
        annotations = mne.Annotations([], [], [])

    if not isinstance(annotations, mne.Annotations):
        raise TypeError('annotations must be an MNE Annotations object.')

    # MNE stores annotation onset and duration in seconds. Callers must pass
    # original-recording onsets; no crop offset is encoded in the TSV.
    tsv_data = pandas.DataFrame({
        'onset': annotations.onset,
        'duration': annotations.duration,
        'label': annotations.description,
    }, columns=_ARTIFACT_ANNOTATION_COLUMNS)

    # Creates the JSON dictionary.
    json_data = {
        'onset': {
            'LongName': 'Artifact onset',
            'Description': (
                'Onset relative to the beginning of the original recording.'
            ),
            'Units': 's'
        },
        'duration': {
            'LongName': 'Artifact duration',
            'Description': 'Duration of the artifact interval.',
            'Units': 's'
        },
        'label': {
            'LongName': 'Artifact label',
            'Description': 'Kind of artifact present in the interval.'
        }
    }

    # Builds the path to the files.
    tsv_file, json_file = _artifact_annotation_paths(BIDS)

    # Writes the data.
    tsv.write_tsv(tsv_data, tsv_file)

    # Writes the JSON dictionary.
    with open(json_file, 'w', encoding='utf-8') as fp:
        json.dump(json_data, fp, indent=4)

    # Appends the created file to the list.
    created_files.append(tsv_file)
    created_files.append(json_file)

    return created_files


def read_annotations(config,BIDS):
    """
    Load artifact annotations or return an empty annotations object.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    annotations : mne.Annotations
        Loaded annotations, or an empty annotations object.
    """

    # Builds the path to the file.
    tsv_file, _ = _artifact_annotation_paths(BIDS)

    # Loads the derivative data pieces.
    if tsv_file.exists():
        tsv_data = tsv.read_tsv(tsv_file, ismatrix=False)

        # Validate the schema before indexing columns so malformed derivatives
        # produce one explicit error listing every missing field.
        missing_columns = (
            set(_ARTIFACT_ANNOTATION_COLUMNS) - set(tsv_data.columns)
        )
        if missing_columns:
            raise ValueError(
                'The artifact annotations TSV is missing required columns: '
                f'{sorted(missing_columns)}.'
            )

        # Builds the MNE annotation object.
        annotations = mne.Annotations(
            tsv_data['onset'],
            tsv_data['duration'],
            tsv_data['label']
        )

    else:
        annotations = mne.Annotations([], [], [])

    # Returns the annotations object.
    return annotations


def read_channels(config, BIDS):
    """
    Load the derivative channel-status table.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    channels : pandas.DataFrame
        Channel status table.
    """

    # Builds the path to the file.
    tsv_file = build_derivatives_path(BIDS, 'preprocess', 'channels.tsv')

    # Loads the derivative data pieces.
    channels = tsv.read_tsv(tsv_file, ismatrix=False)

    # Returns the channels table object.
    return channels


@init_derivatives
def write_badchannels(config, BIDS, badchannels=None, badchannels_description=None):
    """
    Update and save channel status and bad-channel descriptions.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    badchannels : list of str | None
        Channel names classified as bad.
    badchannels_description : list of str | None
        Reason associated with each bad channel.
    """

    # Builds the path to the file.
    tsv_file = build_derivatives_path(BIDS, 'preprocess', 'channels.tsv')

    # If not exists, copy from the standardize folder
    if not(tsv_file.exists()):
        bids_file_path = build_standardize_path(BIDS, 'channels.tsv')
        derivatives_file_path = build_derivatives_path(BIDS, config['subsystem'], 'channels.tsv')
        shutil.copy(bids_file_path, derivatives_file_path)

    # Reads the contents of the raw file.
    tsv_data = mne_bids.tsv_handler._from_tsv(tsv_file)

    # Identifies the bad channels.
    badindex = find_matches(badchannels, tsv_data['name'])

    # Update the values
    for ibad in range(len(badindex)):

        # Update the status
        tsv_data['status'][badindex[ibad]] = 'bad'

        # If it is the first change in description, change it. If not, append it.
        current_status = tsv_data['status_description'][badindex[ibad]]
        if current_status == 'n/a':
            tsv_data['status_description'][badindex[ibad]] = badchannels_description[ibad]
        else:
            current_status = current_status + ',' + badchannels_description[ibad]
            tsv_data['status_description'][badindex[ibad]] = current_status

    # Saves the TSV file.
    mne_bids.tsv_handler._to_tsv(tsv_data, tsv_file)


@init_derivatives
def write_ica(config, BIDS, ica, desc='ica', overwrite=True):
    """
    Save an ICA decomposition, reference provenance and ICLabel scores.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica : mne.preprocessing.ICA
        Fitted ICA decomposition with sEEGnal reference provenance.
    desc : str
        Description identifying the derivative file.
    overwrite : bool
        Whether an existing output may be replaced.

    Returns
    -------
    created_files : list
        ICA and score files created by the function.
    """


    if not isinstance(ica, mne.preprocessing.ICA):
        raise TypeError(
            'ica must be an instance of mne.preprocessing.ICA, '
            f'not {type(ica).__name__}.'
        )

    if ica.current_fit == 'unfitted':
        raise RuntimeError('The ICA object must be fitted before saving it.')

    # Validate provenance before creating any derivative: a saved ICA without
    # its fitting reference could later be applied to incompatible data.
    reference_state = getattr(ica, '_seegnal_reference', None)
    if not isinstance(reference_state, dict):
        raise RuntimeError(
            'The ICA must record the effective fitting reference before it '
            'can be saved.'
        )

    # Build derivative paths. The FIF name must end in "_ica.fif"
    # for compatibility with MNE.
    ica_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_ica.fif'
    )
    metadata_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_ica.json'
    )
    scores_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_iclabel.tsv'
    )
    scores_metadata_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_iclabel.json'
    )

    created_files = []

    # Validate ICLabel scores before writing any file.
    scores = getattr(ica, 'labels_scores_', None)
    score_table = None
    label_names = []

    if scores is not None:
        scores = numpy.asarray(scores)

        if scores.ndim != 2:
            raise ValueError('ica.labels_scores_ must be a two-dimensional array.')

        if scores.shape[0] != ica.n_components_:
            raise ValueError(
                'The number of rows in labels_scores_ must match the '
                f'number of ICA components: {scores.shape[0]} != '
                f'{ica.n_components_}.'
            )

        # ICLabel score columns follow the insertion order of labels_, which is
        # also persisted as the column order in the accompanying JSON.
        label_names = list(getattr(ica, 'labels_', {}).keys())

        if scores.shape[1] != len(label_names):
            raise ValueError(
                'The number of columns in labels_scores_ must match the '
                f'number of labels: {scores.shape[1]} != '
                f'{len(label_names)}.'
            )

        score_table = pandas.DataFrame(
            scores,
            columns=label_names
        )
        score_table.insert(
            0,
            'component',
            numpy.arange(ica.n_components_, dtype=int)
        )

    # Save the complete ICA object.
    ica.save(ica_file, overwrite=overwrite)
    created_files.append(ica_file)

    # Store human-readable metadata not required by ICA.apply().
    extended = bool(
        getattr(ica, 'fit_params', {}).get('extended', False)
    )

    if ica.method == 'infomax' and extended:
        algorithm = 'Extended Infomax'
    elif ica.method == 'infomax':
        algorithm = 'Infomax'
    else:
        algorithm = str(ica.method)

    random_state = getattr(ica, 'random_state', None)
    if isinstance(random_state, (int, numpy.integer)):
        random_state = int(random_state)
    else:
        random_state = None

    metadata = {
        'Description': (
            'Complete MNE-Python Independent Component Analysis '
            'decomposition.'
        ),
        'Algorithm': algorithm,
        'Method': str(ica.method),
        'Extended': extended,
        'NumberOfComponents': int(ica.n_components_),
        'NumberOfChannels': len(ica.ch_names),
        'Channels': list(ica.ch_names),
        'RandomState': random_state,
        'Iterations': int(ica.n_iter_),
        # MNE's ICA FIF writer does not preserve arbitrary Python attributes,
        # so sEEGnal reference provenance belongs in this JSON sidecar.
        'Reference': reference_state,
        'Software': {
            'Name': 'MNE-Python',
            'Version': mne.__version__
        },
        'ICLabelScores': (
            os.path.basename(scores_file) if scores is not None else None
        )
    }

    with open(metadata_file, 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=4)

    created_files.append(metadata_file)

    if score_table is not None:
        # Save the score matrix with an explicit component index.
        tsv.write_tsv(
            score_table,
            scores_file,
            float_format='%.17g'
        )

        scores_metadata = {
            'Description': (
                'ICLabel prediction probabilities for each independent '
                'component.'
            ),
            'IntendedFor': os.path.basename(ica_file),
            'Software': 'ICLabel',
            'Rows': 'Independent components indexed from zero.',
            'Columns': label_names,
            'component': {
                'LongName': 'Independent component index',
                'Description': (
                    'Zero-based index of the component in the associated '
                    'MNE ICA object.'
                )
            }
        }

        with open(scores_metadata_file, 'w', encoding='utf-8') as fp:
            json.dump(scores_metadata, fp, indent=4)

        created_files.extend([
            scores_file,
            scores_metadata_file
        ])

    elif overwrite:
        # Avoid retaining scores from a previously written ICA derivative.
        for stale_file in (scores_file, scores_metadata_file):
            if os.path.isfile(stale_file):
                os.remove(stale_file)

    return created_files


@init_derivatives
def read_ica(config, BIDS, desc='ica'):
    """
    Load an ICA decomposition, reference provenance and ICLabel scores.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    desc : str
        Description identifying the derivative file.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Loaded ICA with restored reference provenance and ICLabel scores.
    """


    ica_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_ica.fif'
    )
    scores_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_iclabel.tsv'
    )
    metadata_file = build_derivatives_path(
        BIDS,
        'preprocess',
        f'desc-{desc}_ica.json'
    )

    # MNE raises FileNotFoundError if the FIF derivative does not exist.
    ica = mne.preprocessing.read_ica(ica_file)

    # Restore sEEGnal-only provenance from the sidecar after MNE reads the
    # standard FIF content.
    if os.path.isfile(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)

        reference_state = metadata.get('Reference')
        if reference_state is not None:
            if not isinstance(reference_state, dict):
                raise ValueError(
                    'ICA metadata "Reference" must contain an object: '
                    f'{metadata_file}'
                )
            ica._seegnal_reference = reference_state

    # labels_ is already restored from the FIF file. Only the custom
    # labels_scores_ attribute needs to be loaded separately.
    if os.path.isfile(scores_file):
        score_table = tsv.read_tsv(scores_file, ismatrix=False)

        if 'component' not in score_table.columns:
            raise ValueError(
                f'ICLabel score file has no "component" column: '
                f'{scores_file}'
            )

        component_index = pandas.to_numeric(
            score_table['component'],
            errors='coerce'
        )

        if component_index.isna().any():
            raise ValueError(
                f'ICLabel score file contains invalid component indices: '
                f'{scores_file}'
            )

        component_values = component_index.to_numpy()

        if not numpy.all(component_values == numpy.floor(component_values)):
            raise ValueError(
                f'ICLabel component indices must be integers: {scores_file}'
            )

        component_index = component_index.astype(int)

        if component_index.duplicated().any():
            raise ValueError(
                f'ICLabel score file contains duplicated components: '
                f'{scores_file}'
            )

        expected_components = list(range(ica.n_components_))

        if set(component_index) != set(expected_components):
            raise ValueError(
                'The components in the ICLabel score file do not match '
                'the components in the ICA object.'
            )

        # Order rows exactly as the components in the ICA object.
        score_table = score_table.assign(component=component_index)
        score_table = score_table.set_index('component')
        score_table = score_table.loc[expected_components]

        score_labels = list(score_table.columns)
        ica_labels = list(getattr(ica, 'labels_', {}).keys())

        if set(score_labels) != set(ica_labels):
            raise ValueError(
                'The labels in the ICLabel score file do not match '
                'ica.labels_.'
            )

        # Order score columns exactly as labels_.
        score_table = score_table.loc[:, ica_labels]

        ica.labels_scores_ = score_table.to_numpy(dtype=float)

    return ica


@init_derivatives
def write_forward_model(config,BIDS,forward_model):
    """
    Save an EEG forward solution in the derivatives structure.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    forward_model : mne.Forward
        Forward solution to save.

    Returns
    -------
    forward_model : mne.Forward
        Saved forward solution.
    """

    # Get the output path
    forward_path = build_derivatives_path(BIDS, config['subsystem'], 'desc-fwd.h5')

    # Save
    forward_model.save(forward_path,overwrite=True)

    return forward_model


def read_forward_model(config,BIDS):
    """
    Load a previously saved EEG forward solution.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    forward_model : mne.Forward
        Loaded forward solution.
    """

    # Get the forward model path
    forward_path = build_derivatives_path(BIDS, config['subsystem'], 'desc-fwd.h5')

    # Read
    forward_model = mne.read_forward_solution(forward_path)

    return forward_model


def _coerce_real_lcmv_arrays(inverse_solution):
    """
    Convert numerically real LCMV arrays stored with a complex dtype.

    NumPy can represent real max-power eigenvectors and weights with a
    complex dtype containing only numerical round-off in the imaginary part.
    Those arrays are converted to real values before serialization. A
    significant imaginary component is rejected instead of silently lost.

    Parameters
    ----------
    inverse_solution : mne.beamformer.Beamformer | dict
        Dictionary-like inverse solution containing the LCMV arrays.

    Returns
    -------
    inverse_solution : mne.beamformer.Beamformer | dict
        Input solution with numerically real LCMV arrays converted in place.
        Non-LCMV solutions are returned unchanged.
    """

    if inverse_solution.get('kind') != 'LCMV':
        return inverse_solution

    for key in ('weights', 'max_power_ori'):
        values = inverse_solution.get(key)
        if values is None or not numpy.iscomplexobj(values):
            continue

        # Scale the round-off threshold to the real signal magnitude, while
        # retaining a sensible absolute threshold for values below one.
        real_scale = max(float(numpy.max(numpy.abs(values.real))), 1.0)
        imag_max = float(numpy.max(numpy.abs(values.imag)))
        tolerance = 1000 * numpy.finfo(values.real.dtype).eps * real_scale

        if imag_max > tolerance:
            raise ValueError(
                f'LCMV {key} contains a significant imaginary component '
                f'({imag_max:.3e}; tolerance {tolerance:.3e}). Check the '
                'covariance, rank and forward model.'
            )

        inverse_solution[key] = values.real.copy()

    return inverse_solution


@init_derivatives
def write_inverse_solution(config,BIDS,inverse_solution):
    """
    Save an inverse solution together with its reference provenance.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    inverse_solution : mne.beamformer.Beamformer
        Inverse solution to save, including sEEGnal reference provenance.

    Returns
    -------
    inverse_solution : mne.beamformer.Beamformer
        Saved inverse solution.
    """

    # Get the output path
    inverse_solution_path = build_derivatives_path(
        BIDS,
        config['subsystem'],
        f"desc-{config['source_reconstruction']['inverse']['method']}.h5"
    )
    metadata_path = build_derivatives_path(
        BIDS,
        config['subsystem'],
        f"desc-{config['source_reconstruction']['inverse']['method']}.json"
    )

    # Beamformer weights are reference-dependent, so refuse to create an HDF5
    # derivative whose sensor-space provenance is unknown.
    reference_state = inverse_solution.get('_seegnal_reference')
    if not isinstance(reference_state, dict):
        raise RuntimeError(
            'The inverse solution must record the effective estimation '
            'reference before it can be saved.'
        )

    # NumPy may return real max-power eigenvectors in a complex-typed array.
    # Validate the imaginary component before storing a real-valued LCMV.
    inverse_solution = _coerce_real_lcmv_arrays(inverse_solution)

    # Reference provenance belongs to sEEGnal's sidecar contract rather than
    # MNE's Beamformer HDF5 schema. Restore it even if serialization fails.
    inverse_solution.pop('_seegnal_reference')
    try:
        inverse_solution.save(inverse_solution_path,overwrite=True)
    finally:
        inverse_solution['_seegnal_reference'] = reference_state

    # Keep provenance beside the HDF5 in a small JSON file that sEEGnal can
    # evolve without depending on MNE's internal Beamformer schema.
    with open(metadata_path, 'w', encoding='utf-8') as fp:
        json.dump({'Reference': reference_state}, fp, indent=4)

    return inverse_solution


def read_inverse_solution(config,BIDS):
    """
    Load an inverse solution and restore its reference provenance.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    inverse_solution : mne.beamformer.Beamformer
        Loaded inverse solution with restored reference provenance.
    """

    # Get the inverse solution path
    inverse_solution_path = build_derivatives_path(
        BIDS,
        config['subsystem'],
        f"desc-{config['source_reconstruction']['inverse']['method']}.h5"
    )
    metadata_path = build_derivatives_path(
        BIDS,
        config['subsystem'],
        f"desc-{config['source_reconstruction']['inverse']['method']}.json"
    )

    # Read
    inverse_solution = mne.beamformer.read_beamformer(inverse_solution_path)
    inverse_solution = _coerce_real_lcmv_arrays(inverse_solution)

    # MNE reconstructs the Beamformer dictionary from HDF5; restore the
    # sEEGnal-only reference entry from its sidecar afterwards.
    if os.path.isfile(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)

        reference_state = metadata.get('Reference')
        if reference_state is not None:
            if not isinstance(reference_state, dict):
                raise ValueError(
                    'Inverse-solution metadata "Reference" must contain an '
                    f'object: {metadata_path}'
                )
            inverse_solution['_seegnal_reference'] = reference_state

    return inverse_solution

_SENSOR_STRING_METADATA = ('ch_names', 'bad_channels')
_SENSOR_INDEX_METADATA = (
    'good_channel_indices',
    'bad_channel_indices',
    'nan_channel_indices',
    'nan_connection_indices',
)
_SOURCE_STRING_METADATA = (
    'input_ch_names',
    'input_good_channels',
    'input_bad_channels',
    'input_excluded_channels',
    'source_vertex_spaces',
)
_SOURCE_INDEX_METADATA = (
    'input_good_channel_indices',
    'input_bad_channel_indices',
    'input_excluded_channel_indices',
    'source_vertex_indices',
    'source_vertex_numbers',
)


def _write_sensor_metadata(group, metadata):
    """
    Store variable-length sensor metadata as HDF5 datasets.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group that receives the metadata datasets.
    metadata : dict
        Output metadata containing optional sensor names and indices.
    """

    for key in _SENSOR_STRING_METADATA:
        if key in metadata:
            group.create_dataset(
                key,
                data=numpy.asarray(metadata[key], dtype='S'),
            )

    for key in _SENSOR_INDEX_METADATA:
        if key in metadata:
            group.create_dataset(
                key,
                data=numpy.asarray(metadata[key], dtype=numpy.int64),
            )


def _read_sensor_metadata(group, metadata):
    """
    Load variable-length sensor metadata from HDF5 datasets.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing optional sensor metadata datasets.
    metadata : dict
        Output dictionary updated in place with decoded names and indices.
    """

    for key in _SENSOR_STRING_METADATA:
        if key in group:
            metadata[key] = [
                value.decode('utf-8')
                for value in group[key][:]
            ]

    for key in _SENSOR_INDEX_METADATA:
        if key in group:
            metadata[key] = group[key][:].astype(int).tolist()


def _write_source_metadata(group, metadata):
    """
    Store source-input and vertex metadata as HDF5 datasets.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group that receives the metadata datasets.
    metadata : dict
        Output metadata containing optional sensor-input and source mappings.
    """

    for key in _SOURCE_STRING_METADATA:
        if key in metadata:
            group.create_dataset(
                key,
                data=numpy.asarray(metadata[key], dtype='S'),
            )

    for key in _SOURCE_INDEX_METADATA:
        if key in metadata:
            group.create_dataset(
                key,
                data=numpy.asarray(metadata[key], dtype=numpy.int64),
            )


def _read_source_metadata(group, metadata):
    """
    Load source-input and vertex metadata from HDF5 datasets.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing optional source metadata datasets.
    metadata : dict
        Output dictionary updated in place with decoded mappings.
    """

    for key in _SOURCE_STRING_METADATA:
        if key in group:
            metadata[key] = [
                value.decode('utf-8')
                for value in group[key][:]
            ]

    for key in _SOURCE_INDEX_METADATA:
        if key in group:
            metadata[key] = group[key][:].astype(int).tolist()


@init_derivatives
def write_relative_power_spectrum(config, BIDS, relative_power_spectrum=None, metadata=None):
    """
    Save relative power values, frequencies and metadata in HDF5 format.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    relative_power_spectrum : numpy.ndarray | None
        Relative power values to save.
    metadata : dict | None
        Metadata associated with the output values.
    """

    # Determine file name
    if 'sensor' in config['current_space']:
        filename = "desc-rel_pow_sensor.h5"
    elif 'source' in config['current_space']:
        filename = "desc-rel_pow_source.h5"
    else:
        raise ValueError("current_space must be 'sensor' or 'source'")

    rel_pow_path = build_derivatives_path(BIDS, config['subsystem'], filename)

    overwrite = bool(config['feature_extraction']['relative_power_spectrum'].get('overwrite', True))
    mode = "w" if overwrite else "x"

    with h5py.File(rel_pow_path, mode) as f:
        grp = f.create_group("power")

        # Datasets
        grp.create_dataset("relative_power_spectrum", data=relative_power_spectrum, compression="gzip")
        grp.create_dataset("freqs", data=metadata['freqs'], compression="gzip")
        if 'sensor' in config['current_space']:
            _write_sensor_metadata(grp, metadata)
        if 'source' in config['current_space']:
            _write_source_metadata(grp, metadata)

        # Attributes for scalar/short metadata
        for key, value in metadata.items():
            if key not in (
                'freqs',
                *_SENSOR_STRING_METADATA,
                *_SENSOR_INDEX_METADATA,
                *_SOURCE_STRING_METADATA,
                *_SOURCE_INDEX_METADATA,
            ):
                grp.attrs[key] = value


def read_relative_power_spectrum(config, BIDS, space='sensor'):
    """
    Load relative power values, frequencies and metadata from HDF5.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    space : str
        Signal space to read, such as sensor or source.

    Returns
    -------
    relative_power_spectrum : numpy.ndarray
        Relative power values.
    freqs : numpy.ndarray
        Spectral frequencies.
    metadata : dict
        Stored power-spectrum metadata.
    """

    filename = "desc-rel_pow_sensor.h5" if space=='sensor' else "desc-rel_pow_source.h5"
    rel_pow_path = build_derivatives_path(BIDS, config['subsystem'], filename)

    with h5py.File(rel_pow_path, "r") as f:
        grp = f["power"]
        relative_power_spectrum = grp["relative_power_spectrum"][:]
        freqs = grp["freqs"][:]
        metadata = dict(grp.attrs)
        _read_sensor_metadata(grp, metadata)
        _read_source_metadata(grp, metadata)

    return relative_power_spectrum, freqs, metadata


@init_derivatives
def write_plv(config,BIDS,plv=None,metadata=None):
    """
    Save PLV values and connectivity metadata for a frequency band.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    plv : numpy.ndarray | None
        PLV values to save.
    metadata : dict | None
        Metadata associated with the output values.
    """

    if 'sensor' in config['current_space']:

        # Get the output path
        plv_path = build_derivatives_path(
            BIDS,
            config['subsystem'],
            f"desc-plv_sensor_{metadata['band_name']}_band.h5"
        )

    if 'source' in config['current_space']:
        # Get the output path
        plv_path = build_derivatives_path(
            BIDS,
            config['subsystem'],
            f"desc-plv_source_{metadata['band_name']}_band.h5"
        )

    with h5py.File(plv_path, "w") as f:

        g = f.create_group("fc")

        # datasets
        g.create_dataset("plv", data=plv.astype(numpy.float32))

        # attributes
        g.attrs["method"] = metadata["method"]
        g.attrs["n_nodes"] = metadata["n_nodes"]
        g.attrs["ch_names"] = metadata['ch_names']
        g.attrs["n_epochs_used"] = metadata["n_epochs_used"]
        g.attrs["band_name"] = metadata["band_name"]
        g.attrs["description"] = metadata["description"]
        for key in (
            'n_good_nodes',
            'bad_channel_policy',
            'vectorization',
            'n_sources',
            'n_input_channels',
            'n_good_input_channels',
            'input_bad_channel_policy',
            'source_space_type',
            'source_subject',
            'source_spacing',
            'source_value_policy',
        ):
            if key in metadata:
                g.attrs[key] = metadata[key]
        if 'sensor' in config['current_space']:
            _write_sensor_metadata(g, metadata)
        if 'source' in config['current_space']:
            _write_source_metadata(g, metadata)


def read_plv(config, BIDS, space='sensor',band_name=None):
    """
    Load PLV values and metadata for a frequency band.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    space : str
        Signal space to read, such as sensor or source.
    band_name : str | None
        Name of the frequency band to read.

    Returns
    -------
    plv : numpy.ndarray
        Stored PLV values.
    metadata : dict
        Stored connectivity metadata.
    """

    filename = f"desc-plv_sensor_{band_name}_band.h5" if space == 'sensor' else f"desc-plv_source_{band_name}_band.h5"
    plv_path = build_derivatives_path(BIDS, config['subsystem'], filename)

    with h5py.File(plv_path, "r") as f:

        grp = f["fc"]
        plv = grp["plv"][:]
        metadata = dict(grp.attrs)
        _read_sensor_metadata(grp, metadata)
        _read_source_metadata(grp, metadata)

    return plv, metadata


@init_derivatives
def write_ciplv(config,BIDS,ciplv=None,metadata=None):
    """
    Save corrected imaginary PLV values and connectivity metadata.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ciplv : numpy.ndarray | None
        Corrected imaginary PLV values to save.
    metadata : dict | None
        Metadata associated with the output values.
    """

    if 'sensor' in config['current_space']:

        # Get the output path
        ciplv_path = build_derivatives_path(
            BIDS,
            config['subsystem'],
            f"desc-ciplv_sensor_{metadata['band_name']}_band.h5"
        )

    if 'source' in config['current_space']:
        # Get the output path
        ciplv_path = build_derivatives_path(
            BIDS,
            config['subsystem'],
            f"desc-ciplv_source_{metadata['band_name']}_band.h5"
        )

    with h5py.File(ciplv_path, "w") as f:

        g = f.create_group("fc")

        # datasets
        g.create_dataset("ciplv", data=ciplv.astype(numpy.float32))


        # attributes
        g.attrs["method"] = metadata["method"]
        g.attrs["n_nodes"] = metadata["n_nodes"]
        g.attrs["ch_names"] = metadata['ch_names']
        g.attrs["n_epochs_used"] = metadata["n_epochs_used"]
        g.attrs["band_name"] = metadata["band_name"]
        g.attrs["description"] = metadata["description"]
        for key in (
            'n_good_nodes',
            'bad_channel_policy',
            'vectorization',
            'n_sources',
            'n_input_channels',
            'n_good_input_channels',
            'input_bad_channel_policy',
            'source_space_type',
            'source_subject',
            'source_spacing',
            'source_value_policy',
        ):
            if key in metadata:
                g.attrs[key] = metadata[key]
        if 'sensor' in config['current_space']:
            _write_sensor_metadata(g, metadata)
        if 'source' in config['current_space']:
            _write_source_metadata(g, metadata)


def read_ciplv(config, BIDS, space='sensor',band_name=None):
    """
    Load corrected imaginary PLV values and metadata.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    space : str
        Signal space to read, such as sensor or source.
    band_name : str | None
        Name of the frequency band to read.

    Returns
    -------
    ciplv : numpy.ndarray
        Stored corrected imaginary PLV values.
    metadata : dict
        Stored connectivity metadata.
    """

    filename = f"desc-ciplv_sensor_{band_name}_band.h5" if space == 'sensor' else f"desc-ciplv_source_{band_name}_band.h5"
    plv_path = build_derivatives_path(BIDS, config['subsystem'], filename)

    with h5py.File(plv_path, "r") as f:
        grp = f["fc"]
        plv = grp["ciplv"][:]
        metadata = dict(grp.attrs)
        _read_sensor_metadata(grp, metadata)
        _read_source_metadata(grp, metadata)

    return plv, metadata


def build_standardize_path(BIDS, fname_tail):
    """
    Construct a file path inside a recording's standardized BIDS directory.

    Parameters
    ----------
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    fname_tail : str
        Final entity and extension appended to the BIDS basename.

    Returns
    -------
    path : pathlib.Path
        Path inside the standardized BIDS recording.
    """

    from mne_bids.config import ALLOWED_PATH_ENTITIES_SHORT

    # Gets the base filename structure.
    basename = []

    for key, val in BIDS.entities.items():
        if val is not None and key != 'datatype':
            long_to_short_entity = {val: key for key, val in ALLOWED_PATH_ENTITIES_SHORT.items()}
            key = long_to_short_entity[key]
            basename.append(f'{key}-{val}')

    # Builds the file name for the piece of data.
    file_bids = '_'.join(basename + [fname_tail])

    # Builds the complete path to the piece of data.
    path_rel = BIDS.directory.relative_to(BIDS.root)
    path_bids = os.path.join(BIDS.root,path_rel,file_bids)

    return path_bids


def build_derivatives_path(BIDS, process, fname_tail):
    """
    Construct and create a path inside the sEEGnal derivatives tree.

    Parameters
    ----------
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    process : str
        Processing stage used as the derivative subdirectory.
    fname_tail : str
        Final entity and extension appended to the BIDS basename.

    Returns
    -------
    path : pathlib.Path
        Path inside the sEEGnal derivatives tree.
    """

    from mne_bids.config import ALLOWED_PATH_ENTITIES_SHORT

    # Gets the base filename structure.
    basename = []

    for key, val in BIDS.entities.items():
        if val is not None and key != 'datatype':
            long_to_short_entity = {val: key for key, val in ALLOWED_PATH_ENTITIES_SHORT.items()}
            key = long_to_short_entity[key]
            basename.append(f'{key}-{val}')

    # Builds the file name for the piece of data.
    file_der = '_'.join(basename + [fname_tail])

    # Builds the complete path to the piece of data.
    path_rel = BIDS.directory.relative_to(BIDS.root)
    path_etl = BIDS.root.joinpath(os.path.join('derivatives', 'sEEGnal',process))
    path_der = path_etl.joinpath(path_rel).joinpath(file_der)

    # Returns the path to the piece of data.
    return path_der


def build_BIDS_object(
    config,
    current_sub,
    current_ses,
    current_task,
    current_run=None
):
    """
    Build an EEG BIDSPath from the configured recording identifiers.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    current_sub : str
        Subject identifier.
    current_ses : str
        Session identifier.
    current_task : str
        Task identifier.
    current_run : str | None
        Run identifier. The public manifest always supplies this value; None
        remains accepted for direct API compatibility.

    Returns
    -------
    BIDS : mne_bids.BIDSPath
        BIDS path for the requested recording.
    """

    # Builds the BIDS path from the metadata.
    BIDS = mne_bids.BIDSPath(
        subject=current_sub,
        session=current_ses,
        task=current_task,
        run=current_run,
        datatype='eeg',
        root=config['path']['data_root'],
        suffix='eeg',
        extension='.vhdr'
    )

    return BIDS


# Helper function to find matches of one array in another.
def find_matches(array, target):
    """
    Find the index of each requested value in a target array.

    Parameters
    ----------
    array : array-like
        Values whose positions are requested.
    target : array-like
        Values in which matches are searched.

    Returns
    -------
    matches : list of int | None
        Target index corresponding to each requested value.
    """

    # Converts the inputs into Numpy arrays, if required.
    array = numpy.asarray(array)
    target = numpy.asarray(target)

    # Initializes the array of matches.
    matches = [None] * len(array)

    # Iterates through the array.
    for index in range(array.size):

        # Looks for the match.
        match = numpy.flatnonzero(target == array[index])

        # Stores the match, if any.
        if match.size > 0:
            matches[index] = match[0]

    # Returns the matches.
    return matches
