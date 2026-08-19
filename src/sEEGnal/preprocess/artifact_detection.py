# -*- coding: utf-8 -*-
"""

Detect and annotate EEG artifacts using independent component analysis.

First, it estimates an Extended Infomax ICA after excluding bad channels
and detects muscle and sensor artifacts. These artifacts can negatively
affect the estimation of the remaining independent components.

After annotating muscle and sensor artifacts, a final ICA is fitted while
excluding the annotated segments. EOG and other artifacts are then detected.

Functions
---------
artifact_detection
    Coordinate artifact detection, quality control and execution reporting.
eeg_artifact_detection
    Run the complete artifact-detection sequence for an EEG recording.
estimate_artifact_components
    Fit, classify and save the ICA decomposition used to detect artifacts.
EOG_detection
    Convert detected ocular-artifact samples into MNE annotations.
muscle_detection
    Convert detected muscle-artifact samples into MNE annotations.
sensor_detection
    Convert detected sensor-jump samples into MNE annotations.
other_detection
    Convert other implausible-signal samples into MNE annotations.

Federico Ramírez-Toraño
27/09/2023

"""

# Imports
import traceback
from datetime import datetime as dt, timezone

import mne
import mne_icalabel.iclabel as iclabel

import sEEGnal.tools.bids_tools as bids
import sEEGnal.tools.mne_tools as mne_tools
import sEEGnal.preprocess.find_artifacts as find_artifacts
from sEEGnal.tools.qc_tools import artifact_qc

# Set the output levels
mne.utils.set_log_level(verbose='ERROR')


# Modules
def artifact_detection(config, BIDS):
    """
    Coordinate artifact detection, quality control and execution reporting.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    results : dict
        Execution status and generated artifact annotations.
    """

    # Add the subsystem flag
    config['subsystem'] = 'preprocess'

    # For EEG
    if BIDS.datatype == 'eeg':

        try:

            # Detect badchannels in the current recording
            annotations = eeg_artifact_detection(config, BIDS)

            # QC
            artifact_qc(config, BIDS)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {
                'result': 'ok',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'annotations': annotations
            }

        except Exception as e:

            # Save the error
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            results = {
                'result': 'error',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                "details": f"Exception: {str(e)}, {traceback.format_exc()}"
            }

    else:

        # Not accepted type to process
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'error',
            'file': BIDS.basename,
            'details': 'Not accepted type of file to process',
            'date': formatted_now
        }

    return results


def eeg_artifact_detection(config, BIDS):
    """
    Run the complete artifact-detection sequence for an EEG recording.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    annotations : mne.Annotations
        Combined artifact annotations.
    """


    # Estimate the preliminary ICA used to detect artifacts.
    ica_desc = 'artifacts'
    estimate_artifact_components(config, BIDS, ica_desc)

    # Muscle and sensor artifact detection.
    # Muscle
    muscle_annotations = muscle_detection(config, BIDS)

    # Save the muscle annotations for better jump detection
    _ = bids.write_annotations(config,BIDS, muscle_annotations)

    # Sensor (jumps)
    sensor_annotations = sensor_detection(config, BIDS,ica_desc)

    # Impossible amplitude
    other_annotations = other_detection(config, BIDS,ica_desc)

    # Combine the annotations
    annotations = muscle_annotations.__add__(sensor_annotations).__add__(other_annotations)

    # Save the annotations in BIDS format
    _ = bids.write_annotations(config,BIDS, annotations)

    # Estimate the final ICA excluding the previously detected artifacts.
    ica_desc = 'cleaning'
    estimate_artifact_components(config, BIDS, ica_desc)

    # I have to find again all the artifacts
    # For EEG
    # Muscle
    muscle_annotations = muscle_detection(config, BIDS)

    # Save the muscle annotations for better jump detection
    _ = bids.write_annotations(config,BIDS, muscle_annotations)

    # Sensor (jumps)
    sensor_annotations = sensor_detection(config, BIDS,ica_desc)

    # EOG
    # Select the frontal channels
    EOG_annotations = EOG_detection(config, BIDS)

    # Impossible amplitude
    other_annotations = other_detection(config, BIDS,ica_desc)

    # Merge all the annotations into a MNE Annotation object
    annotations = other_annotations.__add__(EOG_annotations).__add__(muscle_annotations).__add__(sensor_annotations)

    # Save the annotations in BIDS format
    _ = bids.write_annotations(config,BIDS, annotations)

    return annotations


def estimate_artifact_components(config, BIDS, ica_desc):
    """
    Fit, classify and save the ICA decomposition used to detect artifacts.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica_desc : str
        Description identifying the ICA derivative to use.
    """


    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config[
        'component_estimation'
    ]['resample_frequency']
    channels_to_include = config['global']['channels_to_include']
    channels_to_exclude = config['global']['channels_to_exclude']
    # The final ICA excludes previously annotated artifacts.
    set_annotations = ica_desc == 'cleaning'

    # Fit the ICA only on channels that remain usable after bad-channel
    # detection. The same subset is used when the ICA is applied later.
    raw = mne_tools.prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        resample_frequency=resample_frequency,
        notch_filter=True,
        freq_limits=freq_limits,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True,
        set_annotations=set_annotations,
        rereference='average'
    )

    # Fit Extended Infomax ICA, omitting samples already marked BAD.
    ica = mne_tools.fit_ica(raw, reject_by_annotation=True)

    # ICLabel does not honor Raw annotations when extracting its temporal
    # features, so materialize the same usable samples seen by ICA.fit.
    classification_raw = mne_tools.get_unannotated_raw(raw)

    # Classify the independent components and retain the probability assigned
    # to every ICLabel category for each component.
    ica.labels_scores_ = iclabel.iclabel_label_components(
        classification_raw,
        ica,
        inplace=True,
        backend='onnx'
    )

    # Components with insufficient classification confidence are
    # reassigned to the "other" category.
    max_probability = ica.labels_scores_.max(axis=1)
    unclear_components = {
        component
        for component, probability in enumerate(max_probability)
        if probability
        < config['component_estimation']['unclear_threshold']
    }

    classified_labels = [
        'brain',
        'muscle',
        'eog',
        'ecg',
        'line_noise',
        'ch_noise'
    ]

    for label in classified_labels:
        ica.labels_[label] = [
            component
            for component in ica.labels_[label]
            if component not in unclear_components
        ]

    ica.labels_['other'] = sorted(
        set(ica.labels_['other']) | unclear_components
    )

    bids.write_ica(
        config,
        BIDS,
        ica,
        desc=ica_desc
    )


def EOG_detection(config, BIDS):
    """
    Convert detected ocular-artifact samples into MNE annotations.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    annotations : mne.Annotations
        Ocular-artifact annotations.
    """


    # Find EOG aritfacts index
    EOG_index, n_times, sfreq = find_artifacts.EOG_detection(config, BIDS)

    # If any artifact
    if len(EOG_index) > 0:

        # Create the annotations
        EOG_annotations = find_artifacts.create_annotations(
            EOG_index,
            n_times,
            sfreq,
            'bad_EOG'
        )

    else:

        # Create an empty Annotation
        EOG_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return EOG_annotations


def muscle_detection(config, BIDS):
    """
    Convert detected muscle-artifact samples into MNE annotations.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    annotations : mne.Annotations
        Muscle-artifact annotations.
    """


    # Look the position of muscular artifacts
    muscle_index, n_times, sfreq = find_artifacts.muscle_detection(
        config,
        BIDS)

    # If any index
    if len(muscle_index) > 0:

        # Create the annotations
        muscle_index.sort()
        muscle_annotations = find_artifacts.create_annotations(
            muscle_index,
            n_times,
            sfreq,
            'bad_muscle',
            fictional_artifact_duration=0.5
        )

    else:

        # If no artifacts, create empty Annotation
        muscle_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return muscle_annotations


def sensor_detection(config, BIDS, ica_desc):
    """
    Convert detected sensor-jump samples into MNE annotations.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica_desc : str
        Description identifying the ICA derivative to use.

    Returns
    -------
    annotations : mne.Annotations
        Sensor-artifact annotations.
    """


    # Look the position of muscular artifacts
    sensor_index, n_times, sfreq = find_artifacts.sensor_detection(
        config,
        BIDS,
        ica_desc)

    # Create as Annotations
    if len(sensor_index) > 0:

        # Create the annotations
        sensor_index.sort()  # First sort the list
        sensor_annotations = find_artifacts.create_annotations(
            sensor_index,
            n_times,
            sfreq,
            'bad_jump',
            fictional_artifact_duration=0.3
        )

    # If no artifacts, create empty Annotation
    else:
        sensor_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return sensor_annotations


def other_detection(config, BIDS,ica_desc):
    """
    Convert other implausible-signal samples into MNE annotations.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    ica_desc : str
        Description identifying the ICA derivative to use.

    Returns
    -------
    annotations : mne.Annotations
        Other artifact annotations.
    """


    # Look the position of muscular artifacts
    other_index, n_times, sfreq = find_artifacts.other_detection(
        config,
        BIDS,
        ica_desc
    )

    # Create as Annotations
    if len(other_index) > 0:

        # Create the annotations
        other_index.sort()  # First sort the list
        sensor_annotations = find_artifacts.create_annotations(
            other_index,
            n_times,
            sfreq,
            'bad_other',
            fictional_artifact_duration=0.3
        )

    # If no artifacts, create empty Annotation
    else:
        sensor_annotations = mne.Annotations(onset=[], duration=[], description=[])

    return sensor_annotations
