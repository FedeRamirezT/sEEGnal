"""

Build an EEG forward model using the configured source template.

Functions
---------
make_forward_model
    Select, estimate and save the configured EEG forward model.
template_forward_model
    Build a forward solution using the configured template anatomy.

Federico Ramírez-Toraño
11/02/2026

"""

# Import
import traceback
from datetime import datetime as dt, timezone

import mne

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import write_forward_model
from sEEGnal.tools.template_tools import get_subjects_dir
from sEEGnal.tools.qc_tools import forward_model_qc



def make_forward_model(config, BIDS):
    """
    Select, estimate and save the configured EEG forward model.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    results : dict
        Execution status and forward-model information.
    """

    # Add the subsystem info
    config['subsystem'] = 'source_reconstruction'

    try:

        # Choose the correct estimation
        if config['source_reconstruction']['forward']['use_template']:

            # Estimate the forward model
            forward_model = template_forward_model(config, BIDS)

        else:
            forward_model = []

        # If estimated, QC
        if isinstance(forward_model,mne.forward.forward.Forward):
            forward_model_qc(config,BIDS)

        # Save the metadata
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        results = {
            'result': 'ok',
            'bids_basename': BIDS.basename,
            "date": formatted_now,
            'forward_model': forward_model
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



    return results


def template_forward_model(config, BIDS):
    """
    Build a forward solution using the configured template anatomy.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    forward_model : mne.Forward
        Forward solution built from template anatomy.
    """


    # Load the EEG to obtain the montage
    raw = prepare_eeg(config, BIDS)

    # Get the FreeSurfer fsaverage information
    subject = config['source_reconstruction']['forward']['template']['subject']
    subjects_dir, subject = get_subjects_dir(subject)

    bem = (
            subjects_dir
            / subject
            / 'bem'
            / config['source_reconstruction']['forward']['template']['bem']
    )
    trans = config['source_reconstruction']['forward']['template']['trans']

    # Define our sources
    spacing = config['source_reconstruction']['forward']['template']['spacing']
    src = mne.setup_source_space(
        subject=subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False
    )

    # Read the BEM solution (MNI template)
    bem = mne.read_bem_solution(bem)

    # Create the output forward model
    forward_model = mne.make_forward_solution(
        raw.info,
        trans,
        src,
        bem,
        meg=False,
        eeg=True
    )

    # Save the forward solution
    write_forward_model(config, BIDS, forward_model)

    # Save the forward model
    return forward_model
