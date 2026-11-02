"""

Generate quality-control plots and summaries for sEEGnal outputs.

Functions
---------
_compute_strict_window_spectrum
    Estimate a spectrum from complete, fixed-length good-data windows.
badchannels_qc
    Run the quality-control visualization for detected bad channels.
plot_badchannels_head
    Plot channel positions colored by their bad-channel classification.
artifact_qc
    Run the configured quality-control plots for artifact detection.
plot_components_type
    Plot ICA components grouped by their assigned component type.
plot_occipital_power_spectrum
    Plot the cleaned occipital EEG power spectrum.
plot_bad_epochs
    Summarize accepted and rejected epochs in a quality-control plot.
forward_model_qc
    Plot the forward-model source space over the template anatomy.

Federico Ramírez-Toraño
06/04/2026

"""

# Imports
import os
import copy
import numbers
from collections import Counter

import numpy
import mne
from mne.transforms import Transform
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

from sEEGnal.tools.mne_tools import apply_ica, prepare_eeg
from sEEGnal.tools.bids_tools import (
    build_derivatives_path,
    read_channels,
    read_forward_model,
    read_ica,
)
from sEEGnal.tools.template_tools import get_subjects_dir


_ARTIFACT_EPOCH_COLORS = {
    'good': 'white',
    'bad_muscle': 'tomato',
    'bad_jump': 'orange',
    'bad_other': 'gold',
    'bad_EOG': 'magenta',
    'BAD_ACQ_SKIP': 'dimgray',
}
_ARTIFACT_EPOCH_ORDER = (
    'good',
    'BAD_ACQ_SKIP',
    'bad_muscle',
    'bad_jump',
    'bad_other',
    'bad_EOG',
)
_ARTIFACT_EPOCH_DISPLAY_LABELS = {
    'BAD_ACQ_SKIP': 'acquisition loss',
}
_ICA_QC_WINDOW_SECONDS = 4.0


def _compute_strict_window_spectrum(
    raw,
    window_seconds=_ICA_QC_WINDOW_SECONDS,
    fmin=2,
    fmax=45,
):
    """
    Estimate a spectrum from complete, fixed-length good-data windows.

    Samples covered by BAD annotations split the recording into uninterrupted
    good-data spans. Each span contributes every complete non-overlapping
    window it contains; shorter spans and trailing partial windows are ignored
    for this QC calculation. Consequently every Welch estimate uses exactly
    the same data length and no window crosses an artifact boundary.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous data whose BAD annotations define the usable spans.
    window_seconds : float
        Strict duration of every Welch window in seconds.
    fmin : float
        Lowest frequency to include.
    fmax : float
        Highest frequency to include.

    Returns
    -------
    mean_psd : numpy.ndarray
        Mean power spectral density across all complete good-data windows,
        shaped ``(n_channels, n_frequencies)``.
    freqs : numpy.ndarray
        Frequencies corresponding to the last dimension of ``mean_psd``.
    """

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            'raw must be an MNE Raw object, '
            f'not {type(raw).__name__}.'
        )

    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, numbers.Real)
    ):
        raise TypeError('window_seconds must be a positive number.')

    window_seconds = float(window_seconds)
    if not numpy.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('window_seconds must be finite and greater than zero.')

    sfreq = float(raw.info['sfreq'])
    n_per_seg = int(round(window_seconds * sfreq))
    if n_per_seg < 2:
        raise ValueError(
            'window_seconds must contain at least two samples at the current '
            'sampling frequency.'
        )

    annotation_masked_data = raw.get_data(
        picks=[0],
        reject_by_annotation='NaN'
    )
    good_samples = numpy.isfinite(annotation_masked_data[0])

    padded_good_samples = numpy.pad(
        good_samples.astype(int),
        (1, 1)
    )
    transitions = numpy.diff(padded_good_samples)
    span_starts = numpy.flatnonzero(transitions == 1)
    span_stops = numpy.flatnonzero(transitions == -1)
    span_lengths = span_stops - span_starts

    eligible = span_lengths >= n_per_seg
    if not numpy.any(eligible):
        raise RuntimeError(
            'Cannot estimate the QC spectrum because no uninterrupted '
            f'good-data span contains a complete {window_seconds:g}-second '
            'window.'
        )

    weighted_psd = None
    spectrum_freqs = None
    total_windows = 0

    for span_start, span_length in zip(
        span_starts[eligible],
        span_lengths[eligible],
    ):
        n_windows = int(span_length // n_per_seg)
        span_stop = int(span_start + n_windows * n_per_seg)
        span_data = raw.get_data(
            picks='all',
            start=int(span_start),
            stop=span_stop,
        )

        span_psd, freqs = mne.time_frequency.psd_array_welch(
            span_data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_per_seg,
            n_per_seg=n_per_seg,
            n_overlap=0,
            average='mean',
            verbose=False,
        )

        if weighted_psd is None:
            weighted_psd = span_psd * n_windows
            spectrum_freqs = freqs
        else:
            weighted_psd += span_psd * n_windows
        total_windows += n_windows

    mean_psd = weighted_psd / total_windows
    return mean_psd, spectrum_freqs


### BADCHANNELS QC
def badchannels_qc(config,BIDS):
    """
    Run the quality-control visualization for detected bad channels.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Plot the badchannels in the head
    plot_badchannels_head(config,BIDS)


def _badchannel_color(description, color_dict, default_good_color='white'):
    """Return a truthful plot color for a channel-status description."""

    if description is None or (
        isinstance(description, float) and numpy.isnan(description)
    ):
        return default_good_color

    descriptions = {
        item.strip()
        for item in str(description).split(',')
        if item.strip() and item.strip() != 'n/a'
    }
    recognized = [item for item in color_dict if item in descriptions]

    if len(recognized) == 1:
        return color_dict[recognized[0]]
    if len(recognized) > 1:
        return 'black'
    if not descriptions:
        return default_good_color
    return 'gray'


def plot_badchannels_head(config,BIDS):
    """
    Plot channel positions colored by their bad-channel classification.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Draw head

    # Get the current channel information
    raw = prepare_eeg(config, BIDS, preload=False)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_pos3d = numpy.array([raw.info["chs"][pick]["loc"][:3] for pick in eeg_picks], dtype=float)
    eeg_pos2d = eeg_pos3d[:, :2].copy()
    max_abs = numpy.nanmax(numpy.abs(eeg_pos2d))
    if max_abs > 0:
        eeg_pos2d = eeg_pos2d / max_abs

    # Define categories for badchannels (and colors)
    chan = read_channels(config, BIDS)
    chan_names = list(chan['name'])
    chan_types = list(chan['status_description'])
    bad_dict = dict(zip(chan_names, chan_types))

    # Define colors
    skin_color = numpy.array([255, 243, 231]) / 255.0
    color_dict = {
        'bad_impossible_amplitude_badchannels': 'orange',
        'bad_component': 'red',
        'bad_gel_bridge': 'gold',
        'bad_high_deviation': 'magenta'
    }
    default_good_color = 'white'
    edge_color = 'black'

    # Assign a color to each EEG channel
    point_colors = []
    for ch_name in chan_names:
        point_colors.append(
            _badchannel_color(
                bad_dict[ch_name],
                color_dict,
                default_good_color
            )
        )

    # Draw head
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.25, 1.35)

    # Head
    r_head = 1.0
    head = Circle(
        (0, 0),
        r_head,
        facecolor=skin_color,
        edgecolor='black',
        linewidth=1.5
    )

    # Ears
    r_ear = 0.2
    left_ear = Circle(
        (-r_head / 1.2, 0),
        r_ear,
        facecolor=skin_color,
        edgecolor='black',
        linewidth=1.0
    )
    right_ear = Circle(
        (r_head / 1.2, 0),
        r_ear,
        facecolor=skin_color,
        edgecolor='black',
        linewidth=1.0
    )

    # Nose
    x_nose = 2 * numpy.array([-0.07, 0.07, 0.0])
    y_nose = 2 * numpy.array([0.49, 0.49, 0.55])
    nose = Polygon(
        numpy.column_stack((x_nose, y_nose)),
        closed=True,
        facecolor=skin_color,
        edgecolor='black',
        linewidth=1.0
    )

    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    ax.add_patch(nose)
    ax.add_patch(head)

    # Plot scatter on top of the head
    scatter = ax.scatter(
        eeg_pos2d[:, 0],
        eeg_pos2d[:, 1],
        s=80,
        c=point_colors,
        edgecolors=edge_color,
        linewidths=1.0,
        zorder=10
    )

    # Create legend handles
    legend_elements = []

    # Good channels (añádelo si quieres que aparezca)
    legend_elements.append(
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='good',
            markerfacecolor=default_good_color,
            markeredgecolor='black',
            markersize=8
        )
    )

    # Bad channel categories
    for key, color in color_dict.items():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=key,
                markerfacecolor=color,
                markeredgecolor='black',
                markersize=8
            )
        )

    legend_elements.extend([
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='multiple_badchannel_causes',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=8
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='unknown_badchannel_cause',
            markerfacecolor='gray',
            markeredgecolor='black',
            markersize=8
        )
    ])

    # Add legend to axis
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=True
    )
    plt.tight_layout()

    # Save the figure
    process = 'check'
    tail = 'badchannels_head'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()



### ARTIFACTS QC
def artifact_qc(config, BIDS):
    """
    Run the configured quality-control plots for artifact detection.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Check whether the occipital power spectrum looks correct
    plot_occipital_power_spectrum(config, BIDS)

    # Plot components classification
    plot_components_type(config, BIDS)

    # Plot artifactual epochs
    plot_bad_epochs(config,BIDS)


def plot_components_type(config,BIDS):
    """
    Plot ICA components grouped by their assigned component type.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Parameters for loading EEG recordings
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    set_annotations = True

    # Load raw EEG
    raw = prepare_eeg(
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
        interpolate_badchannels=False,
        set_annotations=set_annotations,
        rereference='average'
    )

    # Read the ICA information
    ica = read_ica(
        config,
        BIDS,
        desc='cleaning'
    )

    ica_sources = ica.get_sources(raw)

    # Estimate every component from complete four-second good-data windows.
    # Partial windows and short islands between annotations are ignored only
    # for this QC plot, so they do not force a shorter estimator for the rest
    # of the recording.
    component_psd, spectrum_freqs = _compute_strict_window_spectrum(
        ica_sources
    )

    # Get the common figure
    fig, axes = plt.subplots(2, 4,
                             figsize=(16, 8),
                             constrained_layout=True)
    axes = axes.flatten()

    # Plot IC classification
    iclabel_components = [
        'brain',
        'muscle',
        'eog',
        'ecg',
        'line_noise',
        'ch_noise',
        'other'
    ]

    # Plots
    for current_category, current_ax in zip(iclabel_components, axes):

        # Print the values
        component_indices = ica.labels_[current_category]

        if component_indices:
            # ICA sources use arbitrary units. Plot their amplitude spectral
            # density directly because MNE SpectrumArray accepts only data
            # channel types, whereas ICA source channels are typed as misc.
            amplitude_spectrum = numpy.sqrt(
                component_psd[component_indices]
            )
            current_ax.plot(
                spectrum_freqs,
                amplitude_spectrum.T,
                alpha=0.2,
            )

        # Add info
        current_ax.set_title(current_category)
        current_ax.set_xlabel('Frequency (Hz)')
        current_ax.set_ylabel('Amplitude spectral density (AU/√Hz)')
        current_ax.set_xlim(spectrum_freqs[0], spectrum_freqs[-1])
        current_ax.grid(True, alpha=0.3)

    # The layout has eight slots for seven ICLabel categories.
    axes[-1].axis('off')

    # Save the figure
    process = 'check'
    tail = 'check_ica_components'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()


def plot_occipital_power_spectrum(config, BIDS):
    """
    Plot the cleaned occipital EEG power spectrum.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Load the clean EEG
    ica_config = {
        'desc': 'cleaning',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
    }

    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['source_reconstruction']['epoch_definition']

    # Load the clean data
    raw = prepare_eeg(
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
        interpolate_badchannels=False,
        set_annotations=True,
        rereference='average'
    )

    raw = apply_ica(config, BIDS, raw, ica_config)

    epochs = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        exclude_badchannels=True,
        interpolate_badchannels=False,
        epoch_definition=epoch_definition
    )

    epochs.load_data()

    # Get occipital channels based on their position
    montage = epochs.get_montage()
    if montage is None:
        raise RuntimeError('No montage available to select occipital channels.')

    ch_pos = montage.get_positions()['ch_pos']
    available_channels = [ch for ch in epochs.ch_names if ch in ch_pos]

    if len(available_channels) == 0:
        raise RuntimeError('No channel positions available for the selected channels.')

    pos = numpy.array([ch_pos[ch] for ch in available_channels])

    # Posterior sensors in head coordinates (meters)
    occipital_idx = numpy.where(pos[:, 1] < -0.06)[0]
    picks = [available_channels[i] for i in occipital_idx]

    # Fallback: take the 8 most posterior channels
    if len(picks) == 0:
        n_keep = min(8, len(available_channels))
        occipital_idx = numpy.argsort(pos[:, 1])[:n_keep]
        picks = [available_channels[i] for i in occipital_idx]

    epochs_occ = epochs.copy().pick(picks)

    # Estimate spectrum
    spectrum = epochs_occ.compute_psd(
        method='welch',
        fmin=2,
        fmax=45,
    )

    freqs = spectrum.freqs.copy()
    psd = spectrum.get_data().copy()   # shape: (n_epochs, n_channels, n_freqs)

    # Convert PSD to relative power spectrum
    delta_f = freqs[1] - freqs[0]
    power_spectrum = psd * delta_f
    relative_power_spectrum = power_spectrum / power_spectrum.sum(axis=-1, keepdims=True)

    # Average across epochs -> shape: (n_channels, n_freqs)
    relative_power_spectrum_epochs_mean = numpy.mean(relative_power_spectrum, axis=0)

    # Mean and std across channels -> shape: (n_freqs,)
    relative_power_spectrum_mean = numpy.mean(relative_power_spectrum_epochs_mean, axis=0)
    power_spectrum_std = numpy.std(relative_power_spectrum_epochs_mean, axis=0)

    # Reshape to plot all epoch x channel spectra
    # shape: (n_epochs * n_channels, n_freqs)
    relative_power_spectrum_all = relative_power_spectrum.reshape(-1, relative_power_spectrum.shape[-1])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top subplot: mean ± std across occipital channels
    axes[0].plot(freqs, relative_power_spectrum_mean, label='Mean power spectrum')
    axes[0].fill_between(
        freqs,
        relative_power_spectrum_mean - power_spectrum_std,
        relative_power_spectrum_mean + power_spectrum_std,
        alpha=0.3
    )
    axes[0].set_ylabel("Relative power")
    axes[0].set_title(f"Occipital Power Spectrum (averaged across {len(picks)} channels)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Bottom subplot: all spectra from all epochs and channels
    for i_spectrum in range(relative_power_spectrum_all.shape[0]):
        axes[1].plot(freqs, relative_power_spectrum_all[i_spectrum, :], alpha=0.15, linewidth=0.8)

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative power")
    axes[1].set_title("Relative power spectrum of all occipital epochs and channels")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure
    process = 'check'
    tail = 'occipital_power_spectrum'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()


def plot_bad_epochs(config, BIDS):
    """
    Summarize accepted and rejected epochs in a quality-control plot.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Load the clean EEG
    epoch_definition = config['source_reconstruction']['epoch_definition']
    crop_seconds = config['component_estimation']['crop_seconds']

    # Load the clean data
    epochs = prepare_eeg(
        config,
        BIDS,
        preload=True,
        crop_seconds=crop_seconds,
        metadata_badchannels=True,
        exclude_badchannels=True,
        interpolate_badchannels=False,
        set_annotations=True,
        epoch_definition=epoch_definition
    )

    # Get the bad epochs info
    epoch_mask = []
    epoch_status = []
    for x in epochs.drop_log:
        if 'NO_DATA' in x:
            continue
        elif len(x) == 0:
            epoch_mask.append(True)
            epoch_status.append('good')
        else:
            epoch_mask.append(False)
            epoch_status.append(str(x[0]))

    # Define colors
    color_dict = _ARTIFACT_EPOCH_COLORS
    default_color = 'red'
    edge_color = 'black'

    # Create figure
    fig, (ax_timeline, ax_pie) = plt.subplots(
        1, 2,
        figsize=(14, 2.8),
        gridspec_kw={'width_ratios': [5, 1.8]}
    )

    # Plot one rectangle per epoch
    for iepoch, current_status in enumerate(epoch_status):
        current_color = color_dict.get(current_status, default_color)

        rect = Rectangle(
            (iepoch, 0),  # bottom-left corner
            1,  # width
            1,  # height
            facecolor=current_color,
            edgecolor='none'
        )
        ax_timeline.add_patch(rect)

    # Axis formatting
    ax_timeline.set_xlim(0, len(epoch_status))
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xticks([])
    ax_timeline.set_yticks([])

    for spine in ax_timeline.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # Legend below
    legend_order = _ARTIFACT_EPOCH_ORDER
    legend_elements = []

    for key in legend_order:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='s',
                color='w',
                label=_ARTIFACT_EPOCH_DISPLAY_LABELS.get(key, key),
                markerfacecolor=color_dict[key],
                markeredgecolor=edge_color,
                markersize=10
            )
        )

    ax_timeline.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        frameon=True,
        fontsize=9
    )

    counts = Counter(epoch_status)

    # Orden fijo (para consistencia entre sujetos)
    row_order = _ARTIFACT_EPOCH_ORDER

    # Crear datos para la tabla
    table_data = []
    row_colors = []

    for key in row_order:
        n = counts.get(key, 0)
        table_data.append([
            _ARTIFACT_EPOCH_DISPLAY_LABELS.get(key, key),
            str(n),
        ])
        row_colors.append(color_dict[key])

    # Crear tabla
    ax_pie.axis('off')

    table = ax_pie.table(
        cellText=table_data,
        colLabels=['Type', 'N'],
        cellLoc='center',
        loc='center'
    )

    # Ajustar tamaño
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Colorear filas
    for i, color in enumerate(row_colors, start=1):  # start=1 porque fila 0 son headers
        table[(i, 0)].set_facecolor(color)
        table[(i, 1)].set_facecolor(color)

    # Adjust layout so the legend fits below
    plt.subplots_adjust(bottom=0.45, wspace=0.3)

    # Save the figure
    process = 'check'
    tail = 'bad_epochs'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()



### FORWARD MODEL
def forward_model_qc(config,BIDS):
    """
    Plot the forward-model source space over the template anatomy.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.
    """

    # Read forward model
    config['subsystem'] = 'source_reconstruction'
    forward_model = read_forward_model(config, BIDS)
    src = forward_model['src']

    # Get the FreeSurfer fsaverage information
    subject = config['source_reconstruction']['forward']['template']['subject']
    subjects_dir, subject = get_subjects_dir(subject)

    # Copy src
    src_mri = copy.deepcopy(forward_model['src'])

    # Create inverse Transform from mri_head_t
    mri_head_t = forward_model['mri_head_t']  # this is an MNE Transform
    head_mri_t = mne.transforms.invert_transform(mri_head_t)

    # Apply inverse to each source point
    for s in src_mri:
        coords_hom = numpy.hstack([s['rr'], numpy.ones((s['rr'].shape[0], 1))])
        coords_mri = (head_mri_t['trans'] @ coords_hom.T).T[:, :3]
        s['rr'] = coords_mri

    # Plot
    for view in ['coronal', 'axial', 'sagittal']:

        fig = mne.viz.plot_bem(
            src=src_mri,
            trans=Transform('mri', 'head', numpy.eye(4)),  # identity transform
            subject=subject,
            subjects_dir=subjects_dir,
            brain_surfaces='white',
            orientation=view,
            slices=[50, 100, 150, 200],
            show=False
        )

        # Save the figure
        process = 'check'
        tail = f"sources_pos_{view}"
        figure_path = build_derivatives_path(BIDS, process, tail)
        if not (os.path.exists(figure_path.parent)):
            os.makedirs(figure_path.parent)
        plt.savefig(figure_path)
        plt.close()
