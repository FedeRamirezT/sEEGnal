"""
Quality check utilities for sEEGnal

Federico Ramírez-Toraño
06/04/2026
"""

# Imports
import os
import copy
from collections import Counter

import numpy
import mne
from mne.transforms import Transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.bids_tools import build_derivatives_path, read_sobi, read_badchannels, read_forward_model
from sEEGnal.tools.template_tools import get_subjects_dir

# Activate plot in debugging
matplotlib.use('Qt5Agg')


### BADCHANNELS QC
def badchannels_qc(config,BIDS):

    # Plot the badchannels in the head
    plot_badchannels_head(config,BIDS)


def plot_badchannels_head(config,BIDS):

    # Draw head

    # Get the current channel information
    raw = prepare_eeg(
        config,
        BIDS,
        preload=False)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_pos3d = numpy.array([raw.info["chs"][pick]["loc"][:3] for pick in eeg_picks], dtype=float)
    eeg_pos2d = eeg_pos3d[:, :2].copy()
    max_abs = numpy.nanmax(numpy.abs(eeg_pos2d))
    if max_abs > 0:
        eeg_pos2d = eeg_pos2d / max_abs

    # Define categories for badchannels (and colors)
    chan = read_badchannels(config, BIDS)
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
        current_type = bad_dict[ch_name]

        if isinstance(current_type, float) and numpy.isnan(current_type):
            point_colors.append(default_good_color)
        else:
            point_colors.append(color_dict.get(current_type, 'red'))

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

    # Check whether the occipital power spectrum looks correct
    plot_occipital_power_spectrum(config, BIDS)

    # Plot components classification
    plot_components_type(config, BIDS)

    # Plot artifactual epochs
    plot_bad_epochs(config,BIDS)


def plot_components_type(config,BIDS):

    # Parameters for loading EEG recordings
    freq_limits = [
        config['component_estimation']['low_freq'],
        config['component_estimation']['high_freq']
    ]
    crop_seconds = config['component_estimation']['crop_seconds']
    resample_frequency = config['component_estimation']['resample_frequency']
    channels_to_include = config['global']["channels_to_include"]
    channels_to_exclude = config['global']["channels_to_exclude"]
    epoch_definition = config['component_estimation']['epoch_definition']
    set_annotations = True

    # Load raw EEG
    raw = prepare_eeg(
        config,
        BIDS,
        preload=True,
        channels_to_include=channels_to_include,
        channels_to_exclude=channels_to_exclude,
        notch_filter=True,
        freq_limits=freq_limits,
        resample_frequency=resample_frequency,
        metadata_badchannels=True,
        interpolate_badchannels=True,
        set_annotations=set_annotations,
        crop_seconds=crop_seconds,
        rereference='average',
        epoch_definition=epoch_definition
    )

    # Read the ICA information
    sobi_dict = {
        'desc': 'sobi',
        'components_to_include': [],
        'components_to_exclude': []
    }
    sobi = read_sobi(config, BIDS, raw, sobi_dict['desc'])
    ICs_time_series = sobi.get_sources(raw)

    # Get the common figure
    fig, axes = plt.subplots(2, 4,
                             figsize=(16, 8),
                             constrained_layout=True)
    axes = axes.flatten()

    # Plot IC classification
    IClabel_componets = ['brain', 'muscle', 'eog', 'ecg', 'line_noise',
                         'ch_noise', 'other']

    # Plots
    for current_category, current_ax in zip(IClabel_componets, axes):

        # Print the values
        component_index = sobi.labels_[current_category]

        if len(component_index) > 0:
            current_IC_time_series = ICs_time_series.copy().pick(component_index)

            # Estimate the power
            spectrum = current_IC_time_series.compute_psd(
                method='welch',
                fmin=2,
                fmax=45,
                picks='all'
            )
            spectrum = spectrum.average()

            # Plot
            spectrum.plot(
                dB=False,
                amplitude=True,
                picks='all',
                axes=current_ax,
                show=False)

        # Add info
        current_ax.set_title(current_category)

    # Save the figure
    process = 'check'
    tail = 'check_components'
    figure_path = build_derivatives_path(BIDS, process, tail)

    if not os.path.exists(figure_path.parent):
        os.makedirs(figure_path.parent)

    plt.savefig(figure_path)
    plt.close()


def plot_occipital_power_spectrum(config, BIDS):

    # Load the clean EEG
    sobi = {
        'desc': 'sobi',
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
        freq_limits=freq_limits,
        notch_filter=True,
        resample_frequency=resample_frequency,
        set_annotations=True,
        crop_seconds=crop_seconds,
        rereference='average'
    )

    epochs = prepare_eeg(
        config,
        BIDS,
        raw=raw,
        apply_sobi=sobi,
        freq_limits=[2, 45],
        metadata_badchannels=True,
        exclude_badchannels=True,
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

    # Load the clean EEG
    epoch_definition = config['source_reconstruction']['epoch_definition']
    crop_seconds = config['component_estimation']['crop_seconds']

    # Load the clean data
    epochs = prepare_eeg(
        config,
        BIDS,
        preload=True,
        crop_seconds=crop_seconds,
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
    color_dict = {
        'good': 'white',
        'bad_muscle': 'tomato',
        'bad_jump': 'orange',
        'bad_other': 'gold',
        'bad_EOG': 'magenta'
    }
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
    legend_order = ['good', 'bad_muscle', 'bad_jump', 'bad_other', 'bad_EOG']
    legend_elements = []

    for key in legend_order:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='s',
                color='w',
                label=key,
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
    row_order = ['good', 'bad_muscle', 'bad_jump', 'bad_other', 'bad_EOG']

    # Crear datos para la tabla
    table_data = []
    row_colors = []

    for key in row_order:
        n = counts.get(key, 0)
        table_data.append([key, str(n)])
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
