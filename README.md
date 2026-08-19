# sEEGnal

sEEGnal is a fully automated and modular pipeline for EEG preprocessing. It is
designed to standardize processing across recordings while reducing the manual
work and variability associated with expert-driven workflows.

The core pipeline combines three stages:

1. Standardization according to the EEG extension of the
   [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/).
2. Bad-channel detection using complementary signal-based criteria.
3. Artifact identification using physiologically informed criteria,
   Extended Infomax independent component analysis (ICA), and ICLabel-based
   classification.

The first published version of sEEGnal was evaluated against expert-driven EEG
preprocessing. The package has evolved since that evaluation; see
[Publication and citation](#publication-and-citation) for details.

> [!IMPORTANT]
> sEEGnal is a signal-processing tool and is not intended for clinical or
> diagnostic use. It does not provide diagnoses, medical recommendations, or
> clinical interpretation. Users are responsible for validating their
> workflows and interpreting the generated outputs.

## Project status

sEEGnal is currently **Beta**. The automated core pipeline—BIDS
standardization, bad-channel detection, and artifact detection—is the primary
supported workflow. Its methods have been scientifically evaluated, while the
public packaging and user interface are still being validated with external
users.

Source reconstruction and feature extraction are available as experimental,
early-stage modules. Their interfaces and behavior may change more rapidly
than the core preprocessing workflow.

## Scope

The main purpose of sEEGnal is automated EEG preprocessing:

- conversion of source recordings to BIDS;
- automated bad-channel detection;
- automated artifact detection and annotation;
- generation of quality-control information and reproducible derivatives.

The repository also contains functional, evolving modules for source
reconstruction and feature extraction. These currently include template-based
LCMV source reconstruction, relative power spectra, phase-locking value (PLV),
and corrected imaginary PLV (ciPLV). They are additional capabilities rather
than part of the three-stage core pipeline evaluated in the original
publication.

## Supported inputs

sEEGnal's public input workflow currently accepts continuous EEG recordings
stored as:

- BrainVision files (`.vhdr` with the associated `.vmrk` and `.eeg` files).

sEEGnal can process recordings with events and use event information from BIDS
for event-based epoching.

## Installation

sEEGnal requires Python 3.11 or later. It is not yet distributed through PyPI;
install it from the public
[GitHub repository](https://github.com/FedeRamirezT/sEEGnal).

Using an isolated virtual environment is recommended.

### Install directly from GitHub

This is the shortest installation method and requires Git:

```console
python -m pip install "sEEGnal @ git+https://github.com/FedeRamirezT/sEEGnal.git@v4.0.1"
```

### Clone and install locally

Use this option when you want a local copy of the quickstart, configuration, and
source code:

```console
git clone https://github.com/FedeRamirezT/sEEGnal.git
cd sEEGnal
git checkout v4.0.1
python -m pip install .
```

For development, install the local repository in editable mode:

```console
python -m pip install -e .
```

`pyproject.toml` is the single source of truth for the package's runtime
dependencies, and `python -m pip install .` installs them automatically. For
the exact dependency versions tested with Python 3.13, clone the repository
and apply the optional constraints file during installation:

```console
python -m pip install -c constraints-py313.txt .
```

## Quick start

The included quickstart is the recommended starting point. It provides a
complete project structure, a tabular input manifest, a documented
configuration, the three-stage core preprocessing workflow, and an example
for opening the cleaned EEG with MNE.

After cloning and installing the repository, run from its root directory:

```console
python -m quickstart.run_sEEGnal
```

Add continuous BrainVision source files to `quickstart/data/sourcedata/eeg`,
describe each recording in `quickstart/data/recordings.tsv`, and then run the
command above.
The complete manifest is validated before any BIDS file is written; missing or
ambiguous inputs stop the run with an aggregated error report.

See the [quickstart documentation](quickstart/README.md) for input naming, configuration,
generated files, optional stages, and plotting examples.

The launcher loads the quickstart configuration and delegates to the public
Python entry point. The equivalent advanced use is:

```python
import quickstart.init.init as init
import sEEGnal

config = init.load_config()
sEEGnal.run_sEEGnal(config)
```

The distributed configuration enables standardization, bad-channel detection,
and artifact detection. Experimental source reconstruction and feature
extraction must be enabled explicitly.

## Outputs

Depending on the selected stages, sEEGnal creates:

- a standardized BIDS EEG dataset;
- bad-channel classifications and rejection reasons;
- ICA decompositions, ICLabel classifications, and probability scores;
- artifact annotations with onset, duration, and artifact type;
- quality-control figures and metadata;
- optional forward models and inverse solutions;
- optional sensor- and source-space feature files.

Derived outputs are organized under `derivatives/sEEGnal/` by processing
stage. The main processing stages also return result dictionaries that report
`ok` or `error` and provide execution details.

Artifact annotation onsets are stored relative to the beginning of the
original, uncropped recording. Loading them into a cropped Raw object preserves
that time base, discards intervals outside the crop, and trims intervals that
cross its boundaries.

Sensor-space feature derivatives preserve the canonical prepared channel
order. Rows or connections involving excluded bad channels contain `NaN`, and
their channel and output indices are recorded in the HDF5 metadata. Source-space
features exclude bad input sensors before reconstruction, require finite output
values, and record the input-channel and source-vertex mappings needed to
interpret each derivative.

### Effective EEG reference

`prepare_eeg()` accepts `rereference=False` to preserve the current reference
or `rereference='average'` to return data with the average EEG reference
already applied. The effective state can be inspected without accessing
sEEGnal's private MNE attributes:

```python
from sEEGnal.tools.mne_tools import get_reference_info

reference_info = get_reference_info(raw_or_epochs)
```

The function returns an independent dictionary, so changing it does not alter
the provenance stored on the `Raw` or `Epochs` object. This metadata is kept
in memory and propagated by sEEGnal when it creates new epochs or temporary
Raw objects. MNE does not automatically persist arbitrary Python attributes
when Raw or Epochs data are written; sEEGnal stores the corresponding ICA and
LCMV reference provenance in their JSON derivative sidecars.

### Applying ICA

ICA application is kept separate from general EEG preparation. First create a
preloaded Raw object with the reference and ordered EEG sensor space used to
fit the ICA, then apply the selected component policy, and finally prepare the
data required by the downstream analysis:

```python
from sEEGnal.tools.mne_tools import apply_ica, prepare_eeg

raw = prepare_eeg(
    config,
    bids_path,
    preload=True,
    metadata_badchannels=True,
    rereference='average',
)
raw = apply_ica(config, bids_path, raw, ica_config)
epochs = prepare_eeg(
    config,
    bids_path,
    raw=raw,
    exclude_badchannels=True,
    epoch_definition=epoch_definition,
)
```

`apply_ica()` modifies and returns the same Raw object. It rejects incompatible
references, missing or reordered fitted channels, fitted channels marked bad,
and additional good EEG channels that would otherwise remain uncleaned.

## Compatibility

- Required Python version: 3.11 or later.
- Formally tested environment: Python 3.13 on Windows.
- Tested Linux environment: Ubuntu 24.04 under WSL2, using a command-line
  virtual environment.
- Other supported Python versions and operating systems have not yet been
  tested.
- Supported public source format: continuous BrainVision.

The dependency ranges supported by the package are declared in
`pyproject.toml`. `constraints-py313.txt` records the exact package versions
used for the tested Python 3.13 environment.

## Documentation

The full documentation is under construction. It will provide detailed guides
for:

- configuration;
- BIDS standardization;
- bad-channel and artifact detection;
- ICA and ICLabel processing;
- source reconstruction;
- feature extraction;
- the Python API;
- troubleshooting and validation.

Until then, the [quickstart documentation](quickstart/README.md) provides the most complete
working example and configuration reference.

## Publication and citation

The original sEEGnal pipeline was evaluated in:

> Ramírez-Toraño F, Hatlestad-Hall C, Drews A, Renvall H, Rossini PM, Marra C,
> Haraldsen IH, Maestu F, Bruña R. *sEEGnal: an automated EEG preprocessing
> pipeline evaluated against expert-driven preprocessing*. Computers in Biology
> and Medicine. 2026;213:111837.
> [https://doi.org/10.1016/j.compbiomed.2026.111837](https://doi.org/10.1016/j.compbiomed.2026.111837)

The publication describes an earlier version of the pipeline. When reporting
work performed with the current package, cite the article and record the
sEEGnal release or Git commit used in the analysis.

## Maintainer and support

- Lead developer and maintainer:
  [Federico Ramírez-Toraño](https://github.com/FedeRamirezT)
- Original pipeline contributor:
  [Ricardo Bruña](https://github.com/rbruna)

Use [GitHub Issues](https://github.com/FedeRamirezT/sEEGnal/issues) to report
bugs, request features, or ask questions about the public package.

## License

sEEGnal is distributed under the [BSD 3-Clause License](LICENSE).
