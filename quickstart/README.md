# sEEGnal quickstart

This directory contains the recommended reproducible starting point for
sEEGnal. It runs the supported core preprocessing workflow without requiring
users to work with its individual Python modules. General installation and API
documentation belongs in the main sEEGnal README or project Wiki.

The quickstart processes continuous BrainVision recordings declared by the
user in `data/recordings.tsv`. It validates the complete input manifest and
the selected pipeline stages before writing anything.

## What the quickstart runs

The default workflow is executed by one launcher and comprises three stages:

1. **Standardization** converts every declared BrainVision recording to BIDS.
2. **Bad-channel detection** applies the configured complementary detectors.
3. **Artifact detection** generates artifact annotations and the cleaning ICA.

Source reconstruction and feature extraction remain available as experimental
stages. They are disabled by default and must be enabled explicitly in
`init/config.json`.

Each stage reports `ok` or `error`. When `global.verbose` is set to `full`,
error details are also printed.

## Requirements

- A Python environment with sEEGnal and its dependencies installed.
- A local copy of this repository.
- An internet connection is not required by the default core workflow. An
  experimental source-reconstruction run may download the `fsaverage`
  template if it is not already available locally.

See the repository's main README for installation instructions.

## Running the quickstart

Open a terminal in the repository root and execute:

```console
python -m quickstart.run_sEEGnal
```

The same file can be run from an IDE. It does not require command-line
arguments, and its data paths do not depend on the IDE's working directory.

The public stage selection is stored in `init/config.json`:

```json
"pipeline": {
    "standardize": true,
    "badchannel_detection": true,
    "artifact_detection": true,
    "source_reconstruction": false,
    "feature_extraction": false
}
```

Set a stage to `true` to enable it and `false` to disable it. Keys may be
omitted and are then treated as disabled. Only the five names shown above are
accepted, values must be boolean, and at least one stage must be enabled. The
quickstart reports an error before processing if the selection is invalid.

Stages always run in the order shown. Source reconstruction runs both the
forward model and inverse solution. An advanced user may select a later stage
when its required outputs already exist from an earlier execution. If a stage
fails, the remaining selected stages are skipped for that recording and the
quickstart continues with the next recording.

## Plotting processed results

After the three core stages finish successfully, the cleaned EEG can be opened
interactively with:

```console
python -m quickstart.plot_clean_eeg
```

The example loads the detected bad channels and artifact annotations, applies
the cleaning ICA, and displays the resulting MNE `Raw` object. It does not save
new results. Advanced users can adapt `quickstart/plot_clean_eeg.py` to work
with the returned `raw` object in their own MNE analyses. Close the current
window to continue to the next recording.

## Configuration

All quickstart parameters are defined in [`init/config.json`](init/config.json).
The scripts do not accept command-line configuration arguments.

By default, input data and generated results are stored in `quickstart/data`:

```json
"path": {
    "data_root": "data"
}
```

A relative `data_root` is resolved from the `quickstart` directory,
independently of the current working directory. An absolute path can be used to
store the data elsewhere, for example on Windows:

```json
"path": {
    "data_root": "D:/Data/seegnal-quickstart"
}
```

or on Linux and macOS:

```json
"path": {
    "data_root": "/home/user/data/seegnal-quickstart"
}
```

### Using your own recordings

Place every original EEG file directly in:

```text
<data_root>/sourcedata/eeg/
```

The accepted source format is:

- BrainVision `.vhdr`, including the data and marker files referenced by its
  header. Relative companion paths may use `/` or `\`, but must remain inside
  `sourcedata`.

Then add one row per recording to `<data_root>/recordings.tsv`:

```text
file	subject	session	task	run
recording.vhdr	subject01	session01	rest	01
recording_repeat.vhdr	subject01	session01	rest	02
```

The distributed `data/recordings.tsv` already contains the required header.
Edit it with a spreadsheet application or text editor and save it as a
tab-separated UTF-8 file.

Every column is required:

- `file` is the source filename only, without a directory;
- `subject`, `session`, `task`, and `run` define the output BIDS identity.

The five column names must be written exactly as shown. Missing, duplicated or
additional columns are validation errors. Subject, session and task values use
ASCII letters and numbers; run is a non-negative integer. All values are read
as text, so leading zeros are preserved.

The source name does not need to contain any BIDS identifier. This makes it
possible to retain acquisition-system filenames while assigning clear BIDS
names during standardization.

Only rows listed in `recordings.tsv` are processed. Other files in
`sourcedata/eeg` are ignored. Before processing begins, the quickstart checks
every row, source file, associated file, BIDS identifier, repeated source and
output destination. If any problem is found, all detected errors are reported
together and no recording is processed. An absent or header-only manifest is
an error.

Two different source files must not produce the same BIDS identity. For
example, these rows are invalid because they have identical subject, session,
task and run values:

```text
recording_a.vhdr	subject01	session01	rest	01
recording_b.vhdr	subject01	session01	rest	01
```

### Configuration reference

This reference describes the configuration used by the quickstart. It is not
yet a general configuration specification for the complete sEEGnal package.

Unless stated otherwise:

- Frequencies and bandwidths are expressed in hertz (Hz), durations in seconds,
  EEG amplitudes in volts (V), and distances in metres (m).
- Except for the boolean `pipeline` settings, numeric switches use `0` for
  false (disabled) and `1` for true (enabled).
- A frequency range is written as `[low, high]`. Frequencies at or above the
  Nyquist frequency cannot be used; notch frequencies above that limit are
  ignored and upper filter limits are reduced when necessary.
- The code does not yet validate the complete configuration against a schema.
  Values should therefore respect the ranges and relationships described here.

#### Paths and input manifest

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `path.data_root` | path, `"data"` | Root directory for input data, BIDS data and derivatives. Relative paths are resolved from `quickstart`; absolute paths are also accepted. |

The source directory and manifest are derived automatically as
`<data_root>/sourcedata/eeg` and `<data_root>/recordings.tsv`. Input selection
is not configured through filename templates or regular expressions.

#### Pipeline selection

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `pipeline.standardize` | boolean, `true` | Convert the original BrainVision recording to BIDS. |
| `pipeline.badchannel_detection` | boolean, `true` | Run automatic bad-channel detection. |
| `pipeline.artifact_detection` | boolean, `true` | Run automatic artifact detection and estimate the cleaning ICA. |
| `pipeline.source_reconstruction` | boolean, `false` | Run the experimental forward model and inverse solution. |
| `pipeline.feature_extraction` | boolean, `false` | Run the experimental sensor- and source-space features present in the configuration. |

Only these names are accepted. Omitted stages are disabled, and at least one
stage must be enabled.

#### Global processing settings

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `global.channels_to_include` | MNE channel selector, `"eeg"` | Channels retained for processing. The default includes every channel typed as EEG and automatically excludes correctly typed EOG, ECG, EMG, stimulus and miscellaneous channels. It also accepts the other selectors supported by `raw.pick()`. |
| `global.channels_to_exclude` | list of channel names; default EOG and mastoid names | Exact channel names removed after inclusion. This is a second safeguard for non-EEG electrodes that the source file labels as EEG. Add or remove names as needed; names absent from a recording are ignored. |
| `global.line_freq` | number in Hz, `50` | Power-line frequency written to the MNE/BIDS recording metadata. |
| `global.notch_frequencies` | list in Hz, `[50, 100, 150, 200]` | Frequencies removed when a processing stage requests notch filtering. Only values below the Nyquist frequency after resampling are applied. |
| `global.verbose` | string, `"full"` | `"full"` prints the details associated with a failed stage. Any other value suppresses those details; the `ok` or `error` stage status is still shown. |

#### Epoch definitions

The quickstart currently uses fixed-length epochs. Their parameters follow
`mne.make_fixed_length_epochs()`:

| Setting | Type and range | Meaning |
| --- | --- | --- |
| `mode` | `"fixed"` | Selects fixed-length epoching. |
| `duration` | seconds, greater than `0` | Complete duration of every epoch. A duration of 4 seconds at 500 Hz contains 2000 samples. |
| `overlap` | seconds, from `0` to less than `duration` | Time shared by consecutive epochs. Their step is `duration - overlap`. The sensor-artifact detector always forces this value to `0`. |
| `reject_by_annotation` | boolean | When `true`, MNE rejects epochs that overlap annotations whose description begins with `bad`. |

Event-based definitions follow `mne.Epochs()`. They use `mode="events"`, an
sEEGnal `event_source` of `"annotations"` or `"stim_channel"`, and MNE
parameters such as `event_id`, `tmin`, `tmax`, `baseline`,
`reject_by_annotation`, and `event_repeated`. As in MNE, both `tmin` and
`tmax` are inclusive, so a window from 0 to 1 second at 500 Hz contains 501
samples.

#### Component estimation

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `component_estimation.low_freq` | Hz, `1` | Lower filter limit used to estimate ICA components. |
| `component_estimation.high_freq` | Hz, `100` | Upper filter limit used to estimate ICA components. |
| `component_estimation.resample_frequency` | Hz, `500` | Sampling frequency used by the processing stages. |
| `component_estimation.crop_seconds` | seconds, `5` | Amount removed from both the beginning and the end of a recording before component estimation and most downstream analyses. Set to `0` to disable cropping. |
| `component_estimation.unclear_threshold` | probability from `0` to `1`, `0.7` | Minimum maximum ICLabel probability required for a confident classification. A component below this threshold is reassigned to `other`. |

The bad-channel, preliminary artifact, and final cleaning ICA decompositions
are fitted on continuous `Raw` data. Samples covered by annotations whose
description starts with `BAD` are omitted from both ICA fitting and ICLabel
classification. Only the final cleaning ICA loads the artifact annotations
created by the preliminary detection pass.

#### Standardization

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `preprocess.standardization.format` | string, `"BrainVision"` | Output format passed to MNE-BIDS. `BrainVision` is currently the only supported and tested output option in the quickstart. |
| `preprocess.standardization.overwrite` | `0` or `1`, `1` | When `1`, existing standardized BIDS files may be replaced. When `0`, MNE-BIDS prevents overwriting them. |

#### Bad-channel detection

`preprocess.badchannel_detection.crop_seconds` defaults to `10` seconds and
removes that duration from both ends of the recording for bad-channel
detection. The individual detectors are configured as follows:

| Detector and setting | Type and default | Meaning |
| --- | --- | --- |
| `impossible_amplitude.low_freq` / `high_freq` | Hz, `2` / `150` | Filter limits used before measuring each channel and epoch. |
| `impossible_amplitude.low_threshold` / `high_threshold` | V, `0.000001` / `0.0005` | Lower and upper limits for the standard deviation of an epoch. |
| `impossible_amplitude.percentage_threshold` | fraction from `0` to `1`, `0.5` | A channel is marked bad when the fraction of epochs outside the amplitude limits is greater than this value. This is a fraction, not a value from 0 to 100. |
| `impossible_amplitude.epoch_definition` | epoch definition | Segmentation used by the detector. |
| `component_detection.low_freq` / `high_freq` | Hz, `2` / `45` | Filter limits applied after retaining the ICA components used by this detector. |
| `component_detection.threshold` | MAD multiplier, `5` | Maximum accepted robust deviation of an epoch/channel standard deviation from the global median. |
| `component_detection.percentage_threshold` | fraction from `0` to `1`, `0.5` | Minimum fraction of deviant epochs required to mark a channel bad. |
| `component_detection.epoch_definition` | epoch definition | Segmentation used by the detector. |
| `gel_bridge.low_freq` / `high_freq` | Hz, `2` / `45` | Filter limits applied before estimating pairwise channel correlations. |
| `gel_bridge.threshold` | correlation coefficient, `0.999` | Minimum pairwise channel correlation considered a possible gel bridge. |
| `gel_bridge.neighbour_distance` | m, `0.05` | Maximum spatial distance allowed between a correlated pair. |
| `high_deviation.low_freq` / `high_freq` | Hz, `2` / `45` | Filter limits used before estimating channel deviation. |
| `high_deviation.threshold` | MAD multiplier, `5` | Maximum accepted robust deviation of an epoch/channel standard deviation from the global median. |
| `high_deviation.percentage_threshold` | fraction from `0` to `1`, `0.5` | Minimum fraction of deviant epochs required to mark a channel bad. |
| `high_deviation.epoch_definition` | epoch definition | Segmentation used by the detector. |

#### Artifact detection

All artifact `low_freq` and `high_freq` values are filter limits in Hz.

| Detector and setting | Type and default | Meaning |
| --- | --- | --- |
| `frontal_channels` | list of names, `["Fp1", "Fpz", "Fp2"]` | Preferred channels for EOG detection. Only names that remain in the prepared recording are used. If none is available, no EOG artifact is reported. |
| `EOG.low_freq` / `high_freq` | Hz, `0.5` / `2` | Filter limits used for ocular-artifact detection. |
| `EOG.threshold` | standard-deviation multiplier, `7` | Threshold applied to the absolute signal obtained by averaging the available frontal channels. |
| `muscle.low_freq` / `high_freq` | Hz, `110` / `145` | Filter limits used for muscle-artifact detection. |
| `muscle.threshold` | standard-deviation multiplier, `7` | Peak threshold applied to the demeaned time series of variability across channels. |
| `sensor.low_freq` / `high_freq` | Hz, `5` / `10` | Filter limits used to detect abrupt sensor artifacts. |
| `sensor.threshold` | dimensionless multiplier, `7` | For every epoch, the maximum absolute amplitude is calculated for each channel. The mean of those channel maxima is multiplied by this value, and the epoch is marked when any channel exceeds the resulting threshold. |
| `sensor.epoch_definition` | epoch definition | Segmentation used by the sensor detector. Its overlap is forced to zero and epochs overlapping preceding muscle annotations are rejected. |
| `other.low_freq` / `high_freq` | Hz, `2` / `45` | Filter limits used for remaining high-amplitude artifacts. |
| `other.threshold` | V, `0.0005` | Maximum absolute sample amplitude after removing each epoch/channel mean. An epoch is marked if any sample exceeds it. |
| `other.epoch_definition` | epoch definition | Segmentation used by the detector. Epochs overlapping preceding muscle annotations are rejected. |

#### Source reconstruction

The values included in the distributed configuration are the combinations
tested for this quickstart. MNE provides additional options for several of these
settings, but they have not yet been validated in sEEGnal.

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `source_reconstruction.epoch_definition` | epoch definition | Clean EEG segmentation used to estimate covariance and the inverse solution. |
| `forward.use_template` | `0` or `1`, `1` | Selects template-based reconstruction. `1` is currently the only functional option; an individual MRI workflow is not implemented. |
| `forward.template.subject` | string, `"fsaverage"` | FreeSurfer template subject. It is downloaded by MNE when necessary. |
| `forward.template.trans` | string, `"fsaverage"` | Head-to-MRI transform supplied to MNE. |
| `forward.template.bem` | filename, `"fsaverage-5120-5120-5120-bem-sol.fif"` | BEM solution inside the template subject's `bem` directory. |
| `forward.template.spacing` | MNE spacing string, `"ico2"` | Source-space density. `ico2` keeps the quickstart computationally small. |
| `covariance.method` | MNE method, `"oas"` | Covariance estimator. |
| `covariance.rank` | MNE rank setting, `"info"` | Rank used by covariance estimation. |
| `inverse.method` | string, `"lcmv"` | Inverse method. LCMV is currently the only implemented option. |
| `inverse.reg` | non-negative number, `0.05` | LCMV regularization parameter. |
| `inverse.pick_ori` | MNE option, `"max-power"` | Orientation selection passed to the LCMV beamformer. |
| `inverse.weight_norm` | MNE option, `"nai"` | Weight normalization passed to the LCMV beamformer. |

#### Feature extraction

Each feature contains an `epoch_definition`. A feature is calculated in sensor
space when its `sensor` object exists and in source space when its `source`
object exists. Remove one of those objects to disable that space; a source
feature also requires the source-reconstruction outputs.

Relative power is always calculated with the quickstart's multitaper implementation
and normalized relative to the total spectrum. Its remaining settings are:

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `relative_power_spectrum.overwrite` | `0` or `1`, `1` | Whether an existing relative-power output may be replaced. |
| `relative_power_spectrum.<space>.freq_limits` | closed interval in Hz, `[0.5, 50]` | Frequencies included in the spectrum for `sensor` or `source`. |
| `relative_power_spectrum.<space>.bandwidth` | Hz, `2` | Multitaper frequency smoothing bandwidth. |
| `relative_power_spectrum.<space>.adaptive` | `0` or `1`, `1` | Enables adaptive multitaper weighting when set to `1`. |

PLV and corrected imaginary PLV (`ciplv`) share the following structure:

| Setting | Type and default | Meaning |
| --- | --- | --- |
| `<feature>.freq_limits` | closed interval in Hz, `[0.5, 50]` | Initial filter range applied before calculating the feature. |
| `<feature>.<space>.freq_bands_name` | list of strings | Output name assigned to each frequency band. |
| `<feature>.<space>.freq_bands_limits` | list of closed `[low, high]` intervals in Hz | Filter limits corresponding positionally to `freq_bands_name`. |

`freq_bands_name` and `freq_bands_limits` must contain the same number of
items. Each band uses both stated boundaries, and the sensor and source lists
may be configured independently. The predefined and tested bands are delta
(`2–4` Hz), theta (`4–8` Hz), alpha (`8–12` Hz), low beta (`12–20` Hz), high
beta (`20–30` Hz), and gamma (`30–45` Hz).

Sensor-space feature files retain the canonical prepared channel order.
Relative-power rows and PLV or ciPLV connections involving an excluded bad
channel contain `NaN`; the HDF5 metadata identifies the good, bad, and NaN
indices. Source-space features exclude bad input channels before applying the
LCMV filter, require finite output values, and store the input-channel and
source-vertex mappings in the same derivative.

## Generated data and outputs

Depending on the enabled stages, the resulting structure is:

```text
quickstart/data/
├── recordings.tsv                  Input manifest
├── sourcedata/eeg/                 BrainVision originals
├── sub-<id>/                       Standardized BIDS recording
└── derivatives/sEEGnal/
    ├── check/                      Quality-control figures
    ├── preprocess/                 Bad-channel and artifact metadata
    ├── source_reconstruction/      Experimental; created when enabled
    └── feature_extraction/         Experimental; created when enabled
```

Generated quickstart data and results are ignored by Git.

Artifact annotation onsets in the preprocessing derivatives are relative to
the beginning of the original uncropped recording, even when detection was
performed on cropped data.

## Troubleshooting

- Run the terminal commands from the repository root and use the `python -m`
  form shown above. This ensures that the `quickstart` package can be imported.
- If validation fails, correct every row listed in the aggregated error report
  before rerunning the quickstart. Check `path.data_root`, `recordings.tsv`, the exact
  source filename and its BrainVision companion files.
- Keep each combination of `subject`, `session`, `task`, and `run` unique. The
  report identifies every pair of rows that would overwrite the same BIDS
  recording.
- If source reconstruction fails on its first run, check the internet
  connection and whether MNE can download or access `fsaverage`.
