# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.1] - 2026-08-19

### Added

- Added the BSD 3-Clause license and contribution terms.
- Added atomic validation of the public `recordings.tsv` manifest, including
  BrainVision companion files and duplicate BIDS destinations.
- Added the public `sEEGnal.run_sEEGnal(config)` pipeline entry point.

### Changed

- Replaced configurable filename-pattern discovery with an explicit
  five-column recording manifest and added the BIDS `run` entity.
- Renamed the public `demo` folder to `quickstart` and moved manifest parsing
  out of BIDS standardization into `sEEGnal.io.recordings`.
- Moved the public runner and cleaned-EEG plot to the root of `quickstart`.
- Made the three core preprocessing stages the default workflow and moved
  stage selection to boolean options in `config.json`.
- Changed the default channel selector to include every typed EEG channel,
  followed by the configurable name-based exclusion list.

### Fixed

- Made the public test suite independent of the private `dev` configuration.

### Removed

- Removed automatic quickstart-data generation and its bundled local outputs.
- Removed active EEGLAB input support until it can be validated with real
  recordings.
- Removed the public forward-model and power-spectrum plotting examples.

## [4.0.0] - 2026-08-18

### Breaking changes

- Removed ICA application from `prepare_eeg()`. Callers must now prepare a
  compatible continuous Raw object, call `apply_ica()`, and call
  `prepare_eeg()` again for downstream filtering, bad-channel handling or
  epoching.
- Replaced the legacy epoch keys `length`, `padding`, and `event_code` with an
  explicit `mode` and the corresponding MNE-Python arguments. Fixed epochs use
  `duration`; event epochs use `event_id`, inclusive `tmin` and `tmax` bounds,
  and an `event_source` of `annotations` or `stim_channel`.
- Renamed `read_badchannels()` to `read_channels()` to reflect that it returns
  the complete derivative channel-status table.
- Changed sensor-space feature derivatives to preserve the canonical prepared
  channel order. Values and connections involving excluded bad channels are
  now represented as `NaN` and identified in the saved metadata.
- Changed `create_annotations()` and `merge_peaks()` to use the original
  recording sample count and half-open interval boundaries. Their former
  `last_sample` arguments are no longer accepted.
- Required ICA and LCMV derivatives to carry effective-reference provenance.
  Derivatives created without compatible provenance must be regenerated.

### Added

- Added `apply_ica()` for category-based component removal from compatible
  continuous EEG data.
- Added `get_reference_info()` to inspect effective-reference provenance and
  `get_unannotated_raw()` to materialize the samples not covered by `BAD`
  annotations.
- Added effective-reference and ordered-sensor-space validation before ICA or
  LCMV application, with reference provenance stored in derivative sidecars.
- Added channel, missing-value, sensor-input, and source-vertex mappings to
  relative-power, PLV, and ciPLV derivative metadata.
- Added configurable MNE handling for simultaneous events through
  `event_repeated`.
- Added regression coverage for epoch construction, EEG preparation,
  reference handling, annotation round trips, ICA derivatives, and sensor- and
  source-space bad-channel policies.

### Changed

- Aligned fixed and event-based epoch construction with the corresponding
  MNE-Python APIs, including MNE-owned event repetition, boundary rejection,
  overlap, rounding, and drop-log semantics.
- Fitted the bad-channel, preliminary artifact, and final cleaning ICA
  decompositions on continuous Raw data. Samples covered by `BAD` annotations
  are omitted consistently from ICA fitting and ICLabel classification.
- Excluded classified bad channels from artifact detection, inverse estimation,
  and source-space feature computation instead of interpolating them.
- Made sensor-space feature outputs retain their canonical layout while
  source-space outputs require finite values and record the exact input-channel
  and source-vertex mappings.
- Made artifact annotations use the original uncropped recording time axis and
  made later artifact detectors reject epochs already covered by preceding
  muscle annotations.
- Made `prepare_eeg()` validate arguments before processing, preserve existing
  electrode coordinates, use MNE's resampling anti-aliasing, and apply
  bad-channel policies in a deterministic order.
- Made an explicitly requested average reference effective immediately and
  recomputed it when the participating EEG sensor space changes.
- Made quality-control power spectra use one Welch segment length supported by
  every uninterrupted span outside `BAD` annotations.
- Consolidated the developer launchers and updated the demo configuration,
  processing scripts, plots, and documentation for the new preparation flow.

### Fixed

- Preserved exact original sample bounds through repeated crops and converted
  artifact detections consistently between local, absolute MNE, and original
  recording coordinates.
- Corrected artifact interval merging at recording boundaries and prevented
  duplicate or out-of-window annotations during saved-annotation round trips.
- Validated the artifact TSV schema before reading its columns and documented
  annotation onset, duration, and label fields in the JSON sidecar.
- Preserved reference provenance when creating Epochs and temporary Raw
  objects and when saving or loading ICA and LCMV derivatives.
- Prevented source filters from being applied to data with an incompatible
  reference or sensor space.

## [3.0.0] - 2026-08-06

### Breaking changes

- Replaced SOBI with MNE-Python Extended Infomax ICA throughout the preprocessing pipeline.
- Replaced `write_sobi()` and `read_sobi()` with `write_ica()` and `read_ica()`.
- Replaced `prepare_eeg(..., apply_sobi=...)` with `prepare_eeg(..., apply_ica=...)`.
- Renamed `mne_tools.ICA()` to `mne_tools.fit_ica()`.
- Changed ICA derivative descriptions to `artifacts`, `cleaning`, and `badchannels`.
- Removed support for legacy SOBI derivatives; existing decompositions and sidecars must be regenerated.

### Added

- Added a reproducible end-to-end demo for preprocessing, source reconstruction, and feature extraction.
- Added demo plots for cleaned EEG, forward models, and power spectra.
- Added validation for saving, reloading, and applying rank-reduced ICA decompositions.
- Added Python 3.13 dependency constraints for the formally tested environment.
- Added citation metadata, contribution guidelines, and expanded package and demo documentation.

### Changed

- ICA decompositions are now stored as native MNE-Python FIF objects, with ICLabel probabilities in TSV/JSON sidecars.
- ICA fitting now uses the rank determined by MNE-Python, supporting rank-reduced data.
- Broadened Python support from Python 3.13.7 only to Python 3.11 or later.
- Replaced fully pinned requirements with bounded direct dependencies in `pyproject.toml`.
- Updated preprocessing, quality-control, feature-extraction, and source-reconstruction workflows to use ICA terminology and derivatives.
- Consolidated the demo launchers and made demo paths independent of the current working directory.
- Updated the demo to process all matching user-provided BrainVision recordings.

### Removed

- Removed the custom `auxsobi` and `raweep` C extensions and their build configuration.
- Removed unused signal-processing, EEP-reading, and SOBI modules.
- Removed manual construction of MNE ICA objects from mixing and unmixing matrices.
- Removed obsolete demo launchers and validation scripts superseded by the end-to-end workflow.
- Removed the obsolete `requirements.txt` installation path.

### Fixed

- Reduced PLV memory use and prevented ciPLV NaNs caused by floating-point rounding.
- Allowed event-based processing to use recording annotations when no BIDS event file is available.
- Made notch filtering and graphical backends optional when they are not configured or required.
- Corrected handling and return types for continuous and epoched MNE data.
- Applied configured frequency limits correctly in source-space power-spectrum estimation.
- Preserved ICLabel probabilities when ICA derivatives are saved and reloaded.
- Used absolute sample indices when creating annotations from cropped recordings.
- Made EOG detection use the frontal channels available after preprocessing.
- Normalized mathematically real LCMV filters stored with complex dtypes and rejected genuinely complex inputs before PSD, PLV, and ciPLV estimation.

## [2.1.2] - 2026-04-15

### Added

- Added event-based epoching alongside fixed-length epoching.

### Fixed

- Corrected annotation handling during task-related preprocessing.

## [2.1.1] - 2026-04-08

### Added

- Added quality-control plots for bad-channel positions, occipital power, component power, and forward models.
- Added a quality-control summary of the number of clean epochs.

### Changed

- Integrated quality-control generation into the corresponding processing stages.

## [2.1.0] - 2026-04-06

### Added

- Added relative power, PLV, and ciPLV estimation in sensor and source space.
- Added resting-state-network atlases and feature-extraction validation scripts.
- Added plotting from saved power-spectrum derivatives.

### Changed

- Changed source reconstruction from volumetric models to fsaverage surfaces provided by MNE-Python.
- Standardized PLV and ciPLV naming, storage, and reading functions.
- Renamed the internal test folder to `dev`.

### Removed

- Removed the large bundled fsaverage files in favor of the MNE-Python dataset.

### Fixed

- Corrected PLV and ciPLV calculations and derivative-folder creation.
- Prevented bad-channel derivatives from being overwritten.
- Corrected package imports in the demo scripts.

## [2.0.0] - 2026-02-24

### Breaking changes

- Reorganized the BIDS derivatives layout for preprocessing, source reconstruction, and features.
- Restructured the configuration file to support source and feature settings.

### Added

- Added template-based forward and inverse source reconstruction with saved LCMV filters.
- Added atlas support and source-localization validation plots.
- Added occipital power-spectrum and source-position visualizations.

### Changed

- Stored forward solutions, inverse filters, and derived results as reusable files.

### Fixed

- Included the required atlas data in the installed package.

## [1.3.2] - 2026-02-04

### Added

- Added `other_annotations` to the first artifact-detection pass.
- Added the `reject_by_annotation` option when creating epochs.
- Added parameter-tuning scripts and plots for bad channels, EOG, muscle, and sensor artifacts.

### Changed

- Retuned bad-channel, artifact-detection, and SOBI parameters.

## [1.3.1] - 2026-01-19

### Added

- Added the option to write bad-channel information to recording metadata.

### Changed

- Estimated median and MAD across the complete data matrix.
- Simplified epoch definition to use length and overlap in seconds.

### Fixed

- Corrected artifact indices and selection of the appropriate SOBI decomposition.
- Prevented failures when every channel is marked as bad.

## [1.3.0] - 2026-01-15

### Added

- Added support for applying new processing parameters to an already loaded EEG recording.
- Added clean-data plots to the artifact-detection workflow.

### Changed

- Fitted artifact-detection SOBI decompositions on epochs.
- Used robust median and MAD statistics for bad-channel, EOG, and sensor-artifact detection.
- Based muscle detection on the absolute component amplitude.

### Fixed

- Excluded previously detected artifacts before fitting the second SOBI decomposition.

## [1.2.0] - 2026-01-09

### Changed

- Renamed `prepare_raw()` to `prepare_eeg()` and reordered the EEG-loading operations.
- Added source-file loading and optional SOBI application to `prepare_eeg()`.

### Removed

- Removed impedance as a bad-channel criterion.

## [1.1.1] - 2025-12-04

### Changed

- Renamed `prepare_raw()` to `prepare_eeg()` and moved bad-channel removal into that function.
- Replaced Welch power with variance for power-based bad-channel detection.
- Applied YAPF formatting and reorganized the validation folders.

### Fixed

- Corrected BIDS-path creation after its function signature changed.
- Removed an unnecessary demeaning step from bad-channel and artifact detection.

## [1.1.0] - 2025-11-12

### Added

- Added average and median EEG rereferencing.
- Added reusable demo scripts.

### Changed

- Made installation and execution independent of the development machine.
- Excluded previously marked bad channels from subsequent detection criteria.
- Interpolated bad channels before artifact detection.
- Limited standardized recordings to the channels selected in the configuration.

### Fixed

- Ensured detected bad channels are always written to metadata.

## [1.0.0] - 2025-10-30

### Added

- Added the first runnable release of the sEEGnal preprocessing pipeline.
- Added package-building, validation, and demo infrastructure.

[Unreleased]: https://github.com/FedeRamirezT/sEEGnal/compare/v4.0.1...HEAD
[4.0.1]: https://github.com/FedeRamirezT/sEEGnal/tree/v4.0.1
[4.0.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v4.0.0
[3.0.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v3.0.0
[2.1.2]: https://github.com/FedeRamirezT/sEEGnal/tree/v2.1.2
[2.1.1]: https://github.com/FedeRamirezT/sEEGnal/tree/v2.1.1
[2.1.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v2.1.0
[2.0.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v2.0.0
[1.3.2]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.3.2
[1.3.1]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.3.1
[1.3.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.3.0
[1.2.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.2.0
[1.1.1]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.1.1
[1.1.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.1.0
[1.0.0]: https://github.com/FedeRamirezT/sEEGnal/tree/v1.0.0
