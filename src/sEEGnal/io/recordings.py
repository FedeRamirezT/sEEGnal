"""Read and validate the public EEG recordings manifest.

The manifest is validated completely before a caller starts processing. This
module does not write BIDS data or alter source recordings.
"""

import csv
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from sEEGnal.tools.bids_tools import build_BIDS_object


RECORDING_COLUMNS = ('file', 'subject', 'session', 'task', 'run')
SUPPORTED_SOURCE_EXTENSIONS = ('.vhdr',)


class RecordingsValidationError(ValueError):
    """Report every problem found in a recordings manifest."""


def _recordings_error(manifest_path, errors):
    """Create one readable exception containing all validation errors."""

    count = len(errors)
    noun = 'error' if count == 1 else 'errors'
    details = '\n'.join(f'- {error}' for error in errors)
    return RecordingsValidationError(
        f'{manifest_path.name} contains {count} {noun}:\n{details}'
    )


def _read_brainvision_references(header_path):
    """Read DataFile and MarkerFile from a BrainVision header."""

    contents = header_path.read_bytes()
    try:
        text = contents.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = contents.decode('cp1252')

    current_section = None
    references = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip().casefold()
            continue
        if current_section == 'common infos' and '=' in line:
            key, value = line.split('=', 1)
            references[key.strip().casefold()] = value.strip().strip('"')

    return references


def _resolve_source_reference(source_root, parent, reference):
    """Resolve a portable relative reference without leaving source_root."""

    if (
        PurePosixPath(reference).is_absolute()
        or PureWindowsPath(reference).is_absolute()
    ):
        raise ValueError('must be relative to sourcedata')

    portable_reference = reference.replace('\\', '/')
    reference_path = PurePosixPath(portable_reference)
    if any(part == '..' for part in reference_path.parts):
        raise ValueError('must not escape sourcedata')

    source_root = source_root.resolve()
    resolved = parent.joinpath(*reference_path.parts).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError as error:
        raise ValueError('must not escape sourcedata') from error

    return resolved


def _validate_brainvision_file(header_path, source_root):
    """Return structural errors for a BrainVision recording."""

    errors = []
    try:
        references = _read_brainvision_references(header_path)
    except (OSError, UnicodeError) as error:
        return [f'cannot read BrainVision header {header_path.name!r}: {error}']

    expected_references = (
        ('datafile', 'DataFile', '.eeg'),
        ('markerfile', 'MarkerFile', '.vmrk'),
    )
    for field, label, expected_extension in expected_references:
        referenced_name = references.get(field)
        if not referenced_name:
            errors.append(
                f'{header_path.name!r} has no {label} entry in [Common Infos]'
            )
            continue
        if PureWindowsPath(referenced_name).suffix.casefold() != expected_extension:
            errors.append(
                f'{header_path.name!r} {label} must reference a '
                f'{expected_extension} file: {referenced_name!r}'
            )
            continue
        try:
            referenced_path = _resolve_source_reference(
                source_root,
                header_path.parent,
                referenced_name,
            )
        except ValueError as error:
            errors.append(
                f'{header_path.name!r} has unsafe {label} reference '
                f'{referenced_name!r}: {error}'
            )
            continue
        if not referenced_path.is_file():
            errors.append(
                f'{header_path.name!r} references missing {label} '
                f'{referenced_name!r}'
            )

    return errors


def _read_recording_rows(manifest_path):
    """Read the TSV as strings and return rows plus schema errors."""

    errors = []
    rows = []
    try:
        file = manifest_path.open('r', encoding='utf-8-sig', newline='')
    except OSError as error:
        return [], [f'cannot read the manifest: {error}']

    with file:
        reader = csv.DictReader(file, delimiter='\t')
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return [], ['the manifest has no header row']

        duplicated_columns = sorted({
            column for column in fieldnames if fieldnames.count(column) > 1
        })
        missing_columns = [
            column for column in RECORDING_COLUMNS
            if column not in fieldnames
        ]
        unexpected_columns = [
            column for column in fieldnames
            if column not in RECORDING_COLUMNS
        ]
        if duplicated_columns:
            errors.append(f'duplicated columns: {duplicated_columns}')
        if missing_columns:
            errors.append(f'missing required columns: {missing_columns}')
        if unexpected_columns:
            errors.append(f'unexpected columns: {unexpected_columns}')

        for row_number, source_row in enumerate(reader, start=2):
            if None in source_row:
                errors.append(
                    f'row {row_number}: has more values than header columns'
                )

            record = {'row': row_number}
            row_is_complete = True
            for column in RECORDING_COLUMNS:
                value = source_row.get(column)
                if value is None or not value.strip():
                    errors.append(
                        f'row {row_number}: missing value for {column!r}'
                    )
                    row_is_complete = False
                else:
                    record[column] = value.strip()
            if row_is_complete:
                rows.append(record)

    if not rows and not any('row ' in error for error in errors):
        errors.append('the manifest contains no recordings')

    return rows, errors


def _is_plain_filename(file_name):
    """Return whether file_name contains no directory or absolute path."""

    return not (
        file_name in ('.', '..')
        or '/' in file_name
        or '\\' in file_name
        or PurePosixPath(file_name).is_absolute()
        or PureWindowsPath(file_name).is_absolute()
        or bool(PureWindowsPath(file_name).drive)
        or Path(file_name).name != file_name
    )


def _validate_bids_identifiers(record):
    """Return public-manifest BIDS identifier errors for one row."""

    errors = []
    invalid_labels = [
        field for field in ('subject', 'session', 'task')
        if re.fullmatch(r'[A-Za-z0-9]+', record[field]) is None
    ]
    if invalid_labels:
        errors.append(
            f'{invalid_labels} must contain ASCII letters and numbers only'
        )
    if re.fullmatch(r'[0-9]+', record['run']) is None:
        errors.append('run must be a non-negative integer')
    return errors


def validate_recordings(config, manifest_path=None):
    """Validate every input row before any BIDS file is written.

    All five public manifest fields are mandatory. Returned dictionaries
    include the source filename and a prospective, validated ``bids_path``.
    """

    if manifest_path is None:
        manifest_path = config['path'].get(
            'recordings',
            Path(config['path']['data_root']) / 'recordings.tsv',
        )
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise _recordings_error(
            manifest_path,
            [f'manifest not found: {manifest_path}'],
        )

    recordings, errors = _read_recording_rows(manifest_path)
    sourcedata = Path(config['path']['sourcedata'])
    available_files = {}
    if sourcedata.is_dir():
        available_files = {
            path.name: path for path in sourcedata.iterdir() if path.is_file()
        }

    files_to_rows = defaultdict(list)
    destinations_to_records = defaultdict(list)
    validated = []

    for record in recordings:
        row_number = record['row']
        file_name = record['file']
        files_to_rows[file_name].append(row_number)

        if not _is_plain_filename(file_name):
            errors.append(
                f'row {row_number}: file must be a name without a path: '
                f'{file_name!r}'
            )
        else:
            extension = Path(file_name).suffix.casefold()
            if extension not in SUPPORTED_SOURCE_EXTENSIONS:
                errors.append(
                    f'row {row_number}: unsupported file extension for '
                    f'{file_name!r}; expected .vhdr'
                )
            else:
                source_path = available_files.get(file_name)
                if source_path is None:
                    errors.append(
                        f'row {row_number}: source file not found in '
                        f'{sourcedata}: {file_name!r}'
                    )
                else:
                    errors.extend(
                        f'row {row_number}: {error}'
                        for error in _validate_brainvision_file(
                            source_path,
                            sourcedata,
                        )
                    )

        bids_errors = _validate_bids_identifiers(record)
        if bids_errors:
            errors.extend(
                f'row {row_number}: invalid BIDS identifiers: {error}'
                for error in bids_errors
            )
            continue

        try:
            bids_path = build_BIDS_object(
                config,
                record['subject'],
                record['session'],
                record['task'],
                record['run'],
            )
        except (TypeError, ValueError) as error:
            errors.append(
                f'row {row_number}: invalid BIDS identifiers: {error}'
            )
            continue

        validated_record = dict(record)
        validated_record['bids_path'] = bids_path
        validated.append(validated_record)
        destinations_to_records[str(bids_path.fpath)].append(validated_record)

    for file_name, row_numbers in files_to_rows.items():
        if len(row_numbers) > 1:
            errors.append(
                f'rows {row_numbers}: source file is repeated: {file_name!r}'
            )

    for destination, destination_records in destinations_to_records.items():
        if len(destination_records) < 2:
            continue
        rows = [record['row'] for record in destination_records]
        files = [record['file'] for record in destination_records]
        errors.append(
            f'rows {rows}: files {files} generate the same BIDS destination: '
            f'{destination}'
        )

    if errors:
        raise _recordings_error(manifest_path, errors)

    return validated
