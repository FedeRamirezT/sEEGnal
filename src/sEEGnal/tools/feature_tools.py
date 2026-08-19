"""

Map sensor and source feature outputs onto stable derivative layouts.

Functions
---------
get_channel_mapping
    Map processed EEG channels onto the canonical recording channel order.
expand_connectivity_vector
    Expand good-channel connections into a canonical NaN-filled vector.
expand_channel_values
    Expand good-channel values into a canonical NaN-filled channel axis.
get_source_metadata
    Describe the sensor input and vertex ordering of a source estimate.
require_finite_source_values
    Reject non-finite source values before derivative serialization.

"""

import numpy


def get_channel_mapping(all_ch_names, good_ch_names):
    """
    Map a processed EEG channel subset onto the canonical channel order.

    Parameters
    ----------
    all_ch_names : array-like of str
        Unique channel names in the canonical recording order.
    good_ch_names : array-like of str
        Unique retained channel names in their processed order.

    Returns
    -------
    good_channel_indices : numpy.ndarray
        Canonical integer index of each retained channel.
    bad_channel_indices : numpy.ndarray
        Canonical indices absent from ``good_ch_names``.
    bad_channels : list of str
        Canonical channel names absent from ``good_ch_names``.

    Raises
    ------
    ValueError
        If either channel list contains duplicates or a retained channel is
        absent from the canonical layout.
    """

    all_ch_names = list(all_ch_names)
    good_ch_names = list(good_ch_names)

    if len(all_ch_names) != len(set(all_ch_names)):
        raise ValueError('all_ch_names must contain unique channel names.')
    if len(good_ch_names) != len(set(good_ch_names)):
        raise ValueError('good_ch_names must contain unique channel names.')

    channel_index = {
        ch_name: index
        for index, ch_name in enumerate(all_ch_names)
    }
    missing_channels = [
        ch_name
        for ch_name in good_ch_names
        if ch_name not in channel_index
    ]
    if missing_channels:
        raise ValueError(
            'Good channels are missing from the canonical channel layout: '
            f'{missing_channels}.'
        )

    good_channel_indices = numpy.asarray(
        [channel_index[ch_name] for ch_name in good_ch_names],
        dtype=int,
    )
    good_channel_index_set = set(good_channel_indices.tolist())
    bad_channel_indices = numpy.asarray(
        [
            index
            for index in range(len(all_ch_names))
            if index not in good_channel_index_set
        ],
        dtype=int,
    )
    bad_channels = [
        all_ch_names[index]
        for index in bad_channel_indices
    ]

    return good_channel_indices, bad_channel_indices, bad_channels


def expand_connectivity_vector(
    connectivity_vector,
    good_channel_indices,
    n_channels,
):
    """
    Expand good-channel connections into a canonical NaN-filled vector.

    The final axis follows ``numpy.triu_indices(n_channels, k=1)``. Entries
    involving a channel outside ``good_channel_indices`` remain NaN.

    Parameters
    ----------
    connectivity_vector : numpy.ndarray
        One- or two-dimensional connectivity values whose final axis contains
        the upper triangle for the retained channels.
    good_channel_indices : array-like of int
        Canonical index of each retained channel, in the order used by
        ``connectivity_vector``.
    n_channels : int
        Number of channels in the canonical layout.

    Returns
    -------
    expanded_vector : numpy.ndarray
        Connectivity values expanded to the complete canonical upper
        triangle, with unavailable connections represented as NaN.
    nan_connection_indices : numpy.ndarray
        Indices of the canonical connections containing at least one absent
        channel.

    Raises
    ------
    ValueError
        If dimensions, indices or the connectivity-vector length are
        inconsistent with the requested channel layout.
    """

    connectivity_vector = numpy.asarray(connectivity_vector)
    good_channel_indices = numpy.asarray(good_channel_indices, dtype=int)

    if connectivity_vector.ndim not in (1, 2):
        raise ValueError(
            'connectivity_vector must have one or two dimensions.'
        )
    if n_channels < 0:
        raise ValueError('n_channels must be non-negative.')
    if len(numpy.unique(good_channel_indices)) != len(good_channel_indices):
        raise ValueError('good_channel_indices must contain unique indices.')
    if numpy.any(good_channel_indices < 0) or numpy.any(
        good_channel_indices >= n_channels
    ):
        raise ValueError(
            'good_channel_indices contains an index outside the canonical '
            'channel layout.'
        )

    n_good_channels = len(good_channel_indices)
    n_good_connections = n_good_channels * (n_good_channels - 1) // 2
    if connectivity_vector.shape[-1] != n_good_connections:
        raise ValueError(
            'The connectivity vector length does not match the number of '
            f'good channels: expected {n_good_connections}, received '
            f'{connectivity_vector.shape[-1]}.'
        )

    n_connections = n_channels * (n_channels - 1) // 2
    output_shape = connectivity_vector.shape[:-1] + (n_connections,)
    output_dtype = (
        connectivity_vector.dtype
        if numpy.issubdtype(connectivity_vector.dtype, numpy.floating)
        else numpy.dtype(float)
    )
    expanded_vector = numpy.full(
        output_shape,
        numpy.nan,
        dtype=output_dtype,
    )

    canonical_rows, canonical_cols = numpy.triu_indices(n_channels, k=1)
    connection_lookup = numpy.full((n_channels, n_channels), -1, dtype=int)
    connection_lookup[canonical_rows, canonical_cols] = numpy.arange(
        n_connections
    )

    good_rows, good_cols = numpy.triu_indices(n_good_channels, k=1)
    canonical_good_rows = good_channel_indices[good_rows]
    canonical_good_cols = good_channel_indices[good_cols]
    output_indices = connection_lookup[
        numpy.minimum(canonical_good_rows, canonical_good_cols),
        numpy.maximum(canonical_good_rows, canonical_good_cols),
    ]
    expanded_vector[..., output_indices] = connectivity_vector

    good_connection_mask = numpy.zeros(n_connections, dtype=bool)
    good_connection_mask[output_indices] = True
    nan_connection_indices = numpy.flatnonzero(~good_connection_mask)

    return expanded_vector, nan_connection_indices


def expand_channel_values(values, good_channel_indices, n_channels):
    """
    Expand good-channel values into a canonical NaN-filled channel axis.

    Parameters
    ----------
    values : numpy.ndarray
        Values whose first dimension follows the retained-channel order.
    good_channel_indices : array-like of int
        Canonical index corresponding to each row of ``values``.
    n_channels : int
        Number of channels in the canonical layout.

    Returns
    -------
    expanded_values : numpy.ndarray
        Values with a complete canonical first axis. Rows for unavailable
        channels contain NaN.

    Raises
    ------
    ValueError
        If the input has no channel axis or its shape and indices are
        inconsistent with the requested channel layout.
    """

    values = numpy.asarray(values)
    good_channel_indices = numpy.asarray(good_channel_indices, dtype=int)

    if values.ndim < 1:
        raise ValueError('values must have at least one dimension.')
    if values.shape[0] != len(good_channel_indices):
        raise ValueError(
            'The first values dimension must match good_channel_indices.'
        )
    if len(numpy.unique(good_channel_indices)) != len(good_channel_indices):
        raise ValueError('good_channel_indices must contain unique indices.')
    if numpy.any(good_channel_indices < 0) or numpy.any(
        good_channel_indices >= n_channels
    ):
        raise ValueError(
            'good_channel_indices contains an index outside the canonical '
            'channel layout.'
        )

    output_dtype = (
        values.dtype
        if numpy.issubdtype(values.dtype, numpy.floating)
        else numpy.dtype(float)
    )
    expanded_values = numpy.full(
        (n_channels,) + values.shape[1:],
        numpy.nan,
        dtype=output_dtype,
    )
    expanded_values[good_channel_indices] = values

    return expanded_values


def get_source_metadata(
    all_ch_names,
    bad_ch_names,
    filters,
    source_estimate,
    source_spacing='',
):
    """
    Describe the sensor input and vertex ordering of a source estimate.

    Parameters
    ----------
    all_ch_names : array-like of str
        Unique channel names in the canonical prepared sensor order.
    bad_ch_names : array-like of str
        Channels classified as bad before source reconstruction.
    filters : mne.beamformer.Beamformer | dict
        Spatial filter containing its ordered ``ch_names`` and source-space
        metadata.
    source_estimate : mne.SourceEstimate
        Source estimate whose vertices and first data dimension are described.
    source_spacing : str
        Configured source-space spacing identifier.

    Returns
    -------
    metadata : dict
        Sensor inclusion and exclusion mappings, source vertex identities,
        source-space descriptors and missing-value policies.

    Raises
    ------
    ValueError
        If channel names are duplicated or inconsistent, the source estimate
        contains no vertices, or its vertices do not match its data.
    """

    all_ch_names = list(all_ch_names)
    if len(all_ch_names) != len(set(all_ch_names)):
        raise ValueError('all_ch_names must contain unique channel names.')
    bad_ch_name_set = set(bad_ch_names)
    input_bad_channels = [
        ch_name
        for ch_name in all_ch_names
        if ch_name in bad_ch_name_set
    ]
    input_bad_channel_indices = [
        index
        for index, ch_name in enumerate(all_ch_names)
        if ch_name in bad_ch_name_set
    ]

    input_good_channels = list(filters['ch_names'])
    if len(input_good_channels) != len(set(input_good_channels)):
        raise ValueError(
            'The source filter channel names must be unique.'
        )
    channel_index = {
        ch_name: index
        for index, ch_name in enumerate(all_ch_names)
    }
    missing_channels = [
        ch_name
        for ch_name in input_good_channels
        if ch_name not in channel_index
    ]
    if missing_channels:
        raise ValueError(
            'The source filter contains channels outside the prepared sensor '
            f'layout: {missing_channels}.'
        )
    input_good_channel_indices = [
        channel_index[ch_name]
        for ch_name in input_good_channels
    ]
    input_good_channel_set = set(input_good_channels)
    input_excluded_channels = [
        ch_name
        for ch_name in all_ch_names
        if ch_name not in input_good_channel_set
    ]
    input_excluded_channel_indices = [
        index
        for index, ch_name in enumerate(all_ch_names)
        if ch_name not in input_good_channel_set
    ]

    vertex_parts = [
        numpy.asarray(vertices, dtype=int)
        for vertices in source_estimate.vertices
    ]
    if not vertex_parts:
        raise ValueError('The source estimate does not contain any vertices.')
    source_space_type = str(filters.get('src_type') or 'unknown')
    if source_space_type == 'surface' and len(vertex_parts) == 2:
        part_names = ['lh', 'rh']
    elif len(vertex_parts) == 1:
        part_names = [source_space_type]
    else:
        part_names = [
            f'part_{part_index}'
            for part_index in range(len(vertex_parts))
        ]

    source_vertex_numbers = numpy.concatenate(vertex_parts)
    source_vertex_spaces = [
        part_name
        for part_name, vertices in zip(part_names, vertex_parts)
        for _ in range(len(vertices))
    ]
    n_sources = source_estimate.data.shape[0]
    if len(source_vertex_numbers) != n_sources:
        raise ValueError(
            'The source vertex metadata does not match the first source-data '
            f'dimension: {len(source_vertex_numbers)} vertices for '
            f'{n_sources} source signals.'
        )

    source_subject = (
        getattr(source_estimate, 'subject', None)
        or filters.get('subject')
        or ''
    )

    return {
        'n_sources': n_sources,
        'n_input_channels': len(all_ch_names),
        'n_good_input_channels': len(input_good_channels),
        'input_ch_names': all_ch_names,
        'input_good_channels': input_good_channels,
        'input_good_channel_indices': input_good_channel_indices,
        'input_bad_channels': input_bad_channels,
        'input_bad_channel_indices': input_bad_channel_indices,
        'input_excluded_channels': input_excluded_channels,
        'input_excluded_channel_indices': input_excluded_channel_indices,
        'input_bad_channel_policy': 'exclude',
        'source_vertex_indices': list(range(n_sources)),
        'source_vertex_numbers': source_vertex_numbers.tolist(),
        'source_vertex_spaces': source_vertex_spaces,
        'source_space_type': source_space_type,
        'source_subject': str(source_subject),
        'source_spacing': str(source_spacing),
        'source_value_policy': 'finite',
    }


def require_finite_source_values(values, description):
    """
    Reject non-finite source values instead of encoding missing sensors.

    Parameters
    ----------
    values : array-like
        Source-space values to validate.
    description : str
        Output description included in the validation error.

    Returns
    -------
    values : numpy.ndarray
        Array view of the input after successful validation.

    Raises
    ------
    ValueError
        If any source value is NaN or infinite.
    """

    values = numpy.asarray(values)
    finite_mask = numpy.isfinite(values)
    if not finite_mask.all():
        n_nonfinite = int(numpy.size(values) - finite_mask.sum())
        raise ValueError(
            f'{description} contains {n_nonfinite} non-finite source values. '
            'Bad input sensors must be excluded before source reconstruction; '
            'they must not be represented as NaN source values.'
        )

    return values
