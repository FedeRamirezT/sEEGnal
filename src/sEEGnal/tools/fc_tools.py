# -*- coding: utf-8 -*-
"""
Functional Connectivity utilities for sEEGnal

Federico Ramírez-Toraño
26/02/2026
"""

# Imports
import numpy
from scipy.signal import hilbert


def compute_plv(data=None, average_epochs=True, dtype=numpy.float32):
    """
    Compute Phase Locking Value (PLV).

    Data must have shape:
        n_epochs x n_sources x n_samples

    This version is memory-efficient when average_epochs=True because it does
    not store the full n_sources x n_sources x n_epochs complex cube.
    """

    nepochs, nsources, nsamples = data.shape

    triu_idx = numpy.triu_indices(nsources, k=1)
    n_connections = len(triu_idx[0])

    if average_epochs:

        # Accumulate only the upper-triangular PLV values across epochs
        plv_sum = numpy.zeros(n_connections, dtype=dtype)

        for iepoch in range(nepochs):

            sourcedata = data[iepoch, :, :]  # shape: n_sources x n_samples

            # Compute analytic signal to extract phase
            analytic_signal = hilbert(sourcedata, axis=-1)
            sourcevector = analytic_signal / numpy.abs(analytic_signal)

            # Complex PLV for this epoch
            cplv = (
                sourcevector @ numpy.conj(sourcevector.T)
            ) / nsamples

            # Store only PLV magnitude from upper triangle
            plv_sum += numpy.abs(cplv[triu_idx]).astype(dtype)

        plv_vector = plv_sum / nepochs

    else:

        # Keep per-epoch PLV values, but only vectorized upper triangle
        plv_vector = numpy.zeros(
            (nepochs, n_connections),
            dtype=dtype
        )

        for iepoch in range(nepochs):

            sourcedata = data[iepoch, :, :]  # shape: n_sources x n_samples

            # Compute analytic signal to extract phase
            analytic_signal = hilbert(sourcedata, axis=-1)
            sourcevector = analytic_signal / numpy.abs(analytic_signal)

            # Complex PLV for this epoch
            cplv = (
                sourcevector @ numpy.conj(sourcevector.T)
            ) / nsamples

            # Store only PLV magnitude from upper triangle
            plv_vector[iepoch, :] = numpy.abs(cplv[triu_idx]).astype(dtype)

    return plv_vector


def compute_ciplv(data=None, average_epochs=True, dtype=numpy.float32):
    """
    Compute corrected imaginary Phase Locking Value (ciPLV).

    Data must have shape:
        n_epochs x n_sources x n_samples

    This version is memory-efficient when average_epochs=True because it does
    not store the full n_sources x n_sources x n_epochs cube.
    """

    nepochs, nsources, nsamples = data.shape

    triu_idx = numpy.triu_indices(nsources, k=1)
    n_connections = len(triu_idx[0])

    if average_epochs:

        # Accumulate only the upper-triangular ciPLV values across epochs
        ciplv_sum = numpy.zeros(n_connections, dtype=dtype)

        for iepoch in range(nepochs):

            sourcedata = data[iepoch, :, :]  # shape: n_sources x n_samples

            # Compute analytic signal to extract phase
            analytic_signal = hilbert(sourcedata, axis=-1)
            sourcevector = analytic_signal / numpy.abs(analytic_signal)

            # Complex PLV for this epoch
            cplv = (
                sourcevector @ numpy.conj(sourcevector.T)
            ) / nsamples

            # ciPLV for this epoch
            real_plv = numpy.real(cplv)
            imag_plv = numpy.imag(cplv)

            # Numerical safeguard:
            # real_plv can be slightly outside [-1, 1] due to floating-point precision,
            # e.g. 1.0000000000000002, which would make sqrt(1 - real_plv**2) NaN.
            sqrt_argument = 1 - real_plv ** 2
            sqrt_argument[sqrt_argument < 0] = 0

            denom = numpy.sqrt(sqrt_argument)
            denom[denom == 0] = numpy.finfo(dtype).eps

            ciplv_epoch = numpy.abs(imag_plv / denom)

            # Store only upper triangle
            ciplv_sum += ciplv_epoch[triu_idx].astype(dtype)

        ciplv_vector = ciplv_sum / nepochs

    else:

        # Keep per-epoch ciPLV values, but only vectorized upper triangle
        ciplv_vector = numpy.zeros(
            (nepochs, n_connections),
            dtype=dtype
        )

        for iepoch in range(nepochs):

            sourcedata = data[iepoch, :, :]  # shape: n_sources x n_samples

            # Compute analytic signal to extract phase
            analytic_signal = hilbert(sourcedata, axis=-1)
            sourcevector = analytic_signal / numpy.abs(analytic_signal)

            # Complex PLV for this epoch
            cplv = (
                sourcevector @ numpy.conj(sourcevector.T)
            ) / nsamples

            # ciPLV for this epoch
            real_plv = numpy.real(cplv)
            imag_plv = numpy.imag(cplv)

            # Numerical safeguard:
            # real_plv can be slightly outside [-1, 1] due to floating-point precision,
            # e.g. 1.0000000000000002, which would make sqrt(1 - real_plv**2) NaN.
            sqrt_argument = 1 - real_plv ** 2
            sqrt_argument[sqrt_argument < 0] = 0

            denom = numpy.sqrt(sqrt_argument)
            denom[denom == 0] = numpy.finfo(dtype).eps

            ciplv_epoch = numpy.abs(imag_plv / denom)

            # Store only upper triangle
            ciplv_vector[iepoch, :] = ciplv_epoch[triu_idx].astype(dtype)

    return ciplv_vector


def reconstruct_fc_matrix(plv_vector, conn_indices, n_sources):
    mat = numpy.zeros((n_sources, n_sources), dtype=plv_vector.dtype)
    mat[conn_indices] = plv_vector
    mat[(conn_indices[1], conn_indices[0])] = plv_vector
    return mat
