"""

Estimate and normalize power spectral density for sEEGnal.

Functions
---------
compute_dpss
    Generate DPSS tapers and retain the tapers that satisfy the bias criterion.
multitaper_psd
    Estimate power spectral density with fixed or adaptive multitaper weights.
_adaptive_weighting
    Iteratively estimate adaptive multitaper weights for each signal.
normalize_psd
    Normalize a power spectrum using the requested normalization mode.

Federico Ramírez-Toraño
26/02/2026

"""

import numpy as np
from scipy.signal.windows import dpss
from scipy.fft import rfft, rfftfreq


# ==========================================================
# DPSS
# ==========================================================

def compute_dpss(n_times, sfreq, bandwidth, low_bias=True):
    """
    Generate DPSS tapers and retain the tapers that satisfy the bias criterion.

    Parameters
    ----------
    n_times : int
        Number of time samples.
    sfreq : float
        Sampling frequency in hertz.
    bandwidth : float
        Multitaper spectral smoothing bandwidth in hertz.
    low_bias : bool
        Whether to retain only high-concentration DPSS tapers.

    Returns
    -------
    tapers : numpy.ndarray
        DPSS tapers with shape ``(n_tapers, n_times)``.
    eigvals : numpy.ndarray
        Concentration ratio of each retained taper.
    """

    NW = bandwidth * n_times / (2 * sfreq)
    Kmax = int(2 * NW)
    tapers, eigvals = dpss(n_times, NW, Kmax, return_ratios=True)
    if low_bias:
        mask = eigvals > 0.9
        tapers = tapers[mask]
        eigvals = eigvals[mask]
    return tapers, eigvals


# ==========================================================
# Multitaper PSD (epoch-by-epoch, taper-by-taper)
# ==========================================================

def multitaper_psd(
    data,
    sfreq,
    fmin=0.0,
    fmax=np.inf,
    bandwidth=4.0,
    adaptive=False,
    low_bias=True,
    scaling="density",
    average_epochs=True,
    dtype=np.float32
):
    """
    Estimate power spectral density with fixed or adaptive multitaper weights.

    Parameters
    ----------
    data : numpy.ndarray
        Signal data supplied to the computation.
    sfreq : float
        Sampling frequency in hertz.
    fmin : float
        Lowest frequency included in the estimate.
    fmax : float
        Highest frequency included in the estimate.
    bandwidth : float
        Multitaper spectral smoothing bandwidth in hertz.
    adaptive : bool
        Whether to use adaptive multitaper weighting.
    low_bias : bool
        Whether to retain only high-concentration DPSS tapers.
    scaling : str
        Spectral scaling mode.
    average_epochs : bool
        Whether to average the result across epochs.
    dtype : numpy.dtype
        Data type used for the output values.

    Returns
    -------
    psd : numpy.ndarray
        Estimated power spectral density.
    freqs : numpy.ndarray
        Frequencies corresponding to the spectral bins.
    """

    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ValueError(
            'multitaper_psd requires real-valued time-domain data; received '
            f'{data.dtype}. Check the source-reconstruction filters.'
        )
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n_epochs, n_signals, n_times = data.shape

    # DPSS tapers
    tapers, eigvals = compute_dpss(n_times, sfreq, bandwidth, low_bias)
    n_tapers = len(tapers)

    # Frequency vector
    freqs = rfftfreq(n_times, 1 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    # Output PSD
    psd = np.zeros((n_epochs, n_signals, n_freqs), dtype=dtype)

    # Loop over epochs to reduce memory
    for e in range(n_epochs):
        # PSD per taper
        taper_psds = np.zeros((n_signals, n_tapers, n_freqs), dtype=dtype)

        for k in range(n_tapers):
            tapered = data[e] * tapers[k]  # n_signals x n_times
            fft_data = rfft(tapered, axis=-1)[:, freq_mask]

            # One-sided correction
            fft_data[:, 1:-1] *= np.sqrt(2.0)

            # Power scaling
            if scaling == "density":
                power = np.abs(fft_data) ** 2 / (sfreq * n_times)
            elif scaling == "spectrum":
                power = np.abs(fft_data) ** 2 / n_times
            else:
                raise ValueError("scaling must be 'density' or 'spectrum'")

            taper_psds[:, k, :] = power

        # Adaptive weighting
        if adaptive:
            epoch_psd = _adaptive_weighting(taper_psds, eigvals)
        else:
            epoch_psd = taper_psds.mean(axis=1)

        psd[e] = epoch_psd

    if average_epochs:
        psd = psd.mean(axis=0)

    return psd, freqs


# ==========================================================
# Thomson Adaptive Weighting
# ==========================================================

def _adaptive_weighting(psd, eigvals, max_iter=50, tol=1e-10):
    """
    Iteratively estimate adaptive multitaper weights for each signal.

    Parameters
    ----------
    psd : numpy.ndarray
        Power spectral density values.
    eigvals : numpy.ndarray
        DPSS concentration ratios.
    max_iter : int | 'auto'
        Maximum number of ICA iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    spectrum : numpy.ndarray
        Adaptively weighted spectral estimate.
    """

    n_signals, n_tapers, n_freqs = psd.shape
    S = psd.mean(axis=1)  # initial estimate n_signals x n_freqs

    eigvals = eigvals[np.newaxis, :, np.newaxis]  # 1 x n_tapers x 1

    for _ in range(max_iter):
        S_prev = S.copy()
        weights = (eigvals * S[:, np.newaxis, :]) / (
            eigvals * S[:, np.newaxis, :] + (1 - eigvals) * psd
        )
        weights /= np.sum(weights, axis=1, keepdims=True)
        S = np.sum(weights * psd, axis=1)
        if np.max(np.abs(S - S_prev)) < tol:
            break

    return S


# ==========================================================
# PSD Normalization
# ==========================================================

def normalize_psd(psd, mode="relative"):
    """
    Normalize a power spectrum using the requested normalization mode.

    Parameters
    ----------
    psd : numpy.ndarray
        Power spectral density values.
    mode : str
        Normalization mode to apply.

    Returns
    -------
    normalized_psd : numpy.ndarray
        Spectrum after applying the requested normalization.
    """

    if mode is None:
        return psd
    if mode == "relative":
        total_power = np.sum(psd, axis=-1, keepdims=True)
        total_power[total_power == 0] = np.finfo(float).eps
        return psd / total_power
    if mode == "log":
        return np.log10(psd + np.finfo(float).eps)
    raise ValueError("mode must be None, 'relative', or 'log'")
