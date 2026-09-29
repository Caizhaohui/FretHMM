"""Signal preprocessing algorithms for single-molecule fluorescence trajectories.

This module provides signal conditioning and artifact rejection prior to
Hidden Markov Model (HMM) classification, addressing common physical and
instrumental issues in single-molecule data:

1. Initial acquisition artifacts (camera shutter delays, readout reset spikes).
2. Isolated impulse noise / outlier spikes (detector surges, cosmic rays).
3. Shot noise suppression via edge-preserving median filtering.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from frethmm.domain.models import ClassificationConfig, FloatArray, SignalTrace

logger = logging.getLogger(__name__)


def estimate_robust_sigma(signal: FloatArray) -> float:
    """Compute a robust scale estimate using the Median Absolute Deviation (MAD).

    In single-molecule data with state transitions or heavy-tailed noise,
    standard deviation is inflated by conformational steps and outliers. The
    MAD provides a breakdown-resistant scale estimate:
        sigma_MAD = 1.4826 * median(|x - median(x)|)

    Parameters
    ----------
    signal : FloatArray
        1-D array of fluorescence intensity or FRET efficiency values.

    Returns
    -------
    float
        Estimated standard deviation of Gaussian background noise. Returns 1.0
        if the signal has zero variance.
    """
    if len(signal) < 2:
        return 1.0
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 1e-12:
        return robust_sigma
    std = float(np.std(signal))
    if std > 1e-12:
        return std
    return 1.0


def detect_outlier_spikes(
    signal: FloatArray,
    threshold_sigma: float = 5.0,
    window_size: int = 11,
) -> np.ndarray:
    """Identify isolated outlier spikes in a single-molecule time series.

    Detector surges, stray photons, and camera readout glitches often manifest
    as transient, 1-2 frame spikes that deviate sharply from the local baseline.
    If left uncleaned, HMM fitting may allocate an entire state to capture a
    single outlier frame.

    Parameters
    ----------
    signal : FloatArray
        1-D fluorescence observation sequence.
    threshold_sigma : float, default=5.0
        Number of robust standard deviations above/below the local median to
        flag as an outlier.
    window_size : int, default=11
        Odd integer specifying the window size for local median baseline
        estimation.

    Returns
    -------
    np.ndarray
        Boolean mask array of shape ``(len(signal),)`` where True indicates
        an outlier spike frame.
    """
    if len(signal) < 3:
        return np.zeros(len(signal), dtype=bool)

    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(signal) if len(signal) % 2 != 0 else len(signal) - 1)
    if window_size < 3:
        return np.zeros(len(signal), dtype=bool)

    local_median = ndimage.median_filter(signal.astype(np.float64), size=window_size, mode="reflect")
    deviation = np.abs(signal - local_median)
    robust_sigma = estimate_robust_sigma(signal)

    spike_mask = deviation > (threshold_sigma * robust_sigma)
    return spike_mask


def clean_outlier_spikes(
    signal: FloatArray,
    spike_mask: np.ndarray,
    method: str = "median",
    window_size: int = 11,
) -> FloatArray:
    """Replace flagged outlier spikes to restore baseline continuity.

    Parameters
    ----------
    signal : FloatArray
        Original 1-D fluorescence signal.
    spike_mask : np.ndarray
        Boolean mask of outlier points produced by :func:`detect_outlier_spikes`.
    method : str, default="median"
        Replacement strategy:
        - "median": replace with local median filter value.
        - "linear_interp": linearly interpolate from adjacent valid frames.
    window_size : int, default=11
        Local window size for median replacement.

    Returns
    -------
    FloatArray
        Cleaned signal array with spikes substituted.
    """
    cleaned = signal.copy().astype(np.float64)
    if not np.any(spike_mask):
        return cleaned

    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(signal) if len(signal) % 2 != 0 else len(signal) - 1)
    if window_size < 3:
        window_size = 3

    if method == "linear_interp":
        valid_indices = np.where(~spike_mask)[0]
        if len(valid_indices) >= 2:
            spike_indices = np.where(spike_mask)[0]
            cleaned[spike_indices] = np.interp(
                spike_indices, valid_indices, signal[valid_indices]
            )
            return cleaned

    # Fallback to local median
    local_median = ndimage.median_filter(signal.astype(np.float64), size=window_size, mode="reflect")
    cleaned[spike_mask] = local_median[spike_mask]
    return cleaned


def detect_and_trim_initial_artifacts(
    time: FloatArray,
    signal: FloatArray,
    max_frames: int = 5,
    threshold_sigma: float = 5.0,
) -> Tuple[FloatArray, FloatArray, int]:
    """Detect and remove transient edge artifacts at the start of acquisition.

    In TIRF and confocal microscopy, the first 1-3 frames often suffer from
    AOTF/laser shutter timing lag, EMCCD clearing pulses, or transient camera
    reset offsets resulting in severe negative or saturated values (e.g. -7000 counts).

    Parameters
    ----------
    time : FloatArray
        1-D time axis values in seconds.
    signal : FloatArray
        1-D fluorescence intensity values.
    max_frames : int, default=5
        Maximum number of initial frames to inspect for edge artifacts.
    threshold_sigma : float, default=5.0
        Deviation multiplier relative to steady-state signal MAD.

    Returns
    -------
    Tuple[FloatArray, FloatArray, int]
        (trimmed_time, trimmed_signal, n_trimmed_frames)
    """
    n_total = len(signal)
    if n_total <= max_frames * 2 or max_frames < 1:
        return time, signal, 0

    steady_signal = signal[max_frames:]
    med = float(np.median(steady_signal))
    sigma = estimate_robust_sigma(steady_signal)

    trim_count = 0
    for i in range(max_frames):
        if np.abs(signal[i] - med) > threshold_sigma * sigma:
            trim_count = i + 1
        else:
            break

    if trim_count > 0:
        return time[trim_count:].copy(), signal[trim_count:].copy(), trim_count
    return time, signal, 0


def apply_median_filter(signal: FloatArray, window_size: int = 3) -> FloatArray:
    """Apply an edge-preserving 1-D median filter.

    Median filtering removes isolated shot noise without blurring abrupt
    single-molecule conformational step transitions.

    Parameters
    ----------
    signal : FloatArray
        1-D observation sequence.
    window_size : int, default=3
        Must be a positive odd integer.

    Returns
    -------
    FloatArray
        Filtered observation sequence.
    """
    if len(signal) < window_size or window_size <= 1:
        return signal.copy().astype(np.float64)
    if window_size % 2 == 0:
        window_size += 1
    return ndimage.median_filter(signal.astype(np.float64), size=window_size, mode="reflect")


def preprocess_signal_trace(
    trace: SignalTrace,
    config: ClassificationConfig,
) -> Tuple[SignalTrace, List[str]]:
    """Execute all configured preprocessing steps on a raw single-molecule trace.

    Parameters
    ----------
    trace : SignalTrace
        Input raw signal trace with time, observations, and channel arrays.
    config : ClassificationConfig
        HMM configuration containing preprocessing flags.

    Returns
    -------
    Tuple[SignalTrace, List[str]]
        The preprocessed SignalTrace and an audit list of actions taken.
    """
    logs: List[str] = []
    time = trace.time.copy()
    signal = trace.signal.copy()
    observations = trace.observations.copy()
    channel_1 = trace.channel_1.copy() if trace.channel_1 is not None else None
    channel_2 = trace.channel_2.copy() if trace.channel_2 is not None else None
    derived_signal = trace.derived_signal.copy() if trace.derived_signal is not None else None

    # Step 1: Initial edge artifact trimming
    if config.trim_initial_artifacts:
        time, observations, n_trimmed = detect_and_trim_initial_artifacts(
            time,
            observations,
            max_frames=config.max_initial_artifact_frames,
            threshold_sigma=config.spike_threshold_sigma,
        )
        if n_trimmed > 0:
            signal = signal[n_trimmed:]
            if channel_1 is not None:
                channel_1 = channel_1[n_trimmed:]
            if channel_2 is not None:
                channel_2 = channel_2[n_trimmed:]
            if derived_signal is not None:
                derived_signal = derived_signal[n_trimmed:]
            msg = (
                f"Trimmed {n_trimmed} initial artifact frame(s) "
                f"(first valid time={time[0]:.4g}s)."
            )
            logs.append(msg)
            logger.info("%s on %s", msg, trace.filepath.name if trace.filepath else "trace")

    # Step 2: Outlier spike detection and cleaning
    if config.remove_spikes:
        spike_mask = detect_outlier_spikes(
            observations,
            threshold_sigma=config.spike_threshold_sigma,
        )
        n_spikes = int(np.sum(spike_mask))
        if n_spikes > 0:
            observations = clean_outlier_spikes(observations, spike_mask, method="median")
            signal = clean_outlier_spikes(signal, spike_mask, method="median")
            msg = (
                f"Cleaned {n_spikes} outlier spike(s) exceeding "
                f"{config.spike_threshold_sigma:g} sigma."
            )
            logs.append(msg)
            logger.info("%s on %s", msg, trace.filepath.name if trace.filepath else "trace")

    # Step 3: Edge-preserving smoothing
    if config.smooth_window is not None and config.smooth_window > 1:
        observations = apply_median_filter(observations, window_size=config.smooth_window)
        signal = apply_median_filter(signal, window_size=config.smooth_window)
        msg = f"Applied median filter with window size {config.smooth_window}."
        logs.append(msg)
        logger.info("%s on %s", msg, trace.filepath.name if trace.filepath else "trace")

    cleaned_trace = SignalTrace(
        time=time,
        signal=signal,
        observations=observations,
        filepath=trace.filepath,
        mode=trace.mode,
        channel_1=channel_1,
        channel_2=channel_2,
        derived_signal=derived_signal,
    )
    return cleaned_trace, logs
