"""Unit tests for single-molecule signal preprocessing algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.preprocess import (
    apply_median_filter,
    clean_outlier_spikes,
    detect_and_trim_initial_artifacts,
    detect_outlier_spikes,
    estimate_robust_sigma,
    preprocess_signal_trace,
)
from frethmm.domain.models import ClassificationConfig, SignalTrace


def test_estimate_robust_sigma():
    # 1. Constant signal
    const = np.full(100, 50.0)
    assert estimate_robust_sigma(const) == 1.0

    # 2. Gaussian noise with std ~ 2.0
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=10.0, scale=2.0, size=2000)
    sigma = estimate_robust_sigma(noise)
    assert 1.8 < sigma < 2.2

    # 3. Robust to outliers: 10 extreme spikes should not blow up MAD
    noise_with_spikes = noise.copy()
    noise_with_spikes[::200] = 5000.0
    sigma_spikes = estimate_robust_sigma(noise_with_spikes)
    assert 1.8 < sigma_spikes < 2.4


def test_detect_and_clean_outlier_spikes():
    # Trace with 2 baseline states + 2 isolated spikes
    rng = np.random.default_rng(123)
    n = 200
    base = np.zeros(n)
    base[100:] = 10.0
    noise = rng.normal(0, 0.5, n)
    signal = base + noise

    # Add isolated single-frame spikes
    signal[30] = 50.0   # Large positive spike
    signal[150] = -40.0  # Large negative spike

    mask = detect_outlier_spikes(signal, threshold_sigma=4.0, window_size=11)
    assert mask[30]
    assert mask[150]
    # No false positives in quiet regions
    assert np.sum(mask) == 2

    # Median cleaning
    cleaned_med = clean_outlier_spikes(signal, mask, method="median", window_size=11)
    assert abs(cleaned_med[30]) < 3.0  # Near baseline 0
    assert abs(cleaned_med[150] - 10.0) < 3.0  # Near baseline 10

    # Linear interpolation cleaning
    cleaned_interp = clean_outlier_spikes(signal, mask, method="linear_interp")
    assert abs(cleaned_interp[30]) < 3.0
    assert abs(cleaned_interp[150] - 10.0) < 3.0


def test_detect_and_trim_initial_artifacts():
    time = np.arange(100, dtype=np.float64) * 0.1
    signal = np.full(100, 1000.0)

    # Frame 0 is an extreme readout negative surge (like spot_102)
    signal[0] = -7000.0

    t_trim, s_trim, n_trimmed = detect_and_trim_initial_artifacts(
        time, signal, max_frames=5, threshold_sigma=5.0
    )
    assert n_trimmed == 1
    assert len(t_trim) == 99
    assert t_trim[0] == pytest.approx(0.1)
    assert s_trim[0] == 1000.0

    # Two frames of artifact
    signal[1] = -5000.0
    t_trim2, s_trim2, n_trimmed2 = detect_and_trim_initial_artifacts(
        time, signal, max_frames=5, threshold_sigma=5.0
    )
    assert n_trimmed2 == 2
    assert t_trim2[0] == pytest.approx(0.2)

    # Clean signal has 0 trimmed frames
    clean_sig = np.full(100, 1000.0)
    t_clean, s_clean, n_clean = detect_and_trim_initial_artifacts(time, clean_sig)
    assert n_clean == 0
    assert len(t_clean) == 100


def test_apply_median_filter():
    signal = np.array([10.0, 10.0, 100.0, 10.0, 10.0, 20.0, 20.0])
    filtered = apply_median_filter(signal, window_size=3)
    # The single point outlier 100.0 is suppressed to 10.0
    assert filtered[2] == 10.0
    # Steps are preserved
    assert filtered[5] == 20.0


def test_preprocess_signal_trace():
    time = np.arange(50, dtype=np.float64) * 0.1
    signal = np.ones(50) * 100.0
    signal[0] = -8000.0  # Initial edge artifact
    signal[25] = 900.0   # Isolated spike

    trace = SignalTrace(
        time=time,
        signal=signal.copy(),
        observations=signal.copy(),
        mode="single_channel",
    )
    config = ClassificationConfig(
        n_states=2,
        trim_initial_artifacts=True,
        remove_spikes=True,
        spike_threshold_sigma=5.0,
    )

    cleaned_trace, logs = preprocess_signal_trace(trace, config)
    assert len(logs) == 2
    assert cleaned_trace.n_frames == 49
    assert cleaned_trace.time[0] == pytest.approx(0.1)
    assert cleaned_trace.observations[24] < 200.0  # Spike at index 25 (shifted to 24) cleaned
