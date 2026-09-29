"""Unit tests for model-selection metrics and fit quality diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.metrics import (
    compute_aic,
    compute_bic,
    compute_fit_diagnostics,
    compute_residuals_metrics,
    compute_snr,
    count_gaussian_hmm_params,
    score_result,
)


def test_param_count_and_criteria():
    n_params = count_gaussian_hmm_params(2)
    assert n_params == 2 * 1 + 1 + 4 + 2  # 9
    aic = compute_aic(n_params, -100.0)
    assert aic == pytest.approx(2 * 9 - 2 * (-100.0))  # 218.0
    bic = compute_bic(n_params, -100.0, n_samples=100)
    assert bic > aic


def test_compute_snr():
    # 2 states: 0 and 10 with sigma 2 -> SNR = 5.0
    means = np.array([0.0, 10.0])
    snr_min, snr_mean = compute_snr(means, state_sigma=2.0)
    assert snr_min == pytest.approx(5.0)
    assert snr_mean == pytest.approx(5.0)

    # 3 states: 0, 4, 10 with sigma 2 -> diffs [4, 6] -> min=2.0, mean=2.5
    means3 = np.array([0.0, 4.0, 10.0])
    snr_min3, snr_mean3 = compute_snr(means3, state_sigma=2.0)
    assert snr_min3 == pytest.approx(2.0)
    assert snr_mean3 == pytest.approx(2.5)

    # 1 state
    assert compute_snr(np.array([5.0]), state_sigma=1.0) == (0.0, 0.0)


def test_compute_residuals_metrics():
    # Perfect fit
    obs = np.array([1.0, 1.0, 2.0, 2.0])
    pred = np.array([1.0, 1.0, 2.0, 2.0])
    metrics = compute_residuals_metrics(obs, pred)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r_squared"] == pytest.approx(1.0)

    # Deviations
    obs_noisy = np.array([1.2, 0.8, 2.1, 1.9])
    metrics_noisy = compute_residuals_metrics(obs_noisy, pred)
    assert metrics_noisy["rmse"] > 0.0
    assert 0.0 < metrics_noisy["r_squared"] < 1.0


def test_compute_fit_diagnostics():
    obs = np.array([0.0, 0.1, -0.1, 10.0, 9.9, 10.1, 10.2])
    means = np.array([0.0, 10.0])
    path = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    diag = compute_fit_diagnostics(
        observations=obs,
        state_means=means,
        state_sigma=0.5,
        state_path=path,
    )

    assert "snr_min" in diag
    assert "snr_mean" in diag
    assert "rmse" in diag
    assert "mae" in diag
    assert "r_squared" in diag
    assert "state_occupancies" in diag
    assert len(diag["state_occupancies"]) == 2
    assert diag["low_occupancy_states"] == []
    assert diag["overlapping_state_pairs"] == []

    # Test overlapping states warning
    overlap_diag = compute_fit_diagnostics(
        observations=obs,
        state_means=np.array([10.0, 10.2]),
        state_sigma=0.5,
        state_path=np.zeros(len(obs), dtype=np.int64),
    )
    assert overlap_diag["overlapping_state_pairs"] == [[0, 1]]
