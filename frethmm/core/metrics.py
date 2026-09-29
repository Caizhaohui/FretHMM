"""Model-selection metrics and fit quality diagnostics for Gaussian HMM fits.

These helpers score fitted models for state-count selection (``--states auto``)
and provide quantitative physical diagnostics for single-molecule trajectories:
- Signal-to-Noise Ratio (SNR) based on inter-state spacing vs. emission width.
- Residual statistics (RMSE, MAE, R^2 coefficient of determination).
- State occupancy fractions and degeneracy warnings.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from frethmm.domain.models import FloatArray, IntArray


def count_gaussian_hmm_params(n_states: int, n_features: int = 1) -> int:
    """Count free parameters for a tied-covariance ``GaussianHMM``.

    The HMM engine in :mod:`frethmm.core.model` is configured with
    ``covariance_type="tied"`` so every state shares one covariance. For a
    1-D signal (``n_features == 1``) the free parameters are:

    - ``n_states`` state means (each a length-``n_features`` vector),
    - one shared covariance (``n_features`` variances on the diagonal for the
      spherical/tied 1-D case used here),
    - ``n_states ** 2`` transition probabilities (rows sum to one),
    - ``n_states`` initial-state probabilities (sum to one).

    Parameters
    ----------
    n_states : int
        Number of hidden states in the fitted model.
    n_features : int, default=1
        Dimensionality of the emission. FretHMM always fits a 1-D signal.

    Returns
    -------
    int
        Total count of estimated parameters.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    means = n_states * n_features
    covariance = n_features
    transition = n_states * n_states
    startprob = n_states
    return means + covariance + transition + startprob


# Backwards-friendly alias used by the refactor plan naming.
gaussian_hmm_n_params = count_gaussian_hmm_params


def compute_aic(n_params: int, log_prob: float) -> float:
    """Akaike information criterion: ``2*k - 2*log_prob``.

    Lower is better. ``log_prob`` is the total log-likelihood of the fitted
    model on the observed trace.
    """
    return 2.0 * n_params - 2.0 * log_prob


def compute_bic(n_params: int, log_prob: float, n_samples: int) -> float:
    """Bayesian information criterion: ``k*ln(n) - 2*log_prob``.

    Lower is better. ``n_samples`` is the number of frames (observations) used
    for fitting. Returns ``+inf`` when ``n_samples <= 1`` so callers can treat
    degenerate traces uniformly.
    """
    if n_samples <= 1:
        return float("inf")
    return n_params * math.log(n_samples) - 2.0 * log_prob


def score_result(
    n_states: int,
    log_prob: float,
    n_samples: int,
    n_features: int = 1,
    criterion: Union[str, type] = "bic",
) -> float:
    """Score a fitted model using the chosen information criterion."""
    n_params = count_gaussian_hmm_params(n_states, n_features=n_features)
    if callable(criterion):
        return float(criterion(n_params, log_prob, n_samples))
    name = str(criterion).lower()
    if name == "aic":
        return compute_aic(n_params, log_prob)
    if name == "bic":
        return compute_bic(n_params, log_prob, n_samples)
    raise ValueError(f"Unknown criterion: {criterion!r}")


def compute_snr(state_means: FloatArray, state_sigma: float) -> Tuple[float, float]:
    """Compute minimum and mean Signal-to-Noise Ratios (SNR) between adjacent states.

    In single-molecule microscopy, the resolvability of neighboring conformational
    states is governed by the ratio of inter-state step height to the emission
    noise standard deviation:
        SNR_min = min(mu_{i+1} - mu_i) / sigma
        SNR_mean = mean(mu_{i+1} - mu_i) / sigma

    Parameters
    ----------
    state_means : FloatArray
        Sorted emission means of the states.
    state_sigma : float
        Emission noise standard deviation (from HMM tied covariance).

    Returns
    -------
    Tuple[float, float]
        (snr_min, snr_mean). Returns (0.0, 0.0) if fewer than 2 states.
    """
    if len(state_means) < 2:
        return 0.0, 0.0
    sorted_means = np.sort(state_means)
    diffs = np.diff(sorted_means)
    denom = max(float(state_sigma), 1e-12)
    snr_min = float(np.min(diffs) / denom)
    snr_mean = float(np.mean(diffs) / denom)
    return snr_min, snr_mean


def compute_residuals_metrics(
    observations: FloatArray,
    classified_signal: FloatArray,
) -> Dict[str, float]:
    """Calculate trajectory reconstruction residual statistics.

    Parameters
    ----------
    observations : FloatArray
        1-D raw observations sequence.
    classified_signal : FloatArray
        1-D idealized trajectory step values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "rmse": Root Mean Square Error.
        - "mae": Mean Absolute Error.
        - "r_squared": Coefficient of determination R^2.
    """
    if len(observations) == 0 or len(classified_signal) == 0:
        return {"rmse": 0.0, "mae": 0.0, "r_squared": 0.0}

    residuals = observations.astype(np.float64) - classified_signal.astype(np.float64)
    ss_res = float(np.sum(residuals**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    mean_obs = float(np.mean(observations))
    ss_tot = float(np.sum((observations - mean_obs) ** 2))

    if ss_tot > 1e-12:
        r_squared = float(max(0.0, 1.0 - (ss_res / ss_tot)))
    else:
        r_squared = 1.0 if rmse < 1e-6 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
    }


def compute_fit_diagnostics(
    observations: FloatArray,
    state_means: FloatArray,
    state_sigma: float,
    state_path: IntArray,
    classified_signal: Optional[FloatArray] = None,
) -> Dict[str, object]:
    """Generate comprehensive fit quality metrics for a single-molecule trajectory.

    Parameters
    ----------
    observations : FloatArray
        Original observation data.
    state_means : FloatArray
        Ascending state emission means.
    state_sigma : float
        Emission standard deviation.
    state_path : IntArray
        Viterbi decoded state path.
    classified_signal : Optional[FloatArray]
        Precomputed idealized signal; computed if not provided.

    Returns
    -------
    Dict[str, object]
        Machine-readable dictionary containing SNR, residuals, state occupancies,
        and flags for low-occupancy or overlapping states.
    """
    n_states = len(state_means)
    total_frames = len(state_path)

    if classified_signal is None:
        classified_signal = state_means[state_path]

    snr_min, snr_mean = compute_snr(state_means, state_sigma)
    res_metrics = compute_residuals_metrics(observations, classified_signal)

    # State occupancy analysis
    counts = np.bincount(state_path, minlength=n_states)
    occupancies = (
        [float(c / total_frames) for c in counts[:n_states]]
        if total_frames > 0
        else [0.0] * n_states
    )

    # Low-occupancy warning (< 0.5% of total trajectory length)
    low_occupancy_states = [s for s, occ in enumerate(occupancies) if occ < 0.005]

    # Overlapping states warning (|mu_{i+1} - mu_i| < 1.0 * sigma)
    sorted_means = np.sort(state_means)
    overlapping_pairs: List[List[int]] = []
    for i in range(len(sorted_means) - 1):
        if (sorted_means[i + 1] - sorted_means[i]) < max(state_sigma, 1e-12):
            overlapping_pairs.append([i, i + 1])

    return {
        "snr_min": round(snr_min, 4),
        "snr_mean": round(snr_mean, 4),
        "rmse": round(res_metrics["rmse"], 4),
        "mae": round(res_metrics["mae"], 4),
        "r_squared": round(res_metrics["r_squared"], 4),
        "state_occupancies": [round(o, 4) for o in occupancies],
        "low_occupancy_states": low_occupancy_states,
        "overlapping_state_pairs": overlapping_pairs,
    }
