"""Model-selection metrics for Gaussian HMM fits.

These helpers complement :mod:`frethmm.core.model` by scoring fitted models so
that state-count selection (``--states auto``) and multi-start fitting can pick
the most defensible result. All formulas use the standard Akaike and Bayesian
information criteria expressed in terms of the model log-likelihood.
"""

from __future__ import annotations

import math

from typing import Union


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
    n_states:
        Number of hidden states in the fitted model.
    n_features:
        Dimensionality of the emission. FretHMM always fits a 1-D signal, so
        the default is ``1``.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    means = n_states * n_features
    # Tied, 1-D covariance -> one variance term per feature.
    covariance = n_features
    transition = n_states * n_states
    startprob = n_states
    return means + covariance + transition + startprob


# Backwards-friendly alias used by the refactor plan naming.
gaussian_hmm_n_params = count_gaussian_hmm_params


def compute_aic(n_params: int, log_prob: float) -> float:
    """Akaike information criterion: ``2*k - 2*log_prob``.

    Lower is better. ``log_prob`` is the total log-likelihood of the fitted
    model on the observed trace (as returned by ``hmmlearn``'s ``score``).
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
    """Score a fitted model using the chosen information criterion.

    Convenience wrapper used by model selection. ``criterion`` accepts
    ``"aic"``/``"bic"`` (case-insensitive) or a callable
    ``(n_params, log_prob, n_samples) -> float``.
    """
    n_params = count_gaussian_hmm_params(n_states, n_features=n_features)
    if callable(criterion):
        return float(criterion(n_params, log_prob, n_samples))
    name = str(criterion).lower()
    if name == "aic":
        return compute_aic(n_params, log_prob)
    if name == "bic":
        return compute_bic(n_params, log_prob, n_samples)
    raise ValueError(f"Unknown criterion: {criterion!r}")
