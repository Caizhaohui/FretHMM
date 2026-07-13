"""Tests for BIC/AIC metrics and BIC-based state-count selection.

Covers the algorithm-hardening behaviour added in v1.2.0:

- :mod:`frethmm.core.metrics` formulas are numerically correct.
- ``--states auto`` (``select_best_state_count``) recovers the true state
  count on an unambiguous multi-state synthetic trace.
- The returned result carries BIC, AIC, and the full candidate list.
"""

from __future__ import annotations

import math

import numpy as np

from frethmm.core.metrics import (
    compute_aic,
    compute_bic,
    count_gaussian_hmm_params,
)
from frethmm.core.model import select_best_state_count
from frethmm.domain.models import ClassificationConfig
from tests._synthetic import make_synthetic_trace


def test_count_gaussian_hmm_params_tied_covariance():
    """n=2, 1-D, tied covariance: 2 means + 1 sigma + 4 trans + 2 start = 9."""
    assert count_gaussian_hmm_params(2) == 9
    # n=3: 3 means + 1 sigma + 9 trans + 3 start = 16
    assert count_gaussian_hmm_params(3) == 16


def test_compute_bic_formula():
    """BIC = k*ln(n) - 2*log_prob (lower is better)."""
    n_params = 5
    log_prob = -100.0
    n_samples = 1000
    expected = n_params * math.log(n_samples) - 2.0 * log_prob
    assert compute_bic(n_params, log_prob, n_samples) == expected


def test_compute_aic_formula():
    """AIC = 2*k - 2*log_prob (lower is better)."""
    assert compute_aic(5, -100.0) == 2.0 * 5 - 2.0 * (-100.0)


def test_compute_bic_degenerate_samples_returns_inf():
    """n_samples <= 1 cannot define a penalty; return +inf to stay safe."""
    assert math.isinf(compute_bic(5, -100.0, 1))
    assert math.isinf(compute_bic(5, -100.0, 0))


def test_select_best_state_count_recovers_three_states():
    """A clearly three-state trace should be picked as 3, not 2 or 5.

    Well-separated means (0.1 / 0.5 / 0.9) with low noise leave no ambiguity,
    so BIC must favour the true count.
    """
    trace = make_synthetic_trace(
        means=[0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
        durations=[500, 500, 500, 500, 500, 500],
        noise=0.03,
        seed=11,
    )
    config = ClassificationConfig(
        n_states="auto", min_states=2, max_states=5, n_init=4
    )

    result = select_best_state_count(trace, config)

    assert result.n_states == 3
    # Recovered means should match the three synthetic levels.
    np.testing.assert_allclose(np.sort(result.state_means), [0.1, 0.5, 0.9], atol=0.05)


def test_select_best_state_count_populates_bic_and_candidates():
    """Auto-selection annotates the result with BIC and the candidate table."""
    trace = make_synthetic_trace(
        means=[0.2, 0.8], durations=[800, 800], noise=0.05, seed=3
    )
    config = ClassificationConfig(
        n_states="auto", min_states=2, max_states=4, n_init=3
    )

    result = select_best_state_count(trace, config)

    assert result.bic is not None
    assert result.aic is not None
    assert result.model_candidates is not None
    # One candidate per state count in [min_states, max_states].
    assert len(result.model_candidates) == 4 - 2 + 1
    # The winner's BIC must be the minimum across candidates.
    candidate_bics = [c["bic"] for c in result.model_candidates]
    assert result.bic == min(candidate_bics)
    # Each candidate row records the expected keys.
    for candidate in result.model_candidates:
        assert {"n_states", "log_prob", "bic", "aic"} <= set(candidate.keys())


def test_select_best_state_count_requires_auto_sentinel():
    """Calling the selector with a fixed int n_states is a programmer error."""
    import pytest

    trace = make_synthetic_trace(
        means=[0.2, 0.8], durations=[400, 400], noise=0.05, seed=5
    )
    with pytest.raises(ValueError):
        select_best_state_count(trace, ClassificationConfig(n_states=2))
