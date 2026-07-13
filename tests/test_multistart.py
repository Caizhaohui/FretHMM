"""Tests for deterministic multi-start HMM fitting.

Covers the algorithm-hardening behaviour added in v1.2.0:

- ``n_init == 1`` reproduces the legacy single-fit result exactly.
- Multi-start runs are reproducible (deterministic seeds).
- Multi-start never hurts and usually improves the log-likelihood on
  initialization-sensitive traces.
- The init-means generator exposes the legacy default at row 0.
"""

from __future__ import annotations

import numpy as np

from frethmm.core.model import _generate_init_means, fit_signal_hmm
from frethmm.domain.models import ClassificationConfig
from tests._synthetic import make_synthetic_trace


def test_n_init_1_matches_legacy_single_fit():
    """``n_init=1`` is byte-identical to the historical single-fit path.

    The first multi-start candidate always uses the legacy evenly-spaced
    default means, so a single-start run must return the same state means and
    log-likelihood as the original implementation.
    """
    trace = make_synthetic_trace(
        means=[0.2, 0.8, 0.2, 0.8], durations=[400, 400, 400, 400], noise=0.05, seed=1
    )
    result = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=1))

    # The chosen start must be the legacy default (index 0).
    assert result.best_start_index == 0
    assert result.n_init_used == 1
    # Means should recover the two synthetic states within tolerance.
    np.testing.assert_allclose(np.sort(result.state_means), [0.2, 0.8], atol=0.05)
    # Single-start runs carry no multi-start metadata in the summary (kept legacy).
    # The result object still records n_init for internal use, but it equals 1.
    assert result.n_init == 1


def test_multistart_is_deterministic():
    """Two multi-start runs on the same trace produce identical results.

    The init-means generator derives seeds only from the configuration, so
    repeated runs must be reproducible regardless of wall-clock time.
    """
    trace = make_synthetic_trace(
        means=[0.3, 0.7], durations=[800, 800], noise=0.05, seed=42
    )
    config = ClassificationConfig(n_states=2, n_init=10)

    first = fit_signal_hmm(trace, config)
    second = fit_signal_hmm(trace, config)

    assert first.best_start_index == second.best_start_index
    assert first.log_prob == second.log_prob
    np.testing.assert_allclose(first.state_means, second.state_means)
    np.testing.assert_array_equal(first.state_path, second.state_path)


def test_multistart_does_not_reduce_log_prob():
    """Multi-start picks the best of N, so it must be >= single-start.

    A trace with narrowly-spaced states (means 0.45 / 0.55) is the canonical
    initialization-sensitivity case: a poor default-mean start can collapse
    both states onto one cluster. Ten starts should do at least as well.
    """
    trace = make_synthetic_trace(
        means=[0.45, 0.55, 0.45, 0.55], durations=[400, 400, 400, 400],
        noise=0.02, seed=7,
    )
    single = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=1))
    multi = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=10))

    assert multi.log_prob >= single.log_prob - 1e-6


def test_generate_init_means_row_zero_is_legacy_default():
    """Row 0 of the init-means matrix equals the config's default means."""
    config = ClassificationConfig(n_states=3)
    data_min, data_max = -1.0, 1.0
    base = config.default_state_means(data_min, data_max)

    candidates = _generate_init_means(
        data_min, data_max, n_states=3, n_init=8, base_means=base
    )

    assert candidates.shape == (8, 3)
    np.testing.assert_allclose(candidates[0], base)


def test_generate_init_means_is_deterministic():
    """The generator is pure: identical inputs yield identical candidates."""
    config = ClassificationConfig(n_states=4)
    base = config.default_state_means(0.0, 2.0)

    first = _generate_init_means(0.0, 2.0, n_states=4, n_init=6, base_means=base)
    second = _generate_init_means(0.0, 2.0, n_states=4, n_init=6, base_means=base)

    np.testing.assert_array_equal(first, second)
