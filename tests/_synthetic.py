"""Synthetic trace builders for algorithm-hardening tests.

These helpers live under ``tests/`` (not in the ``frethmm`` package) on
purpose: they are test fixtures, not a public API. They construct
:class:`frethmm.domain.models.SignalTrace` instances with known ground-truth
state means so that multi-start fitting and BIC model selection can be
validated deterministically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from frethmm.domain.models import SignalTrace


def make_synthetic_trace(
    means: Sequence[float],
    durations: Sequence[int],
    noise: float = 0.05,
    *,
    seed: int = 0,
    dt: float = 1.0,
) -> SignalTrace:
    """Build a single-channel trace with known state means.

    The trace alternates between the listed ``means`` for the listed
    ``durations`` (one segment per mean/duration pair). Gaussian noise with the
    given standard deviation is added per frame, drawn from a fixed-seed RNG so
    the result is fully reproducible.

    Parameters
    ----------
    means:
        Emission mean for each segment.
    durations:
        Number of frames each segment lasts. Must be the same length as
        ``means``.
    noise:
        Per-frame Gaussian noise standard deviation.
    seed:
        Seed for the noise RNG.
    dt:
        Time step between frames.
    """
    means = np.asarray(means, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.int64)
    if means.shape != durations.shape:
        raise ValueError(
            f"means and durations must have the same length, "
            f"got {means.shape} vs {durations.shape}"
        )
    if np.any(durations <= 0):
        raise ValueError(f"durations must be positive, got {durations.tolist()}")

    rng = np.random.default_rng(seed)
    signal = np.concatenate(
        [rng.normal(loc=m, scale=noise, size=int(d)) for m, d in zip(means, durations)]
    )
    time = np.arange(len(signal), dtype=np.float64) * dt
    return SignalTrace(
        time=time,
        signal=signal.copy(),
        observations=signal.copy(),
        mode="single_channel",
    )
