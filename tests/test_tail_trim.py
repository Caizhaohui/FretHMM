"""Regression tests for terminal low-state tail trimming."""

from __future__ import annotations

import numpy as np

from frethmm.core.model import trim_trace_after_low_state_tail
from frethmm.domain.models import ClassificationResult, SignalTrace


def _trace_and_result(state_path: list[int]) -> tuple[SignalTrace, ClassificationResult]:
    time = np.arange(len(state_path), dtype=float)
    signal = np.asarray(state_path, dtype=float)
    trace = SignalTrace(time=time, signal=signal, observations=signal)
    result = ClassificationResult(
        n_states=2,
        log_prob=0.0,
        state_means=np.array([0.0, 1.0]),
        state_sigma=0.1,
        signal_sigma=0.5,
        transition_matrix=np.eye(2),
        state_path=np.asarray(state_path, dtype=int),
        classified_signal=signal,
        fraction_spent=np.zeros((2, 2)),
        transitions_found=np.zeros((2, 2), dtype=int),
    )
    return trace, result


def test_terminal_low_state_tail_is_trimmed_after_threshold():
    trace, result = _trace_and_result([1] * 10 + [0] * 10)

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff == 13.0
    np.testing.assert_array_equal(trimmed.time, np.arange(14, dtype=float))


def test_nonterminal_low_state_segment_is_not_trimmed():
    trace, result = _trace_and_result([0] * 10 + [1] * 10)

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff is None
    assert trimmed is trace
