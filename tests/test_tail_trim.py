"""Regression tests for terminal low-state tail trimming."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frethmm.core.model import process_trace_file, trim_trace_after_low_state_tail
from frethmm.domain.models import ClassificationConfig, ClassificationResult, ExportOptions, SignalTrace


def _trace_and_result(
    state_path: list[int],
    time: np.ndarray | None = None,
) -> tuple[SignalTrace, ClassificationResult]:
    if time is None:
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


def test_early_nonterminal_low_state_run_is_trimmed_after_threshold():
    trace, result = _trace_and_result([1, 1, 0, 0, 0, 0, 1, 1])

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff == 5.0
    np.testing.assert_array_equal(trimmed.time, np.arange(6, dtype=float))


def test_low_state_run_below_threshold_is_not_trimmed():
    trace, result = _trace_and_result([1, 1, 0, 0, 0, 1, 1])

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff is None
    assert trimmed is trace


def test_non_low_interruption_starts_a_new_low_state_run():
    trace, result = _trace_and_result([1, 0, 0, 1, 0, 0, 0, 0, 1])

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff == 7.0
    np.testing.assert_array_equal(trimmed.time, np.arange(8, dtype=float))


def test_irregular_timestamps_measure_low_state_run_in_seconds():
    trace, result = _trace_and_result(
        [1, 0, 0, 0, 1, 1],
        time=np.array([0.0, 0.5, 1.25, 2.75, 3.0, 4.0]),
    )

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=2.0)

    assert cutoff == 2.5
    np.testing.assert_array_equal(trimmed.time, np.array([0.0, 0.5, 1.25]))


def test_terminal_low_state_tail_is_trimmed_after_threshold():
    trace, result = _trace_and_result([1] * 10 + [0] * 10)

    trimmed, cutoff = trim_trace_after_low_state_tail(trace, result, duration_seconds=3.0)

    assert cutoff == 13.0
    np.testing.assert_array_equal(trimmed.time, np.arange(14, dtype=float))


def test_process_trace_file_refits_trimmed_one_state_trace():
    filepath = Path(__file__).parent / "data" / "single_channel_trace.csv"
    config = ClassificationConfig(
        n_states=2,
        guesses=[0.2, 0.8],
        n_init=1,
        low_state_tail_trim_seconds=5.0,
        data_mode="single_channel",
        signal_column=1,
    )

    result = process_trace_file(
        filepath,
        config,
        export_options=ExportOptions(
            classified_csv=False,
            summary_json=False,
            state_report=False,
            state_path=False,
            dwell_report=False,
        ),
    )

    assert result.low_state_tail_cutoff_time == 5.0
    assert result.low_state_tail_kept_frames == 6
    assert len(result.state_path) == 6
