"""Tests for the classified-CSV reverse-parser.

Verifies that :func:`frethmm.formats.classified_parser.read_classified_csv`
reconstructs ``(state_path, state_means, times)`` correctly from
``*_classified.csv`` files, and round-trips with
:func:`frethmm.core.io.write_classified_csv`.
"""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.io import write_classified_csv
from frethmm.domain.models import ClassificationResult, SignalTrace
from frethmm.formats.classified_parser import read_classified_csv


def _write_classified(tmp_path, state_means, state_path, name="trace.csv"):
    """Write a minimal classified CSV via the production writer."""
    trace = SignalTrace(
        time=np.arange(len(state_path), dtype=np.float64),
        signal=np.asarray(state_means, dtype=np.float64)[np.asarray(state_path)],
        observations=np.zeros(len(state_path)),
        filepath=tmp_path / name,
        mode="single_channel",
    )
    classified = np.asarray(state_means, dtype=np.float64)[np.asarray(state_path)]
    result = ClassificationResult(
        n_states=len(state_means),
        log_prob=0.0,
        state_means=np.asarray(state_means, dtype=np.float64),
        state_sigma=0.1,
        signal_sigma=0.2,
        transition_matrix=np.eye(len(state_means)),
        state_path=np.asarray(state_path, dtype=np.int64),
        classified_signal=classified,
        fraction_spent=np.zeros((len(state_means), len(state_means))),
        transitions_found=np.zeros((len(state_means), len(state_means)), dtype=int),
        filepath=tmp_path / name,
    )
    return write_classified_csv(trace, result, tmp_path)


def test_read_classified_csv_returns_three_arrays(tmp_path):
    """A standard two-column file parses into state_path, state_means, times."""
    csv_path = tmp_path / "trace_classified.csv"
    csv_path.write_text(
        "time,classified_mean\n0,0.2\n1,0.2\n2,0.8\n3,0.8\n",
        encoding="utf-8",
    )

    state_path, state_means, times = read_classified_csv(csv_path)

    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(state_means, [0.2, 0.8])
    np.testing.assert_array_equal(state_path, [0, 0, 1, 1])


def test_read_classified_csv_roundtrip_with_write_classified_csv(tmp_path):
    """write_classified_csv then read_classified_csv preserves the state path."""
    original_means = np.array([0.15, 0.55, 0.92])
    original_path = np.array([0, 0, 1, 2, 2, 1, 0], dtype=np.int64)

    csv_path = _write_classified(tmp_path, original_means, original_path)
    state_path, state_means, _times = read_classified_csv(csv_path)

    np.testing.assert_allclose(np.sort(state_means), np.sort(original_means))
    # Re-map original_path through the parser's ascending sort to compare.
    remap = {v: i for i, v in enumerate(np.argsort(original_means))}
    expected = np.array([remap[s] for s in original_path], dtype=np.int64)
    np.testing.assert_array_equal(state_path, expected)


def test_read_classified_csv_rejects_missing_columns(tmp_path):
    """Missing the classified_mean column raises ValueError."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time,signal\n0,1.0\n1,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="classified_mean"):
        read_classified_csv(csv_path)


def test_read_classified_csv_rejects_empty_file(tmp_path):
    """An empty (header-only) file raises ValueError."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("time,classified_mean\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        read_classified_csv(csv_path)


def test_read_classified_csv_handles_three_states(tmp_path):
    """A 3-state classified CSV yields a length-3 state_means and 3 distinct indices."""
    csv_path = tmp_path / "trace_classified.csv"
    csv_path.write_text(
        "time,classified_mean\n"
        "0,0.2\n1,0.2\n2,0.5\n3,0.9\n4,0.5\n5,0.2\n",
        encoding="utf-8",
    )

    state_path, state_means, _times = read_classified_csv(csv_path)

    assert len(state_means) == 3
    np.testing.assert_allclose(state_means, [0.2, 0.5, 0.9])
    np.testing.assert_array_equal(state_path, [0, 0, 1, 2, 1, 0])
