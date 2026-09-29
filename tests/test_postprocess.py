"""Unit tests for post-fit segment cleanup and state consolidation."""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.postprocess import (
    cleanup_classification_result,
    merge_minimum_dwell_segments,
    merge_nearly_identical_states,
)
from frethmm.domain.models import ClassificationConfig, ClassificationResult, SignalTrace


def test_merge_minimum_dwell_segments():
    means = np.array([10.0, 50.0, 100.0], dtype=np.float64)

    # 1. Isolated 1-frame flicker: 0 -> 1 (1 frame) -> 0
    path = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    merged = merge_minimum_dwell_segments(path, means, min_dwell_frames=2)
    np.testing.assert_array_equal(merged, np.zeros(7, dtype=np.int64))

    # 2. Boundary 1-frame: 1 (1 frame) -> 0 -> 0
    path_boundary = np.array([1, 0, 0, 0], dtype=np.int64)
    merged_boundary = merge_minimum_dwell_segments(path_boundary, means, min_dwell_frames=2)
    np.testing.assert_array_equal(merged_boundary, np.zeros(4, dtype=np.int64))

    # 3. Flanked by different states: 0 -> 1 (1 frame) -> 2
    # mean 10 vs 50 vs 100: |50-10|=40, |50-100|=50. Closer to 0!
    path_diff = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    merged_diff = merge_minimum_dwell_segments(path_diff, means, min_dwell_frames=2)
    assert merged_diff[2] == 0

    # 4. Dwells already >= min_dwell_frames are untouched
    path_valid = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    merged_valid = merge_minimum_dwell_segments(path_valid, means, min_dwell_frames=2)
    np.testing.assert_array_equal(merged_valid, path_valid)


def test_merge_nearly_identical_states():
    # 3 states where state 0 and 1 are nearly identical (e.g. 10.0 and 10.5)
    means = np.array([10.0, 10.5, 50.0], dtype=np.float64)
    transmat = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ], dtype=np.float64)
    path = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1], dtype=np.int64)

    new_means, new_transmat, new_path, logs = merge_nearly_identical_states(
        means, transmat, path, threshold=1.0
    )

    assert len(new_means) == 2
    assert len(logs) == 1
    # State 0 and 1 are merged
    assert new_means[0] == pytest.approx(10.25, abs=0.1)
    assert new_means[1] == pytest.approx(50.0)
    assert new_transmat.shape == (2, 2)
    # Old state 2 should now be state 1
    assert np.all(new_path[4:7] == 1)


def test_cleanup_classification_result_integration():
    time = np.arange(10, dtype=np.float64)
    signal = np.array([10.0, 10.0, 10.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    trace = SignalTrace(time=time, signal=signal, observations=signal)

    # Initial result with a 1-frame transient flicker at frame 3
    path = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    means = np.array([10.0, 50.0])
    res = ClassificationResult(
        n_states=2,
        log_prob=-100.0,
        state_means=means,
        state_sigma=2.0,
        signal_sigma=10.0,
        transition_matrix=np.eye(2),
        state_path=path,
        classified_signal=means[path],
        fraction_spent=np.zeros((2, 2)),
        transitions_found=np.zeros((2, 2), dtype=int),
    )

    config = ClassificationConfig(n_states=2, min_dwell_frames=2)
    cleaned = cleanup_classification_result(res, trace, config)

    # Transient flicker merged to state 0
    np.testing.assert_array_equal(cleaned.state_path, np.zeros(10, dtype=np.int64))
    assert len(cleaned.postprocessing_applied) == 1
    assert "Merged 1 transient flicker frame(s)" in cleaned.postprocessing_applied[0]
