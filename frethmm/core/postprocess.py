"""Post-processing and state cleanup for generic signal classification workflows."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from frethmm.domain.models import ClassificationConfig, ClassificationResult, FloatArray, IntArray, SignalTrace

logger = logging.getLogger(__name__)


def build_classified_signal(
    state_path: np.ndarray,
    state_means: np.ndarray,
) -> np.ndarray:
    """Construct idealized trajectory step function from state path and state means.

    Parameters
    ----------
    state_path : np.ndarray
        Integer array of Viterbi decoded state indices.
    state_means : np.ndarray
        Floating point array of emission means for each state.

    Returns
    -------
    np.ndarray
        Idealized signal matching the length of ``state_path``.
    """
    return state_means[state_path]


def extract_dwell_segments(result: ClassificationResult) -> np.ndarray:
    """Extract consecutive dwell segments from a fitted ClassificationResult.

    Parameters
    ----------
    result : ClassificationResult
        Result containing the Viterbi state path.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_events, 3)`` where columns are:
        ``[start_state, stop_state, duration_frames]``.
    """
    path = result.state_path
    if len(path) < 2:
        return np.empty((0, 3), dtype=np.float64)

    dwells = []
    current_state = int(path[0])
    start_idx = 0

    for i in range(1, len(path)):
        next_state = int(path[i])
        if next_state != current_state:
            dwells.append([current_state, next_state, i - start_idx])
            current_state = next_state
            start_idx = i

    return np.array(dwells, dtype=np.float64) if dwells else np.empty((0, 3), dtype=np.float64)


def compute_transition_stats(
    state_path: np.ndarray,
    n_states: int,
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate transition counts and fraction spent between all state pairs.

    Parameters
    ----------
    state_path : np.ndarray
        Integer sequence of assigned states.
    n_states : int
        Total number of unique states.
    n_frames : int
        Total frame count.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (fraction_spent, transitions_found) matrices of shape ``(n_states, n_states)``.
    """
    fraction_spent = np.zeros((n_states, n_states), dtype=np.float64)
    transitions_found = np.zeros((n_states, n_states), dtype=int)

    if n_frames == 0 or len(state_path) == 0:
        return fraction_spent, transitions_found

    current_state = int(state_path[0])
    start_idx = 0

    for i in range(1, len(state_path)):
        next_state = int(state_path[i])
        if next_state != current_state:
            duration = i - start_idx
            fraction_spent[current_state, next_state] += duration
            transitions_found[current_state, next_state] += 1
            current_state = next_state
            start_idx = i

    fraction_spent /= n_frames
    return fraction_spent, transitions_found


def merge_minimum_dwell_segments(
    state_path: IntArray,
    state_means: FloatArray,
    min_dwell_frames: int = 2,
    max_iter: int = 10,
) -> IntArray:
    """Filter out transient flicker dwells shorter than ``min_dwell_frames``.

    In single-molecule kinetics, brief transitions of 1 frame often represent
    stochastic noise crossing the decision threshold rather than genuine
    molecular conformational changes. This function merges such short segments
    into adjacent flanking states.

    Parameters
    ----------
    state_path : IntArray
        1-D sequence of state assignments.
    state_means : FloatArray
        Emission mean value for each state.
    min_dwell_frames : int, default=2
        Minimum allowable dwell duration in frames. Segments with fewer frames
        are merged.
    max_iter : int, default=10
        Maximum iterative passes to resolve cascades of short dwells.

    Returns
    -------
    IntArray
        Filtered state path with transient flickers merged.
    """
    if min_dwell_frames <= 1 or len(state_path) < min_dwell_frames:
        return state_path.copy()

    path = state_path.copy()
    for _ in range(max_iter):
        # Extract segments: list of (state, start_idx, end_idx)
        segments: List[Tuple[int, int, int]] = []
        cur_state = int(path[0])
        start_idx = 0
        for i in range(1, len(path)):
            if path[i] != cur_state:
                segments.append((cur_state, start_idx, i - 1))
                cur_state = int(path[i])
                start_idx = i
        segments.append((cur_state, start_idx, len(path) - 1))

        # Check if any segment is shorter than min_dwell_frames
        short_segments = [seg for seg in segments if (seg[2] - seg[1] + 1) < min_dwell_frames]
        if not short_segments:
            break

        # Process each short segment
        for seg_idx, (seg_state, start, end) in enumerate(segments):
            dur = end - start + 1
            if dur >= min_dwell_frames:
                continue

            # Case 1: Between two segments
            has_prev = seg_idx > 0
            has_next = seg_idx < len(segments) - 1

            if has_prev and has_next:
                prev_state = segments[seg_idx - 1][0]
                next_state = segments[seg_idx + 1][0]
                if prev_state == next_state:
                    target_state = prev_state
                else:
                    # Merge to closer mean
                    dist_prev = abs(state_means[seg_state] - state_means[prev_state])
                    dist_next = abs(state_means[seg_state] - state_means[next_state])
                    target_state = prev_state if dist_prev <= dist_next else next_state
            elif has_prev:
                target_state = segments[seg_idx - 1][0]
            elif has_next:
                target_state = segments[seg_idx + 1][0]
            else:
                target_state = seg_state

            path[start : end + 1] = target_state

    return path


def merge_nearly_identical_states(
    state_means: FloatArray,
    transition_matrix: FloatArray,
    state_path: IntArray,
    threshold: Optional[float] = None,
    sigma: Optional[float] = None,
    sigma_factor: Optional[float] = None,
) -> Tuple[FloatArray, FloatArray, IntArray, List[str]]:
    """Merge states with emission means that are virtually indistinguishable.

    If state count is over-specified, HMM may split a single physical state into
    two neighboring clusters. This function merges adjacent states whose means
    differ by less than ``threshold`` or less than ``sigma_factor * sigma``.

    Parameters
    ----------
    state_means : FloatArray
        Current ascending state means.
    transition_matrix : FloatArray
        Current transition probability matrix.
    state_path : IntArray
        Viterbi state assignments.
    threshold : Optional[float]
        Absolute difference threshold between adjacent state means.
    sigma : Optional[float]
        Emission standard deviation.
    sigma_factor : Optional[float]
        Relative threshold in units of ``sigma``.

    Returns
    -------
    Tuple[FloatArray, FloatArray, IntArray, List[str]]
        (new_state_means, new_transition_matrix, new_state_path, merge_logs)
    """
    n_states = len(state_means)
    if n_states <= 1:
        return state_means.copy(), transition_matrix.copy(), state_path.copy(), []

    diff_limit = None
    if threshold is not None and threshold > 0:
        diff_limit = float(threshold)
    elif sigma is not None and sigma_factor is not None and sigma_factor > 0:
        diff_limit = float(sigma * sigma_factor)

    if diff_limit is None:
        return state_means.copy(), transition_matrix.copy(), state_path.copy(), []

    logs: List[str] = []
    # Identify adjacent states to merge
    mapping = np.arange(n_states, dtype=np.int64)
    merged_any = False

    for i in range(n_states - 1):
        diff = abs(state_means[i + 1] - state_means[i])
        if diff <= diff_limit:
            mapping[i + 1] = mapping[i]
            msg = (
                f"Merged adjacent states {i} (mean={state_means[i]:.4g}) and "
                f"{i + 1} (mean={state_means[i + 1]:.4g}) with diff {diff:.4g} <= {diff_limit:.4g}."
            )
            logs.append(msg)
            logger.info(msg)
            merged_any = True

    if not merged_any:
        return state_means.copy(), transition_matrix.copy(), state_path.copy(), []

    # Compact mapped IDs to 0..new_n_states-1
    unique_targets = np.unique(mapping)
    new_n_states = len(unique_targets)
    compact_map = {orig: new for new, orig in enumerate(unique_targets)}
    final_mapping = np.array([compact_map[mapping[i]] for i in range(n_states)], dtype=np.int64)

    # Recompute state path
    new_state_path = final_mapping[state_path]

    # Recompute weighted state means from frame counts
    new_means = np.zeros(new_n_states, dtype=np.float64)
    counts = np.bincount(new_state_path, minlength=new_n_states)

    for old_s in range(n_states):
        new_s = final_mapping[old_s]
        old_count = np.sum(state_path == old_s)
        new_means[new_s] += state_means[old_s] * old_count

    for s in range(new_n_states):
        if counts[s] > 0:
            new_means[s] /= counts[s]
        else:
            # Fallback to simple average of mapped old means
            mapped_old = [state_means[j] for j in range(n_states) if final_mapping[j] == s]
            new_means[s] = float(np.mean(mapped_old)) if mapped_old else 0.0

    # Ensure ascending sort
    sort_idx = np.argsort(new_means)
    new_means = new_means[sort_idx]
    resort_map = np.empty_like(sort_idx)
    resort_map[sort_idx] = np.arange(len(sort_idx))
    new_state_path = resort_map[new_state_path]

    # Recompute transition matrix from new state path
    new_transmat = np.zeros((new_n_states, new_n_states), dtype=np.float64)
    for t in range(len(new_state_path) - 1):
        s_cur = new_state_path[t]
        s_nxt = new_state_path[t + 1]
        new_transmat[s_cur, s_nxt] += 1.0

    row_sums = new_transmat.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    new_transmat = np.divide(new_transmat, row_sums, out=np.zeros_like(new_transmat), where=row_sums > 0)
    if np.any(zero_rows):
        new_transmat[zero_rows] = 1.0 / new_n_states

    return new_means, new_transmat, new_state_path, logs


def cleanup_classification_result(
    result: ClassificationResult,
    trace: SignalTrace,
    config: ClassificationConfig,
) -> ClassificationResult:
    """Apply configured post-fit segment filtering and state consolidation.

    Parameters
    ----------
    result : ClassificationResult
        Original Viterbi classification result.
    trace : SignalTrace
        Corresponding input trace.
    config : ClassificationConfig
        Configuration detailing postprocessing thresholds.

    Returns
    -------
    ClassificationResult
        Updated ClassificationResult with cleaned states and transitions.
    """
    logs: List[str] = []
    state_means = result.state_means.copy()
    transition_matrix = result.transition_matrix.copy()
    state_path = result.state_path.copy()
    n_states = result.n_states

    # Step 1: Merge nearly identical states if configured
    if config.merge_state_threshold is not None or config.merge_state_sigma_factor is not None:
        state_means, transition_matrix, state_path, merge_logs = merge_nearly_identical_states(
            state_means,
            transition_matrix,
            state_path,
            threshold=config.merge_state_threshold,
            sigma=result.state_sigma,
            sigma_factor=config.merge_state_sigma_factor,
        )
        logs.extend(merge_logs)
        n_states = len(state_means)

    # Step 2: Merge transient minimum dwell segments if configured
    if config.min_dwell_frames > 1:
        prev_path = state_path.copy()
        state_path = merge_minimum_dwell_segments(
            state_path,
            state_means,
            min_dwell_frames=config.min_dwell_frames,
        )
        n_changed = int(np.sum(prev_path != state_path))
        if n_changed > 0:
            msg = f"Merged {n_changed} transient flicker frame(s) with dwell < {config.min_dwell_frames}."
            logs.append(msg)
            logger.info(msg)

    if not logs:
        return result

    # Update dependent attributes
    classified_signal = build_classified_signal(state_path, state_means)
    fraction_spent, transitions_found = compute_transition_stats(
        state_path, n_states, trace.n_frames
    )

    result.n_states = n_states
    result.state_means = state_means
    result.transition_matrix = transition_matrix
    result.state_path = state_path
    result.classified_signal = classified_signal
    result.fraction_spent = fraction_spent
    result.transitions_found = transitions_found
    result.postprocessing_applied.extend(logs)
    for log in logs:
        if log not in result.warnings:
            result.warnings.append(log)

    return result
