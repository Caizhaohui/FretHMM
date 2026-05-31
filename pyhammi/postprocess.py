"""Post-processing: idealized FRET trajectory, dwell-time extraction."""

import numpy as np

from pyhammi.config import HMMResult


def build_idealized_fret(viterbi_path: np.ndarray, means: np.ndarray) -> np.ndarray:
    return means[viterbi_path]


def extract_dwell_times(result: HMMResult) -> np.ndarray:
    """Extract dwell times from Viterbi path.

    Returns array of [src_state, dst_state, duration] rows.
    Each row represents one transition (dwell in src_state followed by
    a jump to dst_state). The final dwelling segment is NOT included
    since it has no outgoing transition.
    """
    path = result.viterbi_path
    if len(path) < 2:
        return np.empty((0, 3))

    dwells = []
    current_state = int(path[0])
    start_idx = 0

    for i in range(1, len(path)):
        if int(path[i]) != current_state:
            duration = i - start_idx
            next_state = int(path[i])
            dwells.append([current_state, next_state, duration])
            current_state = int(path[i])
            start_idx = i

    return np.array(dwells, dtype=np.float64) if dwells else np.empty((0, 3))


def compute_transition_stats(
    viterbi_path: np.ndarray,
    n_states: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fraction_spent and transitions_found from Viterbi path.

    fraction_spent[i, j] = (total frames in state i that later transition
    to state j) / (total frames in trace).

    transitions_found[i, j] = number of direct i→j transitions.
    """
    fraction_spent = np.zeros((n_states, n_states))
    transitions_found = np.zeros((n_states, n_states), dtype=int)

    if n_frames == 0:
        return fraction_spent, transitions_found

    # Walk the path: record each contiguous dwell segment, then
    # accumulate the correct per-segment duration into fraction_spent.
    current_state = int(viterbi_path[0])
    start_idx = 0

    for i in range(1, len(viterbi_path)):
        state = int(viterbi_path[i])
        if state != current_state:
            duration = i - start_idx
            fraction_spent[current_state, state] += duration
            transitions_found[current_state, state] += 1
            current_state = state
            start_idx = i

    # Normalize by total frame count
    fraction_spent /= n_frames

    return fraction_spent, transitions_found
