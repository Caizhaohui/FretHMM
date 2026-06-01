"""Post-processing for generic signal classification workflows."""

from __future__ import annotations

import numpy as np

from frethmm.domain.models import ClassificationResult


def build_classified_signal(
    state_path: np.ndarray,
    state_means: np.ndarray,
) -> np.ndarray:
    return state_means[state_path]


def extract_dwell_segments(result: ClassificationResult) -> np.ndarray:
    path = result.state_path
    if len(path) < 2:
        return np.empty((0, 3))

    dwells = []
    current_state = int(path[0])
    start_idx = 0

    for i in range(1, len(path)):
        next_state = int(path[i])
        if next_state != current_state:
            dwells.append([current_state, next_state, i - start_idx])
            current_state = next_state
            start_idx = i

    return np.array(dwells, dtype=np.float64) if dwells else np.empty((0, 3))


def compute_transition_stats(
    state_path: np.ndarray,
    n_states: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    fraction_spent = np.zeros((n_states, n_states))
    transitions_found = np.zeros((n_states, n_states), dtype=int)

    if n_frames == 0:
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
