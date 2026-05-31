"""Constants, defaults, and validation utilities for pyHaMMy."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np


DEFAULT_MAX_ITER = 500
DEFAULT_TOL = 1e-4
DEFAULT_N_STATES = 2
MIN_FRET_SEPARATION = 0.03
FRET_MIN = 0.0
FRET_MAX = 1.0


@dataclass
class HMMConfig:
    n_states: int = DEFAULT_N_STATES
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    guesses: Optional[List[float]] = None
    workers: int = 1
    data_mode: Literal["auto", "fret", "donor_acceptor", "single_channel"] = "auto"
    signal_column: int = 1

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {self.n_states}")
        if self.guesses is not None:
            if len(self.guesses) != self.n_states:
                raise ValueError(
                    f"Expected {self.n_states} guesses, got {len(self.guesses)}"
                )

    def default_means(self, data_min: float = 0.0, data_max: float = 1.0) -> np.ndarray:
        if self.guesses is not None:
            return np.array(self.guesses)
        return np.linspace(data_min, data_max, self.n_states + 2)[1:-1]


@dataclass
class TraceData:
    time: np.ndarray
    donor: np.ndarray
    acceptor: np.ndarray
    fret: np.ndarray
    observations: np.ndarray
    filepath: Optional[Path] = None
    mode: Literal["fret", "single_channel"] = "fret"

    @property
    def n_frames(self) -> int:
        return len(self.time)

    def __post_init__(self):
        lengths = [len(self.time), len(self.donor), len(self.acceptor),
                   len(self.fret), len(self.observations)]
        if len(set(lengths)) != 1:
            raise ValueError(f"All arrays must have the same length, got {lengths}")


@dataclass
class HMMResult:
    n_states: int
    log_prob: float
    means: np.ndarray
    sigma: float
    signal_sigma: float
    transmat: np.ndarray
    viterbi_path: np.ndarray
    idealized_fret: np.ndarray
    fraction_spent: np.ndarray
    transitions_found: np.ndarray

    filepath: Optional[Path] = None

    @property
    def dwell_times(self) -> np.ndarray:
        from pyhammi.postprocess import extract_dwell_times
        return extract_dwell_times(self)
