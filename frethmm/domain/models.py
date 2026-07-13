from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np


DEFAULT_MAX_ITER = 500
DEFAULT_TOL = 1e-4
DEFAULT_N_STATES = 2
DEFAULT_N_INIT = 10
DEFAULT_MIN_STATES = 2
DEFAULT_MAX_STATES = 6

# Sentinel accepted by ``ClassificationConfig.n_states`` to request BIC-based
# state-count selection. Kept as a module constant so callers can compare
# without relying on a magic string.
AUTO_STATES = "auto"
StateCount = Union[int, Literal["auto"]]


@dataclass
class ExportOptions:
    classified_csv: bool = True
    summary_json: bool = True
    state_report: bool = True
    state_path: bool = True
    dwell_report: bool = True

    @classmethod
    def classified_only(cls) -> "ExportOptions":
        return cls(
            classified_csv=True,
            summary_json=False,
            state_report=False,
            state_path=False,
            dwell_report=False,
        )


@dataclass
class ClassificationConfig:
    n_states: StateCount = DEFAULT_N_STATES
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    guesses: Optional[list[float]] = None
    workers: int = 1
    data_mode: Literal["auto", "paired_channel", "single_channel"] = "auto"
    signal_column: int = 1
    low_state_tail_trim_seconds: Optional[float] = None
    # Multi-start fitting: run Baum-Welch ``n_init`` times from different
    # initial means and keep the best log-likelihood. ``1`` reproduces the
    # historical single-fit behaviour exactly.
    n_init: int = DEFAULT_N_INIT
    # BIC model-selection range, consulted only when ``n_states == "auto"``.
    min_states: int = DEFAULT_MIN_STATES
    max_states: int = DEFAULT_MAX_STATES

    def __post_init__(self) -> None:
        # Allow either a positive int or the "auto" sentinel.
        if self.n_states != AUTO_STATES:
            if not isinstance(self.n_states, int) or isinstance(self.n_states, bool):
                raise TypeError(
                    f"n_states must be a positive int or 'auto', got {self.n_states!r}"
                )
            if self.n_states < 1:
                raise ValueError(f"n_states must be >= 1, got {self.n_states}")
            if self.guesses is not None and len(self.guesses) != self.n_states:
                raise ValueError(
                    f"Expected {self.n_states} guesses, got {len(self.guesses)}"
                )
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}")
        if self.min_states < 1:
            raise ValueError(f"min_states must be >= 1, got {self.min_states}")
        if self.max_states < self.min_states:
            raise ValueError(
                "max_states must be >= min_states, "
                f"got max_states={self.max_states}, min_states={self.min_states}"
            )
        if (
            self.low_state_tail_trim_seconds is not None
            and self.low_state_tail_trim_seconds <= 0
        ):
            raise ValueError(
                "low_state_tail_trim_seconds must be > 0 when provided, "
                f"got {self.low_state_tail_trim_seconds}"
            )

    @property
    def is_auto_states(self) -> bool:
        return self.n_states == AUTO_STATES

    def default_state_means(
        self,
        data_min: float = 0.0,
        data_max: float = 1.0,
    ) -> np.ndarray:
        if self.guesses is not None:
            return np.array(self.guesses, dtype=np.float64)
        n_states = self._resolved_n_states()
        return np.linspace(data_min, data_max, n_states + 2)[1:-1]

    def default_means(
        self,
        data_min: float = 0.0,
        data_max: float = 1.0,
    ) -> np.ndarray:
        return self.default_state_means(data_min, data_max)

    def _resolved_n_states(self) -> int:
        """Return an int state count, defaulting to ``min_states`` when auto."""
        if self.n_states == AUTO_STATES:
            return self.min_states
        return int(self.n_states)

@dataclass
class SignalTrace:
    time: np.ndarray
    signal: np.ndarray
    observations: np.ndarray
    filepath: Optional[Path] = None
    mode: Literal["single_channel", "paired_channel"] = "single_channel"
    channel_1: Optional[np.ndarray] = None
    channel_2: Optional[np.ndarray] = None
    derived_signal: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        return len(self.time)

    def __post_init__(self) -> None:
        arrays = [self.time, self.signal, self.observations]
        if self.channel_1 is not None:
            arrays.append(self.channel_1)
        if self.channel_2 is not None:
            arrays.append(self.channel_2)
        if self.derived_signal is not None:
            arrays.append(self.derived_signal)
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(f"All arrays must have the same length, got {lengths}")


@dataclass
class ClassificationResult:
    n_states: int
    log_prob: float
    state_means: np.ndarray
    state_sigma: float
    signal_sigma: float
    transition_matrix: np.ndarray
    state_path: np.ndarray
    classified_signal: np.ndarray
    fraction_spent: np.ndarray
    transitions_found: np.ndarray
    filepath: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)
    trace_time: Optional[np.ndarray] = None
    trace_signal: Optional[np.ndarray] = None
    low_state_tail_trim_seconds: Optional[float] = None
    low_state_tail_cutoff_time: Optional[float] = None
    low_state_tail_kept_frames: Optional[int] = None
    # Multi-start metadata (populated by the multi-start fitter).
    n_init: Optional[int] = None
    n_init_used: Optional[int] = None
    best_start_index: Optional[int] = None
    # Model-selection metadata (populated only when ``--states auto``).
    bic: Optional[float] = None
    aic: Optional[float] = None
    model_candidates: Optional[list[dict]] = None

    @property
    def dwell_segments(self) -> np.ndarray:
        from frethmm.core.postprocess import extract_dwell_segments

        return extract_dwell_segments(self)
