from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np


DEFAULT_MAX_ITER = 500
DEFAULT_TOL = 1e-4
DEFAULT_N_STATES = 2


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
    n_states: int = DEFAULT_N_STATES
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    guesses: Optional[list[float]] = None
    workers: int = 1
    data_mode: Literal["auto", "paired_channel", "single_channel"] = "auto"
    signal_column: int = 1
    low_state_tail_trim_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {self.n_states}")
        if self.guesses is not None and len(self.guesses) != self.n_states:
            raise ValueError(
                f"Expected {self.n_states} guesses, got {len(self.guesses)}"
            )
        if (
            self.low_state_tail_trim_seconds is not None
            and self.low_state_tail_trim_seconds <= 0
        ):
            raise ValueError(
                "low_state_tail_trim_seconds must be > 0 when provided, "
                f"got {self.low_state_tail_trim_seconds}"
            )

    def default_state_means(
        self,
        data_min: float = 0.0,
        data_max: float = 1.0,
    ) -> np.ndarray:
        if self.guesses is not None:
            return np.array(self.guesses, dtype=np.float64)
        return np.linspace(data_min, data_max, self.n_states + 2)[1:-1]

    def default_means(
        self,
        data_min: float = 0.0,
        data_max: float = 1.0,
    ) -> np.ndarray:
        return self.default_state_means(data_min, data_max)

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

    @property
    def dwell_segments(self) -> np.ndarray:
        from frethmm.core.postprocess import extract_dwell_segments

        return extract_dwell_segments(self)
