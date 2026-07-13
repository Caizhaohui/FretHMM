"""Reverse-parser for ``*_classified.csv`` files.

The classified CSV is FretHMM's primary output (``time, classified_mean``).
This module reconstructs the minimal data needed for downstream analyses —
event extraction, dwell-time plots — without requiring the full
:class:`~frethmm.domain.models.ClassificationResult` (which also carries
log-probabilities and transition matrices that the CSV does not store).

It mirrors :mod:`frethmm.formats.report_parser` in spirit: a pure function
that turns an output artifact back into structured arrays.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import numpy as np


def read_classified_csv(
    filepath: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ``(state_path, state_means, times)`` from a classified CSV.

    Parameters
    ----------
    filepath:
        Path to a ``*_classified.csv`` file with columns ``time`` and
        ``classified_mean``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(state_path, state_means, times)`` where:

        - ``state_means`` — sorted unique ``classified_mean`` values (ascending),
        - ``state_path`` — per-frame state index (each frame mapped to its
          position in ``state_means``),
        - ``times`` — per-frame timestamps.

    Raises
    ------
    ValueError
        If the file is empty or missing the required columns.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    required = {"time", "classified_mean"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"{path.name} must contain columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )

    times = np.array([float(row["time"]) for row in rows], dtype=np.float64)
    classified = np.array(
        [float(row["classified_mean"]) for row in rows], dtype=np.float64
    )

    state_means = np.unique(classified)  # sorted ascending
    # Map each classified value back to its index in state_means. ``searchsorted``
    # with the default side='left' on exact matches yields the correct index.
    state_path = np.searchsorted(state_means, classified).astype(np.int64)
    return state_path, state_means, times
