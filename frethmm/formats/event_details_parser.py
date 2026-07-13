"""Reverse-parser for ``event_details.csv``.

``event_details.csv`` is the per-event output of ``frethmm events``. This
module reconstructs the :class:`~frethmm.core.events.Event` list so downstream
analyses (e.g. ``frethmm dwell-stats``) can consume the events command's
output without re-running HMM classification or event segmentation.

Mirrors :mod:`frethmm.formats.classified_parser` and
:mod:`frethmm.formats.report_parser`: a pure function that turns an output
artifact back into structured data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from frethmm.core.events import DETAIL_FIELDS, Event


def _to_int(value: str) -> int:
    return int(float(value))


def read_event_details(filepath: Union[str, Path]) -> list[Event]:
    """Rebuild an :class:`Event` list from an ``event_details.csv`` file.

    The file must carry the columns defined by
    :data:`frethmm.core.events.DETAIL_FIELDS`. Missing columns raise
    ``ValueError`` so callers fail fast on a malformed input.

    The ``excluded`` and ``exclude_reason`` fields are preserved verbatim, so a
    downstream consumer can decide whether to pool excluded (terminal-off)
    events into its statistics.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required = set(DETAIL_FIELDS)
    present = set(reader.fieldnames or [])
    missing = required - present
    if missing:
        raise ValueError(
            f"{path.name} is missing event-details columns: {sorted(missing)}"
        )

    events: list[Event] = []
    for row in rows:
        events.append(
            Event(
                source_file=row["source_file"],
                event_label=row["event_label"],
                event_type=row["event_type"],
                event_index=_to_int(row["event_index"]),
                state_value=float(row["state_value"]),
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                duration_seconds=float(row["duration_seconds"]),
                start_frame=_to_int(row["start_frame"]),
                end_frame=_to_int(row["end_frame"]),
                excluded=row["excluded"].strip().lower() in {"true", "1"},
                exclude_reason=row["exclude_reason"],
            )
        )
    return events
