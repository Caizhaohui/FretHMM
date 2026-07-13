"""ON/OFF event extraction from classified state paths.

This module ingests the HMM-classified state path (as written to
``*_classified.csv``) and turns it into discrete ON/OFF *events* suitable for
dwell-time analysis. It is the in-package successor to the standalone
``analyze_on_off_events.py`` script.

State semantics
---------------
A trace may have any number of states. Exactly one state is labelled **ON**:
the state with the highest mean (``argmax(state_means)``). Every other state —
no matter how many — is labelled **OFF**. For a 2-state trace this reproduces
the legacy "high value = ON" rule exactly; for 3+ states it generalises it so
that, e.g., a 3-state FRET trace with means ``[0.2, 0.5, 0.9]`` treats only the
``0.9`` state as ON and both lower states as OFF.

Tail-off exclusion
-------------------
Photobleached tails typically appear as a long final OFF run. Mirroring the
original script, the *last* event is flagged ``excluded=True`` when it is OFF
and lasts at least ``tail_off_threshold_seconds``. The event is still listed in
the output (for transparency) but excluded from the ON/OFF summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional

import numpy as np


@dataclass
class Event:
    """A single contiguous ON or OFF segment from a classified trace."""

    source_file: str
    event_label: str
    event_type: str  # "ON" or "OFF"
    event_index: int
    state_value: float
    start_time: float
    end_time: float
    duration_seconds: float
    start_frame: int
    end_frame: int
    excluded: bool
    exclude_reason: str


def _estimate_dt(times: np.ndarray) -> float:
    """Per-frame time step, defaulting to 1.0 for traces shorter than 2 frames."""
    if len(times) < 2:
        return 1.0
    return float(times[1] - times[0])


def _is_on_state(state: int, state_path_value: int) -> bool:
    return int(state_path_value) == int(state)


def extract_events(
    state_path: np.ndarray,
    state_means: np.ndarray,
    times: np.ndarray,
    source_file: str,
    *,
    tail_off_threshold_seconds: float = 100.0,
) -> list[Event]:
    """Detect ON/OFF events from a classified state path.

    Parameters
    ----------
    state_path:
        Per-frame state index, as produced by the HMM Viterbi decode.
    state_means:
        Emission mean of each state index. The highest-mean state is treated as
        ON; all other states are OFF.
    times:
        Per-frame timestamps (seconds).
    source_file:
        File name used to tag each emitted :class:`Event` (metadata only).
    tail_off_threshold_seconds:
        If the final event is OFF and lasts at least this long, it is flagged
        ``excluded`` (still listed, but omitted from summary statistics).
    """
    if len(state_path) == 0:
        return []

    on_state = int(np.argmax(state_means))
    dt = _estimate_dt(times)
    type_counts = {"ON": 0, "OFF": 0}
    events: list[Event] = []

    start_index = 0
    current_value = int(state_path[0])
    for index in range(1, len(state_path) + 1):
        boundary = index == len(state_path) or int(state_path[index]) != current_value
        if not boundary:
            continue

        end_index = index - 1
        event_type = "ON" if _is_on_state(on_state, current_value) else "OFF"
        type_counts[event_type] += 1
        start_time = float(times[start_index])
        end_time = float(times[end_index])
        duration_seconds = end_time - start_time + dt
        events.append(
            Event(
                source_file=source_file,
                event_label=f"{event_type}_{type_counts[event_type]}",
                event_type=event_type,
                event_index=type_counts[event_type],
                state_value=float(state_means[current_value]),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                start_frame=start_index,
                end_frame=end_index,
                excluded=False,
                exclude_reason="",
            )
        )
        if index < len(state_path):
            start_index = index
            current_value = int(state_path[index])

    if events:
        last_event = events[-1]
        if last_event.event_type == "OFF" and last_event.duration_seconds >= tail_off_threshold_seconds:
            last_event.excluded = True
            last_event.exclude_reason = f"terminal_off_gte_{tail_off_threshold_seconds:g}s"

    return events


def summarize_events(source_file: str, events: list[Event]) -> dict[str, object]:
    """Per-file summary of ON/OFF event counts and dwell statistics."""
    included = [event for event in events if not event.excluded]
    included_on = [event for event in included if event.event_type == "ON"]
    included_off = [event for event in included if event.event_type == "OFF"]
    all_on = [event for event in events if event.event_type == "ON"]
    all_off = [event for event in events if event.event_type == "OFF"]
    last_event = events[-1] if events else None

    return {
        "source_file": source_file,
        "on_event_count": len(all_on),
        "off_event_count": len(all_off),
        "included_on_event_count": len(included_on),
        "included_off_event_count": len(included_off),
        "total_on_time_seconds": round(sum(event.duration_seconds for event in included_on), 6),
        "total_off_time_seconds": round(sum(event.duration_seconds for event in included_off), 6),
        "mean_on_time_seconds": round(mean(event.duration_seconds for event in included_on), 6) if included_on else "",
        "mean_off_time_seconds": round(mean(event.duration_seconds for event in included_off), 6) if included_off else "",
        "last_event_type": last_event.event_type if last_event else "",
        "last_event_duration_seconds": round(last_event.duration_seconds, 6) if last_event else "",
        "tail_off_excluded": bool(last_event.excluded) if last_event else False,
    }


def summarize_overall(all_events: list[Event], file_count: int) -> dict[str, object]:
    """Aggregate ON/OFF statistics across all processed files."""
    included = [event for event in all_events if not event.excluded]
    included_on = [event for event in included if event.event_type == "ON"]
    included_off = [event for event in included if event.event_type == "OFF"]
    excluded = [event for event in all_events if event.excluded]
    return {
        "file_count": file_count,
        "event_count": len(all_events),
        "included_event_count": len(included),
        "excluded_event_count": len(excluded),
        "included_on_event_count": len(included_on),
        "included_off_event_count": len(included_off),
        "total_on_time_seconds": round(sum(event.duration_seconds for event in included_on), 6),
        "total_off_time_seconds": round(sum(event.duration_seconds for event in included_off), 6),
        "mean_on_time_seconds": round(mean(event.duration_seconds for event in included_on), 6) if included_on else "",
        "mean_off_time_seconds": round(mean(event.duration_seconds for event in included_off), 6) if included_off else "",
    }


def event_to_detail_row(event: Event) -> dict[str, object]:
    """Flatten an :class:`Event` into the ``event_details.csv`` row schema."""
    return {
        "source_file": event.source_file,
        "event_label": event.event_label,
        "event_type": event.event_type,
        "event_index": event.event_index,
        "state_value": round(event.state_value, 6),
        "start_time": round(event.start_time, 6),
        "end_time": round(event.end_time, 6),
        "duration_seconds": round(event.duration_seconds, 6),
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "excluded": event.excluded,
        "exclude_reason": event.exclude_reason,
    }


DETAIL_FIELDS = [
    "source_file",
    "event_label",
    "event_type",
    "event_index",
    "state_value",
    "start_time",
    "end_time",
    "duration_seconds",
    "start_frame",
    "end_frame",
    "excluded",
    "exclude_reason",
]

SUMMARY_FIELDS = [
    "source_file",
    "on_event_count",
    "off_event_count",
    "included_on_event_count",
    "included_off_event_count",
    "total_on_time_seconds",
    "total_off_time_seconds",
    "mean_on_time_seconds",
    "mean_off_time_seconds",
    "last_event_type",
    "last_event_duration_seconds",
    "tail_off_excluded",
]


def overall_fields(overall: dict[str, object]) -> list[str]:
    """Column order for ``event_stats_overall.csv`` (matches the dict key order)."""
    return list(overall.keys())
