"""ON/OFF event extraction from classified state paths.

This module ingests the HMM-classified state path (as written to
``*_classified.csv``) and turns it into discrete ON/OFF *events* suitable for
dwell-time analysis. It is the in-package successor to the standalone
``analyze_on_off_events.py`` script.

State semantics
---------------
A two-state trace treats high fluorescence as ON. A low segment is OFF only
when the trace returns to high fluorescence; a terminal low segment represents
permanent loss of activity and is not emitted. For three or more states, every
non-lowest state defines an independent stage. A descent through one or more
lower states is OFF when it later returns to the original stage, regardless of
duration. The classified HMM path, rather than the raw fluorescence, is the
sole source of event boundaries.

Tail-off exclusion
-------------------
The legacy terminal-OFF threshold remains accepted for command compatibility,
but two-state terminal low segments are omitted rather than recorded and later
excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TypeAlias

import numpy as np
import numpy.typing as npt


FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


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
    stage_state_index: int = -1
    stage_state_mean: float = float("nan")
    off_state_index: int = -1
    event_source_type: str = "normal_on"


def _estimate_dt(times: FloatArray) -> float:
    """Per-frame time step, defaulting to 1.0 for traces shorter than 2 frames."""
    if len(times) < 2:
        return 1.0
    return float(times[1] - times[0])


def _is_on_state(state: int, state_path_value: int) -> bool:
    return int(state_path_value) == int(state)


def included_statistical_events(events: list[Event]) -> list[Event]:
    return [
        event
        for event in events
        if not event.excluded and event.event_type in {"ON", "OFF"}
    ]


def extract_events(
    state_path: IntArray,
    state_means: FloatArray,
    times: FloatArray,
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
        Emission mean of each state index. Two-state data uses the legacy
        highest-mean ON rule; multi-state data is segmented by stage.
    times:
        Per-frame timestamps (seconds).
    source_file:
        File name used to tag each emitted :class:`Event` (metadata only).
    tail_off_threshold_seconds:
        Retained for CLI compatibility. Terminal low segments in two-state
        traces are omitted regardless of this value.
    """
    if len(state_path) == 0:
        return []

    if len(state_means) > 2:
        return _extract_multistage_events(state_path, state_means, times, source_file)

    _ = tail_off_threshold_seconds
    on_state = int(np.argmax(state_means))
    boundaries = np.flatnonzero(np.diff(state_path) != 0) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    stops = np.concatenate((boundaries - 1, np.array([len(state_path) - 1])))
    events: list[Event] = []
    on_event_count = 0
    off_event_count = 0

    for run_index, start_frame in enumerate(starts):
        state_index = int(state_path[start_frame])
        if not _is_on_state(on_state, state_index):
            continue
        on_event_count += 1
        end_frame = int(stops[run_index])
        start_time = float(times[start_frame])
        end_time = float(times[end_frame])
        events.append(
            Event(
                source_file=source_file,
                event_label=f"ON_{on_event_count}",
                event_type="ON",
                event_index=on_event_count,
                state_value=float(state_means[state_index]),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time + _estimate_dt(times),
                start_frame=int(start_frame),
                end_frame=end_frame,
                excluded=False,
                exclude_reason="",
            )
        )
        next_run_index = run_index + 1
        if next_run_index + 1 >= len(starts) or not _is_on_state(on_state, int(state_path[starts[next_run_index + 1]])):
            continue
        off_start_frame = int(starts[next_run_index])
        off_end_frame = int(stops[next_run_index])
        off_event_count += 1
        off_start_time = float(times[off_start_frame])
        off_end_time = float(times[off_end_frame])
        events.append(
            Event(
                source_file=source_file,
                event_label=f"OFF_{off_event_count}",
                event_type="OFF",
                event_index=off_event_count,
                state_value=float(state_means[int(state_path[off_start_frame])]),
                start_time=off_start_time,
                end_time=off_end_time,
                duration_seconds=off_end_time - off_start_time + _estimate_dt(times),
                start_frame=off_start_frame,
                end_frame=off_end_frame,
                excluded=False,
                exclude_reason="",
            )
        )

    return events


def _make_event(
    source_file: str,
    event_type: str,
    event_index: int,
    state_value: float,
    times: FloatArray,
    start_frame: int,
    end_frame: int,
    dt: float,
    stage_state_index: int,
    stage_state_mean: float,
    off_state_index: int,
    event_source_type: str,
) -> Event:
    start_time = float(times[start_frame])
    end_time = float(times[end_frame])
    return Event(
        source_file=source_file,
        event_label=f"stage_{stage_state_index}_{event_type}_{event_index}",
        event_type=event_type,
        event_index=event_index,
        state_value=state_value,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=end_time - start_time + dt,
        start_frame=start_frame,
        end_frame=end_frame,
        excluded=False,
        exclude_reason="",
        stage_state_index=stage_state_index,
        stage_state_mean=stage_state_mean,
        off_state_index=off_state_index,
        event_source_type=event_source_type,
    )


def _extract_multistage_events(
    state_path: IntArray,
    state_means: FloatArray,
    times: FloatArray,
    source_file: str,
) -> list[Event]:
    ordered_states = np.argsort(state_means)
    ranks = np.empty(len(state_means), dtype=np.int64)
    ranks[ordered_states] = np.arange(len(state_means), dtype=np.int64)
    ranked_path = ranks[state_path]
    boundaries = np.flatnonzero(np.diff(ranked_path) != 0) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    stops = np.concatenate((boundaries - 1, np.array([len(state_path) - 1])))
    run_ranks = ranked_path[starts]
    dt = _estimate_dt(times)
    events: list[Event] = []
    counts: dict[tuple[int, str], int] = {}

    def emit(
        event_type: str,
        rank: int,
        start_frame: int,
        end_frame: int,
        source_type: str,
        value_rank: int | None = None,
    ) -> None:
        key = (rank, event_type)
        counts[key] = counts.get(key, 0) + 1
        on_state = int(ordered_states[rank])
        value_state = int(ordered_states[rank if value_rank is None else value_rank])
        events.append(
            _make_event(
                source_file,
                event_type,
                counts[key],
                float(state_means[value_state]),
                times,
                start_frame,
                end_frame,
                dt,
                on_state,
                float(state_means[on_state]),
                int(ordered_states[rank - 1]) if rank > 0 else -1,
                source_type,
            )
        )

    run_index = 0
    while run_index < len(run_ranks):
        rank = int(run_ranks[run_index])
        start_frame = int(starts[run_index])
        end_frame = int(stops[run_index])
        if rank == 0:
            run_index += 1
            continue

        emit("ON", rank, start_frame, end_frame, "normal_on")
        next_run_index = run_index + 1
        if next_run_index == len(run_ranks):
            break

        next_rank = int(run_ranks[next_run_index])
        if next_rank >= rank:
            run_index = next_run_index
            continue

        recovery_run_index = next_run_index
        while recovery_run_index < len(run_ranks) and int(run_ranks[recovery_run_index]) < rank:
            recovery_run_index += 1

        if recovery_run_index < len(run_ranks) and int(run_ranks[recovery_run_index]) == rank:
            emit(
                "OFF",
                rank,
                int(starts[next_run_index]),
                int(stops[recovery_run_index - 1]),
                "normal_off",
                next_rank,
            )
            run_index = recovery_run_index
            continue

        run_index = recovery_run_index if recovery_run_index < len(run_ranks) else next_run_index

    return events


def summarize_events(source_file: str, events: list[Event]) -> dict[str, object]:
    """Per-file summary of ON/OFF event counts and dwell statistics."""
    included = included_statistical_events(events)
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
    included = included_statistical_events(all_events)
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


def summarize_stage_events(source_file: str, events: list[Event]) -> list[dict[str, object]]:
    stage_indices = sorted({event.stage_state_index for event in events if event.stage_state_index >= 0})
    rows: list[dict[str, object]] = []
    for stage_index in stage_indices:
        stage_events = [event for event in events if event.stage_state_index == stage_index]
        row = summarize_events(source_file, stage_events)
        representative = stage_events[0]
        row.update(
            {
                "stage_state_index": stage_index,
                "stage_state_mean": round(representative.stage_state_mean, 6),
                "off_state_index": representative.off_state_index,
            }
        )
        rows.append(row)
    return rows


def summarize_stage_overall(all_events: list[Event], file_count: int) -> list[dict[str, object]]:
    stage_indices = sorted({event.stage_state_index for event in all_events if event.stage_state_index >= 0})
    rows: list[dict[str, object]] = []
    for stage_index in stage_indices:
        stage_events = [event for event in all_events if event.stage_state_index == stage_index]
        row = summarize_overall(stage_events, file_count)
        representative = stage_events[0]
        row.update(
            {
                "stage_state_index": stage_index,
                "stage_state_mean": round(representative.stage_state_mean, 6),
                "off_state_index": representative.off_state_index,
            }
        )
        rows.append(row)
    return rows


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
        "stage_state_index": event.stage_state_index,
        "stage_state_mean": round(event.stage_state_mean, 6),
        "off_state_index": event.off_state_index,
        "event_source_type": event.event_source_type,
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
    "stage_state_index",
    "stage_state_mean",
    "off_state_index",
    "event_source_type",
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

STAGE_SUMMARY_FIELDS = [
    "stage_state_index",
    "stage_state_mean",
    "off_state_index",
    *SUMMARY_FIELDS,
]


def overall_fields(overall: dict[str, object]) -> list[str]:
    """Column order for ``event_stats_overall.csv`` (matches the dict key order)."""
    return list(overall.keys())
