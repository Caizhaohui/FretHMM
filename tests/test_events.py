"""Tests for ON/OFF event extraction (:mod:`frethmm.core.events`).

Covers:
- 2-state traces (legacy semantics: highest-mean state = ON).
- 3-state generalisation (only the highest-mean state is ON; all others OFF).
- Single-state boundary (the lone state is ON by the argmax rule).
- Tail-OFF exclusion above/below threshold and ON-tail non-exclusion.
- Duration formula (``end_time - start_time + dt``).
- Per-file and overall summarisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.events import (
    extract_events,
    summarize_stage_events,
    summarize_events,
    summarize_overall,
)


def _run(state_path, state_means, dt=1.0, threshold=100.0, source="trace.csv"):
    state_path = np.asarray(state_path, dtype=np.int64)
    state_means = np.asarray(state_means, dtype=np.float64)
    times = np.arange(len(state_path), dtype=np.float64) * dt
    return extract_events(
        state_path, state_means, times, source,
        tail_off_threshold_seconds=threshold,
    )


def test_extract_events_two_state_basic():
    # state 0 (mean 0.2) = OFF, state 1 (mean 0.8) = ON
    events = _run([0, 0, 1, 1, 0, 0, 1, 1], [0.2, 0.8])

    assert len(events) == 3
    assert [(e.event_type, e.event_index) for e in events] == [
        ("ON", 1), ("OFF", 1), ("ON", 2),
    ]
    # Each ON event carries the high mean; each OFF the low mean.
    for e in events:
        if e.event_type == "ON":
            assert e.state_value == 0.8
        else:
            assert e.state_value == 0.2
    # None excluded (all short).
    assert all(not e.excluded for e in events)


def test_extract_events_two_state_omits_terminal_low_without_recovery():
    # When the higher-indexed state has the LOWER mean, ON still follows the mean.
    # state 0 (mean 0.9) = ON, state 1 (mean 0.1) = OFF
    events = _run([0, 0, 1, 1], [0.9, 0.1])
    assert [e.event_type for e in events] == ["ON"]
    assert events[0].state_value == 0.9


def test_extract_events_three_state_records_adjacent_high_stage_off_return():
    events = _run([2, 2, 1, 1, 2, 2], [0.2, 0.5, 0.9])

    assert [(event.event_type, event.event_source_type) for event in events] == [
        ("ON", "normal_on"),
        ("OFF", "normal_off"),
        ("ON", "normal_on"),
    ]
    assert [(event.start_frame, event.end_frame) for event in events] == [(0, 1), (2, 3), (4, 5)]


def test_extract_events_three_state_tracks_each_adjacent_stage_independently():
    high_events = _run([2, 2, 2, 1, 1, 2, 2], [0.2, 0.5, 0.9])
    middle_events = _run([1, 1, 0, 0, 1, 1], [0.2, 0.5, 0.9])

    stage_two = [event for event in high_events if event.stage_state_index == 2]
    stage_one = [event for event in middle_events if event.stage_state_index == 1]
    assert [(event.event_type, event.duration_seconds) for event in stage_two] == [
        ("ON", 3.0), ("OFF", 2.0), ("ON", 2.0),
    ]
    assert [(event.event_type, event.duration_seconds) for event in stage_one] == [
        ("ON", 2.0), ("OFF", 2.0), ("ON", 2.0),
    ]
    assert all(event.off_state_index == event.stage_state_index - 1 for event in high_events + middle_events)


@pytest.mark.parametrize("off_frames", [1, 2, 3, 4, 10])
def test_extract_events_three_state_counts_any_recovery_duration_as_off(off_frames):
    events = _run([2, 2] + [1] * off_frames + [2, 2], [0.2, 0.5, 0.9])

    assert [(event.stage_state_index, event.event_type) for event in events] == [
        (2, "ON"),
        (2, "OFF"),
        (2, "ON"),
    ]
    assert events[1].start_frame == 2
    assert events[1].end_frame == off_frames + 1


def test_extract_events_three_state_counts_multi_low_state_recovery_as_one_off():
    events = _run([2, 2, 1, 1, 0, 0, 2, 2], [0.2, 0.5, 0.9])

    assert [(event.stage_state_index, event.event_type, event.start_frame, event.end_frame) for event in events] == [
        (2, "ON", 0, 1),
        (2, "OFF", 2, 5),
        (2, "ON", 6, 7),
    ]


def test_extract_events_three_state_starts_lower_stage_without_transition_audit_row():
    events = _run([2, 2, 1, 1, 1, 1], [0.2, 0.5, 0.9])

    assert [(event.stage_state_index, event.event_type, event.start_frame, event.end_frame) for event in events] == [
        (2, "ON", 0, 1),
        (1, "ON", 2, 5),
    ]
    high_stage = next(row for row in summarize_stage_events("trace.csv", events) if row["stage_state_index"] == 2)
    assert high_stage["included_on_event_count"] == 1
    assert high_stage["included_off_event_count"] == 0


def test_extract_events_three_state_stable_drop_followed_by_low_stage_has_no_audit_rows():
    events = _run([2, 2, 1, 1, 0, 0], [0.2, 0.5, 0.9])

    assert [(event.stage_state_index, event.event_type, event.start_frame, event.end_frame) for event in events] == [
        (2, "ON", 0, 1),
        (1, "ON", 2, 3),
    ]


def test_extract_events_marks_jump_down_without_recovery_audit_row():
    events = _run([2, 2, 0, 0, 1, 1], [0.2, 0.5, 0.9])

    assert [(event.event_type, event.event_source_type) for event in events] == [
        ("ON", "normal_on"),
        ("ON", "normal_on"),
    ]
    summary = summarize_stage_events("trace.csv", events)
    high_stage = next(row for row in summary if row["stage_state_index"] == 2)
    assert high_stage["included_on_event_count"] == 1
    assert high_stage["included_off_event_count"] == 0


def test_extract_events_single_state_all_on():
    """A single-state trace: the lone state is the highest, so one ON event."""
    events = _run([0, 0, 0, 0], [0.5])
    assert len(events) == 1
    assert events[0].event_type == "ON"
    assert events[0].excluded is False  # ON tail is never excluded


def test_extract_events_empty_trace():
    """An empty state path yields no events."""
    events = extract_events(
        np.array([], dtype=np.int64),
        np.array([0.2, 0.8]),
        np.array([], dtype=np.float64),
        "empty.csv",
    )
    assert events == []


def test_terminal_low_is_not_an_off_event_regardless_of_duration():
    path = [1, 1] + [0] * 200
    events = _run(path, [0.2, 0.8], threshold=100.0)

    assert [(event.event_type, event.duration_seconds) for event in events] == [("ON", 2.0)]


def test_terminal_on_not_excluded_even_if_long():
    """A long final ON run is never excluded (only OFF tails are)."""
    path = [0] * 5 + [1] * 200  # ends in a 200s ON run
    events = _run(path, [0.2, 0.8], threshold=100.0)

    assert events[-1].event_type == "ON"
    assert events[-1].excluded is False


def test_event_duration_includes_dt():
    """duration = end_time - start_time + dt (matches the original script)."""
    # dt = 0.5: a 4-frame segment spans times [0, 0.5, 1.0, 1.5] => 1.5 - 0 + 0.5 = 2.0
    events = _run([0, 0, 0, 0], [0.5], dt=0.5)
    assert events[0].duration_seconds == 2.0
    assert events[0].start_frame == 0
    assert events[0].end_frame == 3


def test_summarize_events_per_file_fields():
    """Per-file summary counts ON/OFF events and aggregates dwell times."""
    events = _run([0, 0, 1, 1, 0, 0, 1, 1], [0.2, 0.8], source="t.csv")
    summary = summarize_events("t.csv", events)

    assert summary["source_file"] == "t.csv"
    assert summary["on_event_count"] == 2
    assert summary["off_event_count"] == 1
    assert summary["included_on_event_count"] == 2
    assert summary["included_off_event_count"] == 1
    # Each 2-frame segment (dt=1.0): end_time(1.0) - start_time(0.0) + dt(1.0) = 2.0s.
    assert summary["total_on_time_seconds"] == 4.0
    assert summary["total_off_time_seconds"] == 2.0
    assert summary["mean_on_time_seconds"] == 2.0
    assert summary["mean_off_time_seconds"] == 2.0
    assert summary["last_event_type"] == "ON"
    assert summary["tail_off_excluded"] is False


def test_summarize_events_omits_terminal_low_from_totals():
    path = [1, 1] + [0] * 200
    events = _run(path, [0.2, 0.8], threshold=100.0, source="t.csv")
    summary = summarize_events("t.csv", events)

    assert summary["tail_off_excluded"] is False
    assert summary["off_event_count"] == 0
    assert summary["included_off_event_count"] == 0
    # Only the 2-frame ON segment contributes to ON time.
    assert summary["total_on_time_seconds"] == 2.0
    assert summary["total_off_time_seconds"] == 0.0


def test_summarize_overall_aggregates_across_files():
    """The overall summary aggregates events from multiple files."""
    ev1 = _run([0, 0, 1, 1], [0.2, 0.8], source="a.csv")
    ev2 = _run([1, 1, 0, 0], [0.2, 0.8], source="b.csv")
    all_events = ev1 + ev2
    overall = summarize_overall(all_events, file_count=2)

    assert overall["file_count"] == 2
    assert overall["event_count"] == 2
    assert overall["included_event_count"] == 2
    assert overall["included_on_event_count"] == 2
    assert overall["included_off_event_count"] == 0
    assert overall["total_on_time_seconds"] == 4.0  # 2 files * 2s ON each
    assert overall["total_off_time_seconds"] == 0.0
