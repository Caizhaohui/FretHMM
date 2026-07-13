"""Tests for dwell-time descriptive statistics and exponential fitting.

Covers :mod:`frethmm.core.dwell_stats`:
- :func:`describe_durations` numeric correctness and empty-input handling.
- :func:`fit_exponential_dwell` rate-constant recovery, sample-count guard,
  and non-convergence handling.
- :func:`summarize_events_extended` / :func:`summarize_overall_extended`
  preserve the events-module fields and add the new descriptive columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from frethmm.core.dwell_stats import (
    describe_durations,
    fit_dict_to_columns,
    fit_exponential_dwell,
    summarize_events_extended,
    summarize_overall_extended,
)
from frethmm.core.events import Event


def _event(event_type: str, duration: float, *, source: str = "t.csv", excluded: bool = False) -> Event:
    return Event(
        source_file=source,
        event_label=f"{event_type}_1",
        event_type=event_type,
        event_index=1,
        state_value=0.8 if event_type == "ON" else 0.2,
        start_time=0.0,
        end_time=duration - 1.0,
        duration_seconds=duration,
        start_frame=0,
        end_frame=int(duration),
        excluded=excluded,
        exclude_reason="terminal_off_gte_100s" if excluded else "",
    )


# --- describe_durations -------------------------------------------------


def test_describe_durations_complete_fields():
    stats = describe_durations([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["std"] == round(float(np.std([1, 2, 3, 4, 5], ddof=1)), 6)
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["p25"] == 2.0
    assert stats["p75"] == 4.0
    assert stats["total"] == 15.0


def test_describe_durations_empty_returns_blanks():
    """An empty duration list yields count=0 and blank numeric fields."""
    stats = describe_durations([])
    assert stats["count"] == 0
    for key in ("mean", "median", "std", "min", "max", "p25", "p75", "total"):
        assert stats[key] == ""


def test_describe_durations_single_value_has_zero_std():
    """A single sample has std=0; min=max=median=value."""
    stats = describe_durations([3.0])
    assert stats["count"] == 1
    assert stats["std"] == 0.0
    assert stats["min"] == stats["max"] == stats["median"] == 3.0


# --- fit_exponential_dwell ---------------------------------------------


def test_fit_exponential_recovers_known_rate():
    """Fitting exponential(scale=2) dwell times recovers k ~ 0.5."""
    rng = np.random.default_rng(42)
    durations = list(rng.exponential(scale=2.0, size=2000))
    fit = fit_exponential_dwell(durations)
    assert fit is not None
    assert fit["converged"] is True
    # True k = 1/2 = 0.5; allow generous tolerance for histogram-binned fit.
    assert fit["rate"] == pytest.approx(0.5, rel=0.25)
    assert fit["mean_time"] == pytest.approx(1.0 / fit["rate"], rel=1e-6)


def test_fit_exponential_returns_none_for_few_samples():
    """Fewer than 5 samples cannot support a histogram fit."""
    assert fit_exponential_dwell([1.0, 2.0, 3.0]) is None


def test_fit_exponential_uses_custom_bin_count():
    """The n_bins parameter flows through to the result."""
    rng = np.random.default_rng(0)
    durations = list(rng.exponential(scale=1.0, size=500))
    fit = fit_exponential_dwell(durations, n_bins=25)
    assert fit["n_bins"] == 25


def test_fit_dict_to_columns_handles_none_and_result():
    """A None fit yields blanks; a real fit populates every column."""
    blanks = fit_dict_to_columns(None, prefix="on")
    assert set(blanks.values()) == {""}
    assert all(key.startswith("on_") for key in blanks)

    populated = fit_dict_to_columns(
        {"amplitude": 10.0, "rate": 0.5, "rate_std": 0.05,
         "mean_time": 2.0, "n_events": 100, "n_bins": 20, "converged": True},
        prefix="off",
    )
    assert populated["off_rate_constant"] == 0.5
    assert populated["off_fit_converged"] is True
    assert populated["off_fit_n_bins"] == 20


# --- summarize_*_extended ----------------------------------------------


def test_summarize_events_extended_preserves_base_and_adds_percentiles():
    """Extended per-file summary keeps events.py fields and adds median/std/p25/p75."""
    events = [
        _event("ON", 10.0), _event("OFF", 20.0),
        _event("ON", 30.0), _event("OFF", 40.0),
    ]
    summary = summarize_events_extended("t.csv", events)

    # Base fields from events.summarize_events are still present.
    assert summary["source_file"] == "t.csv"
    assert summary["on_event_count"] == 2
    assert summary["mean_on_time_seconds"] == 20.0
    # New descriptive fields exist with correct unit suffixes.
    assert summary["on_count"] == 2
    assert summary["on_median_seconds"] == 20.0
    assert summary["on_min_seconds"] == 10.0
    assert summary["on_max_seconds"] == 30.0
    assert summary["off_count"] == 2
    assert summary["off_p25_seconds"] == 25.0  # np.percentile([20,40],25)


def test_summarize_events_extended_excludes_excluded_events():
    """Excluded (terminal-off) events are dropped from the descriptive stats."""
    events = [
        _event("ON", 10.0),
        _event("OFF", 200.0, excluded=True),  # excluded terminal off
    ]
    summary = summarize_events_extended("t.csv", events)
    # OFF has no included events -> count 0, blanks.
    assert summary["off_count"] == 0
    assert summary["off_mean_seconds"] == ""
    assert summary["on_count"] == 1


def test_summarize_overall_extended_pools_across_files():
    """Overall extended summary pools ON/OFF durations across all files."""
    events = [
        _event("ON", 10.0, source="a.csv"), _event("ON", 30.0, source="b.csv"),
        _event("OFF", 20.0, source="a.csv"),
    ]
    overall = summarize_overall_extended(events, file_count=2)
    assert overall["file_count"] == 2
    assert overall["on_count"] == 2
    assert overall["on_mean_seconds"] == 20.0
    assert overall["off_count"] == 1
    assert overall["off_median_seconds"] == 20.0


# --- event_details round-trip ------------------------------------------


def test_read_event_details_roundtrips_event_to_detail_row(tmp_path):
    """Writing event_details.csv then reading it back preserves Event fields."""
    import csv

    from frethmm.core.events import DETAIL_FIELDS, event_to_detail_row
    from frethmm.formats.event_details_parser import read_event_details

    original = [
        _event("ON", 12.5, source="trace.csv"),
        _event("OFF", 250.0, source="trace.csv", excluded=True),
    ]
    csv_path = tmp_path / "event_details.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows([event_to_detail_row(e) for e in original])

    rebuilt = read_event_details(csv_path)
    assert len(rebuilt) == 2
    assert rebuilt[0].event_type == "ON"
    assert rebuilt[0].duration_seconds == 12.5
    assert rebuilt[1].excluded is True
    assert rebuilt[1].exclude_reason == "terminal_off_gte_100s"


def test_read_event_details_rejects_missing_columns(tmp_path):
    """A file missing required columns raises ValueError."""
    import pytest

    from frethmm.formats.event_details_parser import read_event_details

    bad = tmp_path / "bad.csv"
    bad.write_text("source_file,event_type\ntrace.csv,ON\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing event-details columns"):
        read_event_details(bad)
