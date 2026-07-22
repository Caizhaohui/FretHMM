"""Dwell-time descriptive statistics and exponential rate-constant fitting.

This module consumes the ON/OFF events produced by :mod:`frethmm.core.events`
and turns them into the deeper statistics that single-molecule biophysics
typically needs: medians, standard deviations, percentiles, and an exponential
fit of the dwell-time distribution from which a rate constant is extracted.

It deliberately leaves :mod:`frethmm.core.events` untouched — ``events`` stays a
lightweight segmentation + count/mean report, while this module layers the
extended analysis on top of the same ``Event`` objects.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from frethmm.core.events import (
    Event,
    included_statistical_events,
    summarize_events,
    summarize_overall,
)

# Sentinel for "no value" in CSV output, matching events.py's blank-cell
# convention for empty ON/OFF groups.
_BLANK = ""


class DwellFit(TypedDict):
    amplitude: float
    rate: float
    rate_std: float
    mean_time: float
    n_events: int
    n_bins: int
    converged: bool


def describe_durations(durations: list[float]) -> dict[str, object]:
    """Full descriptive statistics for a list of dwell durations (seconds).

    Returns a dict with ``count``, ``mean``, ``median``, ``std``, ``min``,
    ``max``, ``p25``, ``p75``, and ``total``. An empty input yields blank
    strings for every numeric field (consistent with the ``mean`` blanking in
    :mod:`frethmm.core.events`), so the result can be written straight to CSV.
    """
    keys = ("count", "mean", "median", "std", "min", "max", "p25", "p75", "total")
    if not durations:
        return {key: (0 if key == "count" else _BLANK) for key in keys}

    arr = np.asarray(durations, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "total": round(float(arr.sum()), 6),
    }


def fit_exponential_dwell(
    durations: list[float],
    *,
    n_bins: Optional[int] = None,
) -> Optional[DwellFit]:
    """Single-exponential fit ``A·exp(-k·t)`` of a dwell-time distribution.

    Mirrors the histogram + ``curve_fit`` pattern already used by
    :func:`frethmm.viz.tdp.fit_gaussian_to_rates`: build a histogram, fit an
    exponential model to the bin centres, and read the rate constant off the
    fit. The rate constant ``k`` is the inverse of the mean dwell time ``tau``.

    Parameters
    ----------
    durations:
        Dwell durations (seconds) for one event type (ON or OFF) pooled across
        the molecules of interest.
    n_bins:
        Histogram bin count. ``None`` picks ``max(10, n // 3)`` to match the
        TDP helper's heuristic.

    Returns
    -------
    dict or None
        ``{"amplitude", "rate", "rate_std", "mean_time", "n_events",
        "n_bins", "converged"}``, or ``None`` when there are too few samples
        (``< 5``) or the fit fails to converge. ``mean_time = 1 / rate``.
    """
    if len(durations) < 5:
        return None

    from scipy.optimize import curve_fit

    arr = np.asarray(durations, dtype=np.float64)
    if n_bins is None:
        n_bins = max(10, int(arr.size // 3))

    hist, bin_edges = np.histogram(arr, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Exponential model in histogram-count units. The amplitude scales to the
    # peak bin count; k is the rate constant (1/s).
    def exp_model(t, amp, k):
        return amp * np.exp(-k * t)

    mean_dwell = float(arr.mean())
    k0 = 1.0 / mean_dwell if mean_dwell > 0 else 1.0
    p0 = [float(hist.max()), k0]

    # Constrain amplitude and rate constant to be non-negative so the fit
    # cannot drift into a growing-exponential (negative k) regime on noisy or
    # weakly-decaying histograms. Without bounds a small-N histogram can fit
    # k < 0, which is non-physical for a dwell-time distribution.
    try:
        popt, _pcov = curve_fit(
            exp_model, bin_centers, hist, p0=p0, maxfev=10000,
            bounds=([0.0, 0.0], (np.inf, np.inf)),
        )
    except (RuntimeError, ValueError):
        return None

    amp, k = float(popt[0]), float(popt[1])
    if k <= 0:  # bounds push k to 0 only when decay is undetectable -> non-fit
        return None
    rate_std = float(np.sqrt(max(_pcov[1, 1], 0.0))) if _pcov.size else 0.0
    return {
        "amplitude": round(amp, 6),
        "rate": round(k, 6),
        "rate_std": round(rate_std, 6),
        "mean_time": round(1.0 / k, 6),
        "n_events": int(arr.size),
        "n_bins": int(n_bins),
        "converged": True,
    }


def _merge_describe(base: dict[str, object], desc: dict[str, object], prefix: str) -> None:
    """Merge a describe_durations result into ``base`` with a ``{prefix}_`` tag.

    ``count`` is unitless, so it is stored as ``{prefix}_count``; every other
    key carries a ``_seconds`` suffix to match the events-module convention.
    """
    for key, value in desc.items():
        if key == "count":
            base[f"{prefix}_count"] = value
        else:
            base[f"{prefix}_{key}_seconds"] = value


def summarize_events_extended(source_file: str, events: list[Event]) -> dict[str, object]:
    """Per-file summary with the full descriptive-stat block per event type.

    Preserves every field from :func:`frethmm.core.events.summarize_events`
    (backward compatible) and appends ``median``/``std``/``min``/``max``/
    ``p25``/``p75`` for ON and OFF dwell times.
    """
    base = summarize_events(source_file, events)
    included = included_statistical_events(events)
    on_durations = [event.duration_seconds for event in included if event.event_type == "ON"]
    off_durations = [event.duration_seconds for event in included if event.event_type == "OFF"]

    _merge_describe(base, describe_durations(on_durations), prefix="on")
    _merge_describe(base, describe_durations(off_durations), prefix="off")
    return base


def summarize_overall_extended(all_events: list[Event], file_count: int) -> dict[str, object]:
    """Cross-file aggregate with the full descriptive-stat block per event type."""
    base = summarize_overall(all_events, file_count)
    included = included_statistical_events(all_events)
    on_durations = [event.duration_seconds for event in included if event.event_type == "ON"]
    off_durations = [event.duration_seconds for event in included if event.event_type == "OFF"]

    _merge_describe(base, describe_durations(on_durations), prefix="on")
    _merge_describe(base, describe_durations(off_durations), prefix="off")
    return base


def fit_dict_to_columns(fit: Optional[DwellFit], *, prefix: str) -> dict[str, object]:
    """Flatten a fit result into ``{prefix}_rate_constant``-style CSV columns.

    A ``None`` fit (too few samples / non-convergence) yields blank strings so
    the output table stays rectangular across runs with and without fitting.
    """
    if fit is None:
        return {
            f"{prefix}_rate_constant": _BLANK,
            f"{prefix}_rate_constant_std": _BLANK,
            f"{prefix}_fit_mean_time": _BLANK,
            f"{prefix}_fit_amplitude": _BLANK,
            f"{prefix}_fit_n_bins": _BLANK,
            f"{prefix}_fit_converged": _BLANK,
        }
    return {
        f"{prefix}_rate_constant": fit["rate"],
        f"{prefix}_rate_constant_std": fit["rate_std"],
        f"{prefix}_fit_mean_time": fit["mean_time"],
        f"{prefix}_fit_amplitude": fit["amplitude"],
        f"{prefix}_fit_n_bins": fit["n_bins"],
        f"{prefix}_fit_converged": fit["converged"],
    }
