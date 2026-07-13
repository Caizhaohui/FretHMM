"""End-to-end CLI tests for the ``frethmm dwell-stats`` subcommand.

Self-contained: synthesise a multi-event trace, run ``frethmm events`` to
produce ``event_details.csv``, then run ``frethmm dwell-stats`` on it and
assert on the two output tables. No external sample dependency.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from frethmm.core.io import write_classified_csv
from frethmm.core.model import fit_signal_hmm
from frethmm.domain.models import ClassificationConfig
from tests._synthetic import make_synthetic_trace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _generate_events_csv(tmp_path: Path) -> Path:
    """Synthesise a trace, fit it, run ``events``, return the event_details path."""
    # Sample dwell durations from an exponential so the dwell-time distribution
    # the fit targets is genuinely exponential (constant or uniform dwell times
    # would make the fit degenerate or non-converging).
    rng = np.random.default_rng(7)
    pattern = [0.2, 0.8] * 25  # 25 ON + 25 OFF alternations
    durations = [max(15, int(rng.exponential(scale=120.0))) for _ in pattern]
    trace = make_synthetic_trace(pattern, durations, noise=0.05, seed=3)
    result = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=3))
    result.filepath = tmp_path / "synthetic.csv"
    write_classified_csv(trace, result, tmp_path)

    events_dir = tmp_path / "events"
    subprocess.run(
        [
            sys.executable, "-m", "frethmm.app.cli", "events",
            "--files", str(tmp_path / "synthetic_classified.csv"),
            "--output-dir", str(events_dir),
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return events_dir / "event_details.csv"


def _run_dwell_stats(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "frethmm.app.cli", "dwell-stats", *args],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cli_dwell_stats_consumes_events_output(tmp_path):
    """dwell-stats reads event_details.csv and writes summary + per-file tables."""
    details = _generate_events_csv(tmp_path)
    out_dir = tmp_path / "stats"
    completed = _run_dwell_stats([
        "--input", str(details),
        "--output-dir", str(out_dir),
    ])

    summary_path = out_dir / "dwell_stats_summary.csv"
    per_file_path = out_dir / "dwell_stats_per_file.csv"
    assert summary_path.exists()
    assert per_file_path.exists()

    summary = _read_csv(summary_path)
    assert len(summary) == 1
    row = summary[0]
    # Descriptive columns populated for both ON and OFF.
    assert row["on_count"] and int(row["on_count"]) > 0
    assert row["off_count"] and int(row["off_count"]) > 0
    assert row["on_median_seconds"]
    assert row["off_std_seconds"]
    # With ~30 events the exponential fit should converge and populate k.
    assert "on_rate_constant" in row
    assert row["on_rate_constant"] != ""
    assert "Rate constants:" in completed.stdout


def test_cli_dwell_stats_no_fit_blanks_rate_columns(tmp_path):
    """--no-fit leaves the rate-constant columns blank."""
    details = _generate_events_csv(tmp_path)
    out_dir = tmp_path / "stats_nofit"
    _run_dwell_stats([
        "--input", str(details),
        "--output-dir", str(out_dir),
        "--no-fit",
    ])

    row = _read_csv(out_dir / "dwell_stats_summary.csv")[0]
    # Descriptive stats still present...
    assert row["on_mean_seconds"] != ""
    assert row["off_median_seconds"] != ""
    # ...but fit columns are blank.
    assert row["on_rate_constant"] == ""
    assert row["off_rate_constant"] == ""
    assert row["on_fit_converged"] == ""


def test_cli_dwell_stats_per_file_one_row_per_source(tmp_path):
    """dwell_stats_per_file.csv has one row per distinct source file."""
    details = _generate_events_csv(tmp_path)
    out_dir = tmp_path / "stats_pf"
    _run_dwell_stats(["--input", str(details), "--output-dir", str(out_dir)])

    per_file = _read_csv(out_dir / "dwell_stats_per_file.csv")
    assert len(per_file) == 1  # single synthetic source file
    assert per_file[0]["source_file"] == "synthetic_classified.csv"
