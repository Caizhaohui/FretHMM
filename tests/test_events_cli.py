"""End-to-end CLI tests for the ``frethmm events`` subcommand.

These tests are self-contained: they synthesise a trace, fit it, write a
``*_classified.csv``, then invoke ``frethmm events`` via subprocess and assert
on the three output tables. They do not depend on the optional ``Values1.csv``
sample, so they always run (unlike the golden CLI tests).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from frethmm.core.io import write_classified_csv
from frethmm.core.model import fit_signal_hmm
from frethmm.domain.models import ClassificationConfig
from tests._synthetic import make_synthetic_trace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_classified(tmp_path: Path, *, n_init: int = 1) -> Path:
    """Generate a real ``*_classified.csv`` from a synthetic 2-state trace."""
    trace = make_synthetic_trace(
        means=[0.2, 0.8, 0.2, 0.8],
        durations=[300, 300, 300, 300],
        noise=0.05,
        seed=1,
    )
    result = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=n_init))
    result.filepath = tmp_path / "synthetic.csv"
    write_classified_csv(trace, result, tmp_path)
    return tmp_path / "synthetic_classified.csv"


def _run_events_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "frethmm.app.cli", "events", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cli_events_input_dir_writes_three_tables(tmp_path):
    """``--input-dir`` produces all three output tables with ON/OFF rows."""
    classified = _make_classified(tmp_path)
    out_dir = tmp_path / "events"
    completed = _run_events_cli([
        "--input-dir", str(tmp_path),
        "--output-dir", str(out_dir),
    ])

    assert "Processed 1 file(s)." in completed.stdout
    details = _read_csv(out_dir / "event_details.csv")
    summary = _read_csv(out_dir / "event_summary.csv")
    overall = _read_csv(out_dir / "event_stats_overall.csv")

    # The synthetic trace alternates OFF/ON/OFF/ON.
    types = [row["event_type"] for row in details]
    assert "ON" in types and "OFF" in types
    # Per-file summary has exactly one row.
    assert len(summary) == 1
    assert summary[0]["source_file"] == classified.name
    # Overall aggregate present.
    assert len(overall) == 1
    assert int(overall[0]["file_count"]) == 1
    manifests = list(out_dir.glob("frethmm_run_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["command"] == "events"
    assert manifest["inputs"][0]["name"] == classified.name
    assert {Path(output["path"]).name for output in manifest["outputs"]} == {
        "event_details.csv",
        "event_summary.csv",
        "event_stats_overall.csv",
    }


def test_cli_events_files_mode(tmp_path):
    """``--files`` mode works the same as ``--input-dir``."""
    _make_classified(tmp_path)
    out_dir = tmp_path / "events"
    classified_path = tmp_path / "synthetic_classified.csv"
    _run_events_cli([
        "--files", str(classified_path),
        "--output-dir", str(out_dir),
    ])

    assert (out_dir / "event_details.csv").exists()
    assert (out_dir / "event_summary.csv").exists()
    assert (out_dir / "event_stats_overall.csv").exists()


def test_cli_events_tail_off_threshold(tmp_path):
    """A high threshold keeps the final OFF; a low threshold excludes it."""
    # Build a trace that ends in a long OFF run.
    trace = make_synthetic_trace(
        means=[0.8, 0.2], durations=[50, 400], noise=0.05, seed=2,
    )
    result = fit_signal_hmm(trace, ClassificationConfig(n_states=2, n_init=1))
    result.filepath = tmp_path / "tail.csv"
    write_classified_csv(trace, result, tmp_path)
    classified = tmp_path / "tail_classified.csv"

    # High threshold (1000s): the 400s OFF tail is NOT excluded.
    out_keep = tmp_path / "keep"
    _run_events_cli([
        "--files", str(classified),
        "--output-dir", str(out_keep),
        "--tail-off-threshold-seconds", "1000",
    ])
    summary_keep = _read_csv(out_keep / "event_summary.csv")
    assert summary_keep[0]["tail_off_excluded"] == "False"

    # Low threshold (10s): the 400s OFF tail IS excluded.
    out_exclude = tmp_path / "exclude"
    _run_events_cli([
        "--files", str(classified),
        "--output-dir", str(out_exclude),
        "--tail-off-threshold-seconds", "10",
    ])
    summary_exclude = _read_csv(out_exclude / "event_summary.csv")
    assert summary_exclude[0]["tail_off_excluded"] == "True"


def test_cli_events_and_dwell_stats_write_rows_per_multistage_phase(tmp_path):
    classified = tmp_path / "three_classified.csv"
    with classified.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "classified_mean"])
        writer.writeheader()
        for frame, value in enumerate([0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5]):
            writer.writerow({"time": frame, "classified_mean": value})

    events_dir = tmp_path / "events"
    _run_events_cli(["--files", str(classified), "--output-dir", str(events_dir)])
    details = _read_csv(events_dir / "event_details.csv")
    summary = _read_csv(events_dir / "event_summary.csv")
    assert {row["event_source_type"] for row in details} >= {"normal_on", "normal_off"}
    assert {row["stage_state_index"] for row in summary} == {"1", "2"}

    stats_dir = tmp_path / "stats"
    completed = subprocess.run(
        [sys.executable, "-m", "frethmm.app.cli", "dwell-stats", "--input", str(events_dir / "event_details.csv"), "--output-dir", str(stats_dir)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    dwell_summary = _read_csv(stats_dir / "dwell_stats_summary.csv")
    assert completed.returncode == 0
    assert {row["stage_state_index"] for row in dwell_summary} == {"1", "2"}
