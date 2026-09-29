"""Smoke tests for non-interactive version entry points."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--version"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_version_reports_package_version():
    from frethmm import __version__

    assert _run_module("frethmm.app.cli").stdout.strip() == f"FretHMM {__version__}"


def test_gui_version_does_not_launch_a_window():
    from frethmm import __version__

    assert _run_module("frethmm.app.gui").stdout.strip() == f"FretHMM {__version__}"


def test_cli_run_defaults_to_single_channel_mode_and_250_second_filter():
    from frethmm.app.cli import build_parser

    args = build_parser().parse_args(["run", "--files", "trace.csv"])

    assert args.mode == "single_channel"
    assert args.low_state_tail_trim_seconds == 250.0


def test_cli_run_parses_phase4_hardening_arguments():
    from frethmm.app.cli import build_parser

    args = build_parser().parse_args([
        "run",
        "--files", "trace.csv",
        "--remove-spikes",
        "--spike-threshold-sigma", "4.5",
        "--trim-initial-artifacts",
        "--max-initial-artifact-frames", "3",
        "--smooth-window", "5",
        "--min-dwell-frames", "2",
        "--merge-state-threshold", "0.05",
        "--merge-state-sigma-factor", "0.4",
    ])

    assert args.remove_spikes is True
    assert args.spike_threshold_sigma == 4.5
    assert args.trim_initial_artifacts is True
    assert args.max_initial_artifact_frames == 3
    assert args.smooth_window == 5
    assert args.min_dwell_frames == 2
    assert args.merge_state_threshold == 0.05
    assert args.merge_state_sigma_factor == 0.4


def test_cli_review_grid_accepts_files_argument():
    from frethmm.app.cli import build_parser

    args = build_parser().parse_args([
        "review-grid",
        "--files", "trace1.csv", "trace2.csv",
        "--output", "review.png",
    ])

    assert args.files == ["trace1.csv", "trace2.csv"]
    assert args.input_dir is None
    assert args.output == "review.png"
