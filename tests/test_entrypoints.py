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
    assert _run_module("frethmm.app.cli").stdout.strip() == "FretHMM 1.4.0"


def test_gui_version_does_not_launch_a_window():
    assert _run_module("frethmm.app.gui").stdout.strip() == "FretHMM 1.4.0"


def test_cli_run_defaults_to_single_channel_mode_and_250_second_filter():
    from frethmm.app.cli import build_parser

    args = build_parser().parse_args(["run", "--files", "trace.csv"])

    assert args.mode == "single_channel"
    assert args.low_state_tail_trim_seconds == 250.0
