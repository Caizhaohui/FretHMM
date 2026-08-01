"""End-to-end regression tests using committed, de-identified fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from frethmm.core.model import sort_state_outputs
from frethmm.viz.tdp import aggregate_transitions, load_reports


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "data"
LEGACY_REPORTS = FIXTURE_ROOT / "legacy_reports"
SINGLE_TRACE = FIXTURE_ROOT / "single_channel_trace.csv"
PAIRED_TRACE = FIXTURE_ROOT / "paired_channel_trace.csv"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "frethmm.app.cli", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_aggregate_transitions_reads_bundled_legacy_reports():
    reports = load_reports(LEGACY_REPORTS)

    assert len(reports) == 4
    starts, stops, weights = aggregate_transitions(reports)
    transition_map = {
        (round(float(start), 6), round(float(stop), 6)): int(weight)
        for start, stop, weight in zip(starts, stops, weights)
    }
    assert transition_map[(-253.287, 1813.47)] == 1
    assert transition_map[(1813.47, -253.287)] == 2
    assert transition_map[(-283.731, 1564.96)] == 2
    assert transition_map[(1564.96, -283.731)] == 3


def test_load_reports_handles_bundled_two_state_format():
    reports = load_reports(LEGACY_REPORTS)

    assert sorted(report["n_states"] for report in reports) == [2, 2, 2, 3]
    two_state_means = [report["means"] for report in reports if report["n_states"] == 2]
    assert any(np.allclose(means, [-253.287, 1813.47]) for means in two_state_means)


def test_sort_state_outputs_remaps_viterbi_path_by_sorted_means():
    means = np.array([0.8, 0.2, 0.5], dtype=np.float64)
    transmat = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.25, 0.25, 0.5],
        ],
        dtype=np.float64,
    )
    viterbi_path = np.array([1, 1, 2, 0, 2, 0], dtype=np.int64)

    means_sorted, transmat_sorted, viterbi_sorted = sort_state_outputs(
        means,
        transmat,
        viterbi_path,
    )

    np.testing.assert_allclose(means_sorted, [0.2, 0.5, 0.8])
    np.testing.assert_allclose(
        transmat_sorted,
        [
            [0.4, 0.3, 0.3],
            [0.25, 0.5, 0.25],
            [0.2, 0.1, 0.7],
        ],
    )
    np.testing.assert_array_equal(viterbi_sorted, [0, 0, 1, 2, 1, 2])


def test_cli_run_matches_committed_single_channel_hashes(tmp_path):
    completed = _run_cli(
        "run",
        "--files",
        str(SINGLE_TRACE),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--n-init",
        "1",
        "--output-dir",
        str(tmp_path),
    )

    assert "Done. Processed 1 file(s)." in completed.stdout
    expected = json.loads(
        (FIXTURE_ROOT / "single_channel_expected_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    assert _sha256(tmp_path / "single_channel_trace_classified.csv") == expected[
        "classified_sha256"
    ]
    assert _sha256(tmp_path / "single_channel_trace_summary.json") == expected[
        "summary_sha256"
    ]

    manifests = list(tmp_path.glob("frethmm_run_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["application"]["version"] == "1.6.0"
    assert manifest["command"] == "run"
    assert manifest["parameters"]["fit"]["n_init"] == 1
    assert manifest["inputs"][0]["name"] == SINGLE_TRACE.name
    assert {Path(output["path"]).name for output in manifest["outputs"]} >= {
        "single_channel_trace_classified.csv",
        "single_channel_trace_summary.json",
    }


def test_cli_run_supports_bundled_paired_channel_trace(tmp_path):
    _run_cli(
        "run",
        "--files",
        str(PAIRED_TRACE),
        "--states",
        "2",
        "--mode",
        "paired_channel",
        "--n-init",
        "1",
        "--output-dir",
        str(tmp_path),
    )

    classified = tmp_path / "paired_channel_trace_classified.csv"
    assert classified.read_text(encoding="utf-8").startswith("time,classified_mean\n")
    summary = json.loads(
        (tmp_path / "paired_channel_trace_summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_states"] == 2


def test_cli_run_classified_only_writes_primary_csv_and_manifest(tmp_path):
    _run_cli(
        "run",
        "--files",
        str(SINGLE_TRACE),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--n-init",
        "1",
        "--classified-only",
        "--output-dir",
        str(tmp_path),
    )

    assert (tmp_path / "single_channel_trace_classified.csv").exists()
    assert list(tmp_path.glob("frethmm_run_manifest_*.json"))
    assert not (tmp_path / "single_channel_trace_summary.json").exists()
    assert not (tmp_path / "single_channel_tracereport.dat").exists()
    assert not (tmp_path / "single_channel_tracepath.dat").exists()
    assert not (tmp_path / "single_channel_tracedwell.dat").exists()


def test_cli_run_with_low_state_tail_trim_reports_trim_warning(tmp_path):
    completed = _run_cli(
        "run",
        "--files",
        str(SINGLE_TRACE),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--n-init",
        "1",
        "--guesses",
        "0.2,0.8",
        "--low-state-tail-trim-seconds",
        "5",
        "--output-dir",
        str(tmp_path),
    )

    assert "Applied low-state tail trim after 5s" in completed.stdout
    assert (tmp_path / "single_channel_trace_classified.csv").exists()
    assert (tmp_path / "single_channel_trace_summary.json").exists()


def test_cli_run_multistart_default_runs_without_error(tmp_path):
    _run_cli(
        "run",
        "--files",
        str(SINGLE_TRACE),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--output-dir",
        str(tmp_path),
    )

    summary_data = json.loads(
        (tmp_path / "single_channel_trace_summary.json").read_text(encoding="utf-8")
    )
    assert summary_data["n_init"] == 10
    assert "bic" in summary_data


def test_cli_run_states_auto_selects_state_count(tmp_path):
    _run_cli(
        "run",
        "--files",
        str(SINGLE_TRACE),
        "--states",
        "auto",
        "--mode",
        "single_channel",
        "--min-states",
        "2",
        "--max-states",
        "3",
        "--n-init",
        "2",
        "--output-dir",
        str(tmp_path),
    )

    summary_data = json.loads(
        (tmp_path / "single_channel_trace_summary.json").read_text(encoding="utf-8")
    )
    candidate_counts = sorted(c["n_states"] for c in summary_data["model_candidates"])
    assert candidate_counts == [2, 3]
