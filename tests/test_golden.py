"""Golden tests based on bundled HaMMy reference outputs."""

from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import pytest
import json

from frethmm.core.model import sort_state_outputs
from frethmm.viz.tdp import aggregate_transitions, load_reports


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
HAMMY_MAIN = WORKSPACE_ROOT / "HaMMy-main" / "HaMMy"
SAMPLE_2STATE_DIR = HAMMY_MAIN / "2_states-2_real-J7"
SAMPLE_10STATE_DIR = HAMMY_MAIN / "10_states_5_real-250nM_RecA"
VALUES1_CSV = WORKSPACE_ROOT / "Values1.csv"
VALUES2_CSV = WORKSPACE_ROOT / "Values2.csv"
VALUES1_CLASSIFIED = REPO_ROOT / "tests" / "fixtures" / "Values1_classified.csv"
VALUES1_SUMMARY = REPO_ROOT / "tests" / "fixtures" / "Values1_summary.json"
VALUES2_HASHES = REPO_ROOT / "tests" / "fixtures" / "Values2_hashes.json"


@pytest.mark.skipif(not SAMPLE_2STATE_DIR.exists(), reason="HaMMy sample data not found")
def test_aggregate_transitions_matches_reference_2state_dataset():
    reports = load_reports(SAMPLE_2STATE_DIR)

    assert len(reports) == 26

    starts, stops, weights = aggregate_transitions(reports)
    transition_map = {
        (round(float(start), 6), round(float(stop), 6)): int(weight)
        for start, stop, weight in zip(starts, stops, weights)
    }

    assert transition_map[(0.340479, 0.692749)] == 95
    assert transition_map[(0.692749, 0.340479)] == 95
    assert (0.393029, 1.11388) not in transition_map
    assert (1.11388, 0.393029) not in transition_map


@pytest.mark.skipif(not SAMPLE_10STATE_DIR.exists(), reason="HaMMy sample data not found")
def test_load_reports_reads_all_reference_10state_reports():
    reports = load_reports(SAMPLE_10STATE_DIR)

    assert len(reports) == 196
    assert all(report["n_states"] == 10 for report in reports)


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


@pytest.mark.skipif(not VALUES1_CSV.exists(), reason="Values1 sample data not found")
def test_cli_run_matches_values1_regression_outputs(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "frethmm.app.cli",
        "run",
        "--files",
        str(VALUES1_CSV),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--output-dir",
        str(tmp_path),
    ]

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Done. Processed 1 file(s)." in completed.stdout
    assert "Values1_classified.csv" in completed.stdout
    assert "Values1_summary.json" in completed.stdout

    assert (tmp_path / "Values1_classified.csv").read_text(encoding="utf-8") == (
        VALUES1_CLASSIFIED.read_text(encoding="utf-8")
    )
    assert (tmp_path / "Values1_summary.json").read_text(encoding="utf-8") == (
        VALUES1_SUMMARY.read_text(encoding="utf-8")
    )


@pytest.mark.skipif(not VALUES2_CSV.exists(), reason="Values2 sample data not found")
def test_cli_run_values2_single_channel_column_selection(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "frethmm.app.cli",
        "run",
        "--files",
        str(VALUES2_CSV),
        "--states",
        "2",
        "--mode",
        "single_channel",
        "--signal-column",
        "1",
        "--output-dir",
        str(tmp_path),
    ]

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = json.loads(VALUES2_HASHES.read_text(encoding="utf-8"))
    classified_hash = hashlib.sha256(
        (tmp_path / "Values2_classified.csv").read_bytes()
    ).hexdigest().upper()
    summary_hash = hashlib.sha256(
        (tmp_path / "Values2_summary.json").read_bytes()
    ).hexdigest().upper()

    assert classified_hash == expected["classified_sha256"]
    assert summary_hash == expected["summary_sha256"]
