"""Tests for pyHaMMy Phase 1 I/O module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyhammi.io import (
    compute_fret,
    read_report,
    read_trace,
    write_report,
    write_path,
    write_dwell,
    find_trace_files,
)
from pyhammi.config import TraceData, HMMResult

HAMMY_MAIN = Path(
    r"D:\01_Tool_development\01_Single-molecule_data_analysis_tool\HaMMy-main\HaMMy"
)
SAMPLE_2STATE_REPORT = HAMMY_MAIN / "2_states-2_real-J7" / "1report.dat"
SAMPLE_10STATE_DIR = HAMMY_MAIN / "10_states_5_real-250nM_RecA"


class TestComputeFret:
    def test_basic(self):
        donor = np.array([100.0, 50.0, 0.0])
        acceptor = np.array([100.0, 50.0, 100.0])
        fret = compute_fret(donor, acceptor)
        np.testing.assert_allclose(fret, [0.5, 0.5, 1.0])

    def test_zero_total(self):
        donor = np.array([0.0, 100.0])
        acceptor = np.array([0.0, 0.0])
        fret = compute_fret(donor, acceptor)
        np.testing.assert_allclose(fret, [0.0, 0.0])


class TestReadReport:
    @pytest.mark.skipif(
        not SAMPLE_2STATE_REPORT.exists(),
        reason="Sample report file not found",
    )
    def test_parse_2state_report(self):
        report = read_report(SAMPLE_2STATE_REPORT)
        assert report["n_states"] == 2
        assert len(report["means"]) == 2
        assert report["sigma"] > 0
        assert report["transmat"].shape == (2, 2)

    @pytest.mark.skipif(
        not SAMPLE_10STATE_DIR.exists(),
        reason="Sample 10-state dir not found",
    )
    def test_parse_10state_report(self):
        report = read_report(SAMPLE_10STATE_DIR / "1.dat")
        assert report["n_states"] == 10
        assert len(report["means"]) == 10
        assert report["transmat"].shape == (10, 10)


class TestReadWriteReport:
    def test_roundtrip(self, tmp_path):
        n = 3
        means = np.array([0.2, 0.5, 0.8])
        result = HMMResult(
            n_states=n,
            log_prob=-1234.56,
            means=means,
            sigma=0.07,
            signal_sigma=200.0,
            transmat=np.array([
                [0.0, 0.01, 0.0],
                [0.005, 0.0, 0.005],
                [0.0, 0.02, 0.0],
            ]),
            viterbi_path=np.array([0, 0, 1, 1, 2]),
            idealized_fret=np.array([0.2, 0.2, 0.5, 0.5, 0.8]),
            fraction_spent=np.zeros((3, 3)),
            transitions_found=np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ], dtype=int),
            filepath=tmp_path / "test_trace.dat",
        )

        out = write_report(result, tmp_path)
        assert out.exists()

        parsed = read_report(out)
        assert parsed["n_states"] == 3
        np.testing.assert_allclose(parsed["means"], means, atol=1e-5)
        assert parsed["log_prob"] == pytest.approx(-1234.56, abs=0.01)


class TestReadTrace:
    def test_read_simple_trace(self, tmp_path):
        trace_file = tmp_path / "trace.dat"
        trace_file.write_text(
            "1.0  100  200\n"
            "2.0  150  150\n"
            "3.0   50  300\n"
        )
        td = read_trace(trace_file)
        assert td.n_frames == 3
        np.testing.assert_allclose(td.fret, [200/300, 150/300, 300/350], atol=1e-6)


class TestFindTraceFiles:
    def test_finds_dat_files(self, tmp_path):
        (tmp_path / "trace1.dat").write_text("1 2 3\n")
        (tmp_path / "trace2.dat").write_text("4 5 6\n")
        (tmp_path / "1report.dat").write_text("report\n")
        (tmp_path / "2path.dat").write_text("path\n")

        files = find_trace_files(tmp_path)
        names = [f.name for f in files]
        assert "trace1.dat" in names
        assert "trace2.dat" in names
        assert "1report.dat" not in names
        assert "2path.dat" not in names
