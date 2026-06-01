"""Tests for FretHMM I/O utilities."""

from pathlib import Path

import numpy as np
import pytest

from frethmm.core.io import (
    compute_ratio_signal,
    find_trace_files,
    read_signal_trace,
    write_classified_csv,
    write_state_report,
    write_summary_json,
)
from frethmm.domain.models import ClassificationResult, SignalTrace
from frethmm.formats.report_parser import read_report_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAMMY_MAIN = PROJECT_ROOT / "HaMMy-main" / "HaMMy"
SAMPLE_2STATE_REPORT = HAMMY_MAIN / "2_states-2_real-J7" / "1report.dat"
SAMPLE_2STATE_EDGE_REPORT = HAMMY_MAIN / "2_states-2_real-J7" / "26report.dat"
SAMPLE_10STATE_DIR = HAMMY_MAIN / "10_states_5_real-250nM_RecA"


class TestComputeRatioSignal:
    def test_basic(self):
        channel_1 = np.array([100.0, 50.0, 0.0])
        channel_2 = np.array([100.0, 50.0, 100.0])
        signal = compute_ratio_signal(channel_1, channel_2)
        np.testing.assert_allclose(signal, [0.5, 0.5, 1.0])

    def test_zero_total(self):
        channel_1 = np.array([0.0, 100.0])
        channel_2 = np.array([0.0, 0.0])
        signal = compute_ratio_signal(channel_1, channel_2)
        np.testing.assert_allclose(signal, [0.0, 0.0])


class TestReadReport:
    @pytest.mark.skipif(
        not SAMPLE_2STATE_REPORT.exists(),
        reason="Sample report file not found",
    )
    def test_parse_2state_report(self):
        report = read_report_file(SAMPLE_2STATE_REPORT)
        assert report["n_states"] == 2
        np.testing.assert_allclose(report["means"], [0.340479, 0.692749], atol=1e-6)
        assert report["log_prob"] == pytest.approx(3914.83, abs=0.01)
        assert report["sigma"] == pytest.approx(0.0697278, abs=1e-7)
        assert report["signal_sigma"] == pytest.approx(164.443, abs=1e-3)
        assert report["transmat"].shape == (2, 2)
        assert report["transmat"][0, 1] == pytest.approx(0.0571831, abs=1e-7)
        assert report["transmat"][1, 0] == pytest.approx(0.0455867, abs=1e-7)
        assert report["transitions_found"][0, 1] == 95
        assert report["transitions_found"][1, 0] == 95

    @pytest.mark.skipif(
        not SAMPLE_2STATE_EDGE_REPORT.exists(),
        reason="Sample edge-case report file not found",
    )
    def test_parse_extreme_log_probability(self):
        report = read_report_file(SAMPLE_2STATE_EDGE_REPORT)
        assert report["n_states"] == 2
        assert report["log_prob"] == pytest.approx(-1e100)
        assert report["transitions_found"].sum() == 0

    @pytest.mark.skipif(
        not SAMPLE_10STATE_DIR.exists(),
        reason="Sample 10-state dir not found",
    )
    def test_parse_10state_report(self):
        report = read_report_file(SAMPLE_10STATE_DIR / "1.dat")
        assert report["n_states"] == 10
        assert report["log_prob"] == pytest.approx(2617.29, abs=0.01)
        assert len(report["means"]) == 10
        assert report["means"][0] == pytest.approx(0.139365, abs=1e-6)
        assert report["means"][-1] == pytest.approx(1.57991, abs=1e-5)
        assert report["transmat"].shape == (10, 10)
        assert report["transmat"][0, 1] == pytest.approx(6.64491e-08, rel=1e-6)


class TestReadWriteReport:
    def test_roundtrip(self, tmp_path):
        n = 3
        means = np.array([0.2, 0.5, 0.8])
        result = ClassificationResult(
            n_states=n,
            log_prob=-1234.56,
            state_means=means,
            state_sigma=0.07,
            signal_sigma=200.0,
            transition_matrix=np.array([
                [0.0, 0.01, 0.0],
                [0.005, 0.0, 0.005],
                [0.0, 0.02, 0.0],
            ]),
            state_path=np.array([0, 0, 1, 1, 2]),
            classified_signal=np.array([0.2, 0.2, 0.5, 0.5, 0.8]),
            fraction_spent=np.zeros((3, 3)),
            transitions_found=np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ], dtype=int),
            filepath=tmp_path / "test_trace.dat",
        )

        out = write_state_report(result, tmp_path)
        assert out.exists()

        parsed = read_report_file(out)
        assert parsed["n_states"] == 3
        np.testing.assert_allclose(parsed["means"], means, atol=1e-5)
        assert parsed["log_prob"] == pytest.approx(-1234.56, abs=0.01)

    def test_write_classified_trace_and_summary(self, tmp_path):
        trace = SignalTrace(
            time=np.array([0.0, 1.0, 2.0]),
            signal=np.array([10.0, 12.0, 12.0]),
            observations=np.array([10.0, 12.0, 12.0]),
            filepath=tmp_path / "signal.csv",
            mode="single_channel",
        )
        result = ClassificationResult(
            n_states=2,
            log_prob=-12.34,
            state_means=np.array([11.0, 20.0]),
            state_sigma=1.5,
            signal_sigma=2.0,
            transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
            state_path=np.array([0, 0, 1]),
            classified_signal=np.array([11.0, 11.0, 20.0]),
            fraction_spent=np.array([[0.0, 2 / 3], [0.0, 0.0]]),
            transitions_found=np.array([[0, 1], [0, 0]], dtype=int),
            filepath=trace.filepath,
        )

        classified = write_classified_csv(trace, result, tmp_path)
        summary = write_summary_json(result, tmp_path)

        assert classified.name == "signal_classified.csv"
        assert summary.name == "signal_summary.json"
        assert classified.read_text(encoding="utf-8").startswith("time,classified_mean\n")
        summary_text = summary.read_text(encoding="utf-8")
        assert '"source_file": "signal.csv"' in summary_text
        assert '"state_means": [' in summary_text


class TestReadTrace:
    def test_read_simple_trace(self, tmp_path):
        trace_file = tmp_path / "trace.dat"
        trace_file.write_text(
            "1.0  100  200\n"
            "2.0  150  150\n"
            "3.0   50  300\n"
        )
        td = read_signal_trace(trace_file)
        assert td.n_frames == 3
        np.testing.assert_allclose(td.derived_signal, [200/300, 150/300, 300/350], atol=1e-6)

    def test_read_selects_requested_signal_column(self, tmp_path):
        trace_file = tmp_path / "values2.csv"
        trace_file.write_text(
            "Time,channel1,channel2\n"
            "0,2884,-5096\n"
            "1,2884,1289\n"
            "2,2570,1289\n"
        )
        trace = read_signal_trace(trace_file, mode="single_channel", signal_column=1)
        np.testing.assert_array_equal(trace.signal, [2884, 2884, 2570])
        trace_alt = read_signal_trace(trace_file, mode="single_channel", signal_column=2)
        np.testing.assert_array_equal(trace_alt.signal, [-5096, 1289, 1289])


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
