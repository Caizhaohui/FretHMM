"""I/O utilities for generic signal-classification workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from frethmm.domain.models import ClassificationResult, SignalTrace


def compute_ratio_signal(channel_1: np.ndarray, channel_2: np.ndarray) -> np.ndarray:
    total = channel_1 + channel_2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, channel_2 / total, 0.0)


def _detect_skip_rows(filepath: Path) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    try:
        parts = first_line.replace(",", " ").split()
        float(parts[0])
        return 0
    except ValueError:
        return 1


def _detect_delimiter(filepath: Path) -> str | None:
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    return "," if "," in first_line else None


def _detect_mode(raw: np.ndarray) -> str:
    if raw.ndim != 2 or raw.shape[1] < 3:
        return "single_channel"
    channel_1 = raw[:, 1]
    channel_2 = raw[:, 2]
    frac_negative_1 = np.mean(channel_1 < 0)
    frac_negative_2 = np.mean(channel_2 < 0)
    if frac_negative_1 > 0.3 and frac_negative_2 < 0.01:
        return "single_channel"
    if frac_negative_2 > 0.3 and frac_negative_1 < 0.01:
        return "single_channel"
    return "paired_channel"


def read_signal_trace(
    filepath: Union[str, Path],
    mode: str = "auto",
    signal_column: int = 1,
) -> SignalTrace:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    raw = np.loadtxt(
        str(filepath),
        dtype=np.float64,
        delimiter=_detect_delimiter(filepath),
        skiprows=_detect_skip_rows(filepath),
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if mode == "auto":
        mode = _detect_mode(raw)

    time = raw[:, 0]
    if mode == "single_channel":
        if signal_column < 1 or signal_column >= raw.shape[1]:
            raise ValueError(
                f"Signal column must be between 1 and {raw.shape[1] - 1} for {filepath.name}, got {signal_column}"
            )
        signal = raw[:, signal_column]
        return SignalTrace(
            time=time,
            signal=signal.copy(),
            observations=signal.copy(),
            filepath=filepath,
            mode="single_channel",
        )

    if raw.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns for paired-channel mode, got {raw.shape[1]} in {filepath.name}"
        )
    channel_1 = raw[:, 1]
    channel_2 = raw[:, 2]
    derived_signal = compute_ratio_signal(channel_1, channel_2)
    return SignalTrace(
        time=time,
        signal=derived_signal.copy(),
        observations=derived_signal.copy(),
        filepath=filepath,
        mode="paired_channel",
        channel_1=channel_1,
        channel_2=channel_2,
        derived_signal=derived_signal,
    )


def write_classified_csv(
    trace: SignalTrace,
    result: ClassificationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    out_dir = _resolve_output_dir(result.filepath, output_dir)
    out_path = out_dir / f"{result.filepath.stem}_classified.csv"
    data = np.column_stack([trace.time, result.classified_signal])
    np.savetxt(
        str(out_path),
        data,
        fmt="%.6f",
        delimiter=",",
        header="time,classified_mean",
        comments="",
    )
    return out_path


def write_summary_json(
    result: ClassificationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    out_dir = _resolve_output_dir(result.filepath, output_dir)
    out_path = out_dir / f"{result.filepath.stem}_summary.json"
    state_counts = np.bincount(result.state_path, minlength=result.n_states)
    total_frames = int(state_counts.sum())
    dwell = result.dwell_segments
    dwell_by_state = {str(i): [] for i in range(result.n_states)}
    if len(dwell) > 0:
        for start_state, _stop_state, frames in dwell:
            dwell_by_state[str(int(start_state))].append(int(frames))
    payload = {
        "source_file": result.filepath.name,
        "n_states": result.n_states,
        "log_prob": result.log_prob,
        "state_means": [float(v) for v in result.state_means.tolist()],
        "state_sigma": result.state_sigma,
        "signal_sigma": result.signal_sigma,
        "state_frame_counts": [int(v) for v in state_counts.tolist()],
        "state_frame_fraction": (
            [float(v / total_frames) for v in state_counts.tolist()]
            if total_frames > 0
            else [0.0] * result.n_states
        ),
        "transition_counts": result.transitions_found.astype(int).tolist(),
        "transition_probabilities": result.transition_matrix.astype(float).tolist(),
        "dwell_by_state": dwell_by_state,
        "low_state_tail_trim_seconds": result.low_state_tail_trim_seconds,
        "low_state_tail_cutoff_time": result.low_state_tail_cutoff_time,
        "low_state_tail_kept_frames": result.low_state_tail_kept_frames,
        "warnings": list(result.warnings),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_state_report(
    result: ClassificationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    out_dir = _resolve_output_dir(result.filepath, output_dir)
    out_path = out_dir / f"{result.filepath.stem}report.dat"
    lines = [
        f"Number of states:  {result.n_states}    Max probability found:  {result.log_prob:.2f}",
        f"State means:    {'  '.join(f'{m:.6g}' for m in result.state_means)}",
        f"State sigma:  {result.state_sigma}\t    Signal sigma:  {result.signal_sigma:.1f}",
        "Transition probability matrix: ",
    ]
    for i in range(result.n_states):
        for j in range(result.n_states):
            lines.append(
                f"\t{result.state_means[i]:.6f}\t{result.state_means[j]:.6f}\t"
                f"{result.transition_matrix[i, j]}\t{result.fraction_spent[i, j]}\t"
                f"{result.transitions_found[i, j]}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out_path


def write_state_path(
    trace: SignalTrace,
    result: ClassificationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    out_dir = _resolve_output_dir(result.filepath, output_dir)
    out_path = out_dir / f"{result.filepath.stem}path.dat"
    if trace.mode == "single_channel":
        data = np.column_stack(
            [trace.signal, np.zeros(trace.n_frames), trace.signal, result.classified_signal]
        )
    else:
        data = np.column_stack(
            [trace.channel_1, trace.channel_2, trace.derived_signal, result.classified_signal]
        )
    np.savetxt(str(out_path), data, fmt="%.6f", delimiter="\t")
    return out_path


def write_dwell_report(
    result: ClassificationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    out_dir = _resolve_output_dir(result.filepath, output_dir)
    out_path = out_dir / f"{result.filepath.stem}dwell.dat"
    dwell = result.dwell_segments
    if len(dwell) == 0:
        out_path.write_text("", encoding="ascii")
        return out_path
    start_signal = result.state_means[dwell[:, 0].astype(int)]
    stop_signal = result.state_means[dwell[:, 1].astype(int)]
    frames = dwell[:, 2]
    data = np.column_stack([start_signal, stop_signal, frames])
    np.savetxt(str(out_path), data, fmt="%.6f\t%.6f\t%d", delimiter="\t")
    return out_path


def read_state_report(filepath: Union[str, Path]) -> dict:
    from frethmm.formats.report_parser import read_report_file

    return read_report_file(filepath)


def find_trace_files(
    input_dir: Union[str, Path],
    extensions: tuple = (".dat", ".txt", ".csv", ".tsv"),
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    return sorted(
        f
        for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in extensions
        and "report" not in f.stem.lower()
        and "path" not in f.stem.lower()
        and "dwell" not in f.stem.lower()
        and "_classified" not in f.stem.lower()
        and "_summary" not in f.stem.lower()
    )


def _resolve_output_dir(
    input_path: Optional[Path],
    output_dir: Optional[Union[str, Path]],
) -> Path:
    if input_path is None:
        raise ValueError("filepath is required to determine output path")
    out = Path(output_dir) if output_dir is not None else input_path.parent
    out.mkdir(parents=True, exist_ok=True)
    return out
