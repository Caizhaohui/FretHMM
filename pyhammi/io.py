"""File I/O for pyHaMMy: read input traces, write output reports/paths/dwells."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from pyhammi.config import TraceData, HMMResult


def compute_fret(donor: np.ndarray, acceptor: np.ndarray) -> np.ndarray:
    total = donor + acceptor
    with np.errstate(divide="ignore", invalid="ignore"):
        fret = np.where(total > 0, acceptor / total, 0.0)
    return fret


def _detect_skip_rows(filepath: Path) -> int:
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    try:
        parts = first_line.replace(",", " ").split()
        float(parts[0])
        return 0
    except ValueError:
        return 1


def _detect_delimiter(filepath: Path) -> str | None:
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    if "," in first_line:
        return ","
    return None


def _detect_mode(raw: np.ndarray) -> str:
    if raw.ndim != 2 or raw.shape[1] < 3:
        return "single_channel"
    col1 = raw[:, 1]
    col2 = raw[:, 2]
    frac_negative_col1 = np.mean(col1 < 0)
    frac_negative_col2 = np.mean(col2 < 0)
    if frac_negative_col1 > 0.3 and frac_negative_col2 < 0.01:
        return "single_channel"
    if frac_negative_col2 > 0.3 and frac_negative_col1 < 0.01:
        return "single_channel"
    return "fret"


def read_trace(
    filepath: Union[str, Path],
    mode: str = "auto",
    signal_column: int = 1,
) -> TraceData:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    skip = _detect_skip_rows(filepath)
    delimiter = _detect_delimiter(filepath)

    raw = np.loadtxt(
        str(filepath),
        dtype=np.float64,
        delimiter=delimiter,
        skiprows=skip,
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if mode == "auto":
        mode = _detect_mode(raw)

    time = raw[:, 0]

    if mode == "single_channel":
        signal = raw[:, signal_column]
        donor = np.zeros_like(signal)
        acceptor = signal.copy()
        fret = signal.copy()
        observations = signal.copy()
    else:
        if raw.shape[1] < 3:
            raise ValueError(
                f"Expected at least 3 columns for donor/acceptor mode, "
                f"got {raw.shape[1]} in {filepath.name}"
            )
        donor = raw[:, 1]
        acceptor = raw[:, 2]
        fret = compute_fret(donor, acceptor)
        observations = fret.copy()
        mode = "fret"

    return TraceData(
        time=time,
        donor=donor,
        acceptor=acceptor,
        fret=fret,
        observations=observations,
        filepath=filepath,
        mode=mode,
    )


def write_report(result: HMMResult, output_dir: Optional[Union[str, Path]] = None) -> Path:
    if result.filepath is None:
        raise ValueError("HMMResult.filepath is required to determine output path")

    out_dir = _resolve_output_dir(result.filepath, output_dir)
    stem = result.filepath.stem
    out_path = out_dir / f"{stem}report.dat"

    n = result.n_states
    lines = []
    lines.append(
        f"Number of states:  {n}    Max probability found:  {result.log_prob:.2f}"
    )
    peaks_str = "  ".join(f"{m:.6g}" for m in result.means)
    lines.append(f"FRET peaks at:    {peaks_str}")
    lines.append(
        f"FRET sigma:  {result.sigma}\t    Signal sigma:  {result.signal_sigma:.1f}"
    )
    lines.append("Transition probability matrix: ")

    for i in range(n):
        for j in range(n):
            tp = result.transmat[i, j]
            frac = result.fraction_spent[i, j]
            ntrans = result.transitions_found[i, j]
            lines.append(
                f"\t{result.means[i]:.6f}\t{result.means[j]:.6f}\t{tp}\t{frac}\t{ntrans}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out_path


def write_path(
    trace: TraceData,
    result: HMMResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    if result.filepath is None:
        raise ValueError("HMMResult.filepath is required to determine output path")

    out_dir = _resolve_output_dir(result.filepath, output_dir)
    stem = result.filepath.stem
    out_path = out_dir / f"{stem}path.dat"

    if trace.mode == "single_channel":
        data = np.column_stack([
            trace.observations,
            np.zeros(trace.n_frames),
            trace.observations,
            result.idealized_fret,
        ])
    else:
        data = np.column_stack([
            trace.donor,
            trace.acceptor,
            trace.fret,
            result.idealized_fret,
        ])
    _write_columns(out_path, data, fmt="%.6f")
    return out_path


def write_dwell(
    result: HMMResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    if result.filepath is None:
        raise ValueError("HMMResult.filepath is required to determine output path")

    out_dir = _resolve_output_dir(result.filepath, output_dir)
    stem = result.filepath.stem
    out_path = out_dir / f"{stem}dwell.dat"

    from pyhammi.postprocess import extract_dwell_times
    dwell = extract_dwell_times(result)
    if len(dwell) == 0:
        out_path.write_text("", encoding="ascii")
        return out_path

    start_fret = result.means[dwell[:, 0].astype(int)]
    stop_fret = result.means[dwell[:, 1].astype(int)]
    frames = dwell[:, 2]

    data = np.column_stack([start_fret, stop_fret, frames])
    _write_columns(out_path, data, fmt="%.6f\t%.6f\t%d")
    return out_path


def read_report(filepath: Union[str, Path]) -> dict:
    filepath = Path(filepath)
    text = filepath.read_text(encoding="ascii")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    n_states = None
    log_prob = None
    means = None
    sigma = None
    signal_sigma = None
    trans_entries = []

    for line in lines:
        if line.startswith("Number of states:"):
            parts = line.split("Max probability found:")
            n_states = int(parts[0].split(":")[1].strip())
            log_prob = float(parts[1].strip())
        elif line.startswith("FRET peaks at:"):
            vals = line.split(":")[1].strip().split()
            means = np.array([float(v) for v in vals])
        elif line.startswith("FRET sigma:"):
            parts = line.split("Signal sigma:")
            sigma = float(parts[0].split(":")[1].strip())
            signal_sigma = float(parts[1].strip())
        elif line.startswith("Transition probability") or line == "":
            continue
        else:
            vals = line.split()
            if len(vals) == 5:
                try:
                    trans_entries.append([float(v) for v in vals])
                except ValueError:
                    continue

    if n_states is None or means is None:
        raise ValueError(f"Could not parse report file: {filepath}")

    n = n_states
    transmat = np.zeros((n, n))
    fraction_spent = np.zeros((n, n))
    transitions_found = np.zeros((n, n), dtype=int)

    for entry in trans_entries:
        start_fret, stop_fret, tp, frac, nf = entry
        i = int(np.argmin(np.abs(means - start_fret)))
        j = int(np.argmin(np.abs(means - stop_fret)))
        transmat[i, j] = tp
        fraction_spent[i, j] = frac
        transitions_found[i, j] = int(nf)

    return {
        "n_states": n_states,
        "log_prob": log_prob,
        "means": means,
        "sigma": sigma,
        "signal_sigma": signal_sigma,
        "transmat": transmat,
        "fraction_spent": fraction_spent,
        "transitions_found": transitions_found,
    }


def find_trace_files(
    input_dir: Union[str, Path],
    extensions: tuple = (".dat", ".txt", ".csv", ".tsv"),
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    files = sorted(
        f
        for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in extensions
        and "report" not in f.stem.lower()
        and "path" not in f.stem.lower()
        and "dwell" not in f.stem.lower()
    )
    return files


def _resolve_output_dir(
    input_path: Path,
    output_dir: Optional[Union[str, Path]],
) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = input_path.parent
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_columns(path: Path, data: np.ndarray, fmt: str = "%.6f") -> None:
    np.savetxt(str(path), data, fmt=fmt, delimiter="\t")
