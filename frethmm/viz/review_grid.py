"""Batch review grid visualization for raw traces and HMM overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from frethmm.core.batch import process_batch
from frethmm.core.io import read_signal_trace
from frethmm.domain.models import ClassificationConfig, ClassificationResult

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ordered_results(results: list[ClassificationResult]) -> list[ClassificationResult]:
    ordered = sorted(
        [result for result in results if result.filepath is not None],
        key=lambda result: result.filepath.name,
    )
    if not ordered:
        raise ValueError("No results with file paths available for plotting")
    return ordered


def _plot_review_page(
    results: list[ClassificationResult],
    config: ClassificationConfig,
    output: Path,
    *,
    page_index: int,
    page_count: int,
    total_count: int,
    rows: int,
    cols: int = 4,
    figsize_per_panel: tuple[float, float] = (4.0, 2.6),
) -> Path:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for review-grid output")
    if not results:
        raise ValueError("No classification results provided")
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    total = len(results)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_panel[0], rows * figsize_per_panel[1]),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for ax, result in zip(axes.flat, results):
        if result.trace_time is not None and result.trace_signal is not None:
            time = result.trace_time
            signal = result.trace_signal
        else:
            trace = read_signal_trace(
                result.filepath,
                mode=config.data_mode,
                signal_column=config.signal_column,
            )
            time = trace.time
            signal = trace.signal
        classified = result.classified_signal

        ax.plot(time, signal, color="#90A4AE", linewidth=0.9, alpha=0.9)
        ax.plot(time, classified, color="#D32F2F", linewidth=1.4)

        means_text = ", ".join(f"{value:.3f}" for value in result.state_means)
        title = (
            f"{result.filepath.name}\n"
            f"logP={result.log_prob:.1f} | means=[{means_text}]"
        )
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", labelsize=7, length=2)
        ax.grid(alpha=0.18, linewidth=0.5)

        if result.warnings:
            for spine in ax.spines.values():
                spine.set_color("#F57C00")
                spine.set_linewidth(1.5)

    for ax in axes.flat[total:]:
        ax.axis("off")

    fig.suptitle(
        (
            f"FretHMM review grid: {total_count} traces, {config.n_states} states "
            f"(page {page_index}/{page_count})"
        ),
        fontsize=12,
        y=0.995,
    )
    fig.supxlabel("Time", fontsize=10)
    fig.supylabel("Signal / classified mean", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_review_grid(
    results: list[ClassificationResult],
    config: ClassificationConfig,
    output: Path,
    *,
    rows: int = 4,
    cols: int = 4,
    figsize_per_panel: tuple[float, float] = (4.0, 2.6),
) -> list[Path]:
    ordered_results = _ordered_results(results)
    per_page = rows * cols
    if per_page < 1:
        raise ValueError("rows * cols must be >= 1")

    page_results = [
        ordered_results[index:index + per_page]
        for index in range(0, len(ordered_results), per_page)
    ]
    page_count = len(page_results)

    outputs: list[Path] = []
    stem = output.stem
    suffix = output.suffix or ".png"
    for page_number, chunk in enumerate(page_results, start=1):
        page_output = (
            output.parent / f"{stem}_page_{page_number:02d}{suffix}"
            if page_count > 1
            else output
        )
        outputs.append(
            _plot_review_page(
                chunk,
                config,
                page_output,
                page_index=page_number,
                page_count=page_count,
                total_count=len(ordered_results),
                rows=rows,
                cols=cols,
                figsize_per_panel=figsize_per_panel,
            )
        )
    return outputs


def generate_review_grid(
    input_dir: Path,
    config: ClassificationConfig,
    output: Path,
    *,
    results_dir: Optional[Path] = None,
    rows: int = 4,
    cols: int = 4,
    classified_only: bool = True,
) -> tuple[list[ClassificationResult], list[Path]]:
    results = process_batch(
        input_dir=input_dir,
        config=config,
        output_dir=results_dir,
        classified_only=classified_only,
    )
    image_paths = plot_review_grid(results, config, output, rows=rows, cols=cols)
    return results, image_paths
