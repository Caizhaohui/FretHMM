"""Batch processing helpers for signal classification."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from frethmm.core.io import find_trace_files
from frethmm.core.model import process_trace_file
from frethmm.domain.models import ClassificationConfig, ClassificationResult


def process_batch(
    input_dir: Path,
    config: ClassificationConfig,
    output_dir: Optional[Path] = None,
    classified_only: bool = False,
    extensions: tuple[str, ...] = (".dat", ".txt", ".csv", ".tsv"),
) -> list[ClassificationResult]:
    files = find_trace_files(input_dir, extensions)
    if not files:
        print(f"No trace files found in {input_dir}")
        return []
    return process_files(files, config, output_dir, classified_only=classified_only)


def process_files(
    files: list[Path],
    config: ClassificationConfig,
    output_dir: Optional[Path] = None,
    classified_only: bool = False,
) -> list[ClassificationResult]:
    results: list[ClassificationResult] = []
    total = len(files)

    def _format_result_line(result: ClassificationResult) -> str:
        """One-line per-file summary including multi-start / BIC context."""
        parts = [
            f"  -> {result.n_states} states",
            f"log_prob={result.log_prob:.2f}",
        ]
        # Surface BIC when model selection ran; otherwise surface n_init context
        # only when multi-start is active, to avoid churning legacy logs.
        if result.model_candidates is not None and result.bic is not None:
            parts.append(f"bic={result.bic:.2f} (auto)")
        elif result.n_init is not None and result.n_init > 1:
            parts.append(f"n_init={result.n_init}")
        return ", ".join(parts) + f", means={result.state_means}"

    def _print_warnings(result: ClassificationResult) -> None:
        for warning in result.warnings:
            print(f"    WARNING: {warning}")

    try:
        if config.workers <= 1:
            for i, filepath in enumerate(files, 1):
                print(f"[{i}/{total}] Processing {filepath.name}...")
                try:
                    result = process_trace_file(filepath, config, output_dir, classified_only)
                    results.append(result)
                    _print_warnings(result)
                    print(_format_result_line(result))
                except Exception as exc:
                    print(f"  -> ERROR: {exc}")
        else:
            workers = min(config.workers, len(files))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_trace_file, filepath, config, output_dir, classified_only): index
                    for index, filepath in enumerate(files)
                }
                ordered_results: list[Optional[ClassificationResult]] = [None] * len(files)
                for future in as_completed(futures):
                    idx = futures[future]
                    filepath = files[idx]
                    print(f"[{idx + 1}/{total}] Processing {filepath.name}...")
                    try:
                        result = future.result()
                        ordered_results[idx] = result
                        _print_warnings(result)
                        print(_format_result_line(result))
                    except Exception as exc:
                        print(f"  -> ERROR: {exc}")
                results = [result for result in ordered_results if result is not None]
    except KeyboardInterrupt:
        print("\nInterrupted by user. Returning partial results...")

    return results
