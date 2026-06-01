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
    extensions: tuple = (".dat", ".txt", ".csv", ".tsv"),
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
                    print(
                        f"  -> {result.n_states} states, log_prob={result.log_prob:.2f}, "
                        f"means={result.state_means}"
                    )
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
                        print(f"  -> {result.n_states} states, log_prob={result.log_prob:.2f}")
                    except Exception as exc:
                        print(f"  -> ERROR: {exc}")
                results = [result for result in ordered_results if result is not None]
    except KeyboardInterrupt:
        print("\nInterrupted by user. Returning partial results...")

    return results
