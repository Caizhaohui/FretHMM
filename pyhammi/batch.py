"""Batch processing with multiprocessing support."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pyhammi.config import HMMConfig, HMMResult
from pyhammi.io import find_trace_files
from pyhammi.model import process_file


def process_batch(
    input_dir: Path,
    config: HMMConfig,
    output_dir: Optional[Path] = None,
    extensions: tuple = (".dat", ".txt", ".csv", ".tsv"),
) -> list[HMMResult]:
    files = find_trace_files(input_dir, extensions)
    if not files:
        print(f"No trace files found in {input_dir}")
        return []

    return _run_files(files, config, output_dir, config.workers)


def process_files(
    files: list[Path],
    config: HMMConfig,
    output_dir: Optional[Path] = None,
) -> list[HMMResult]:
    return _run_files(files, config, output_dir, config.workers)


def _run_files(
    files: list[Path],
    config: HMMConfig,
    output_dir: Optional[Path],
    workers: int,
) -> list[HMMResult]:
    results: list[HMMResult] = []
    total = len(files)

    def _print_result(r: HMMResult) -> None:
        for w in r.warnings:
            print(f"    WARNING: {w}")

    try:
        if workers <= 1:
            for i, fp in enumerate(files, 1):
                print(f"[{i}/{total}] Processing {fp.name}...")
                try:
                    r = process_file(fp, config, output_dir)
                    results.append(r)
                    _print_result(r)
                    print(f"  -> {r.n_states} states, log_prob={r.log_prob:.2f}, "
                          f"means={r.means}")
                except Exception as e:
                    print(f"  -> ERROR: {e}")
        else:
            workers = min(workers, len(files))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_file, fp, config, output_dir): i
                    for i, fp in enumerate(files)
                }
                results_ordered: list[Optional[HMMResult]] = [None] * len(files)
                for future in as_completed(futures):
                    idx = futures[future]
                    fp = files[idx]
                    print(f"[{idx + 1}/{total}] Processing {fp.name}...")
                    try:
                        r = future.result()
                        results_ordered[idx] = r
                        _print_result(r)
                        print(f"  -> {r.n_states} states, log_prob={r.log_prob:.2f}")
                    except Exception as e:
                        print(f"  -> ERROR: {e}")
                results = [r for r in results_ordered if r is not None]
    except KeyboardInterrupt:
        print("\nInterrupted by user. Returning partial results...")

    return results
