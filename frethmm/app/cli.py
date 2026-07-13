"""Primary CLI entry point for FretHMM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

from frethmm import __version__
from frethmm.core.batch import process_batch, process_files
from frethmm.core.io import find_trace_files
from frethmm.core.provenance import write_run_manifest
from frethmm.domain.models import ClassificationConfig


def _parse_states(value: str):
    """Parse ``--states``: ``"auto"`` triggers BIC model selection, else int."""
    if value is None:
        return 2
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--states must be a positive int or 'auto', got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--states must be >= 1, got {parsed}")
    return parsed


def _add_fit_arguments(sub) -> None:
    """Shared HMM-fit arguments for ``run`` and ``review-grid``."""
    sub.add_argument(
        "--states",
        type=_parse_states,
        default=2,
        help="Number of HMM states, or 'auto' to pick via BIC (default: 2)",
    )
    sub.add_argument("--guesses", type=str, default=None, help="Comma-separated initial signal guesses")
    sub.add_argument("--max-iter", type=int, default=500, help="Max Baum-Welch iterations (default: 500)")
    sub.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    sub.add_argument("--workers", type=int, default=1, help="Parallel workers for batch mode (default: 1)")
    sub.add_argument("--mode", choices=["auto", "paired_channel", "single_channel"], default="auto")
    sub.add_argument(
        "--signal-column",
        type=int,
        default=1,
        help="1-based signal column index after Time for single_channel mode (default: 1)",
    )
    sub.add_argument(
        "--low-state-tail-trim-seconds",
        type=float,
        default=None,
        help=(
            "Optional trim duration in seconds. If set, FretHMM first classifies once, "
            "then trims raw data after the lowest state has persisted for this long, "
            "then classifies the trimmed raw data again."
        ),
    )
    sub.add_argument(
        "--n-init",
        type=int,
        default=10,
        help=(
            "Number of deterministic multi-start Baum-Welch runs; the best "
            "log-likelihood wins (default: 10, use 1 to reproduce legacy single-fit)"
        ),
    )
    sub.add_argument(
        "--min-states",
        type=int,
        default=2,
        help="Minimum state count for BIC selection (only with --states auto, default: 2)",
    )
    sub.add_argument(
        "--max-states",
        type=int,
        default=6,
        help="Maximum state count for BIC selection (only with --states auto, default: 6)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frethmm",
        description="FretHMM: Hidden Markov Model state classification for single-molecule trajectories",
    )
    parser.add_argument("--version", action="version", version=f"FretHMM {__version__}")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run = sub.add_parser("run", help="Run HMM state classification on trace files")
    _add_fit_arguments(run)
    run.add_argument(
        "--classified-only",
        action="store_true",
        help="Write only *_classified.csv and skip summary/report/path/dwell outputs",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    inp = run.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input-dir", type=str, help="Directory of trace files")
    inp.add_argument("--files", nargs="+", type=str, help="Individual trace files")
    run.add_argument("--output-dir", type=str, default=None, help="Output directory")

    tdp = sub.add_parser("tdp", help="Launch the transition density plot workflow")
    tdp.add_argument("--input-dir", type=str, required=True)
    tdp.add_argument("--exposure", type=float, default=0.1)
    tdp.add_argument("--states", type=int, default=None)
    tdp.add_argument("--output", type=str, default=None)

    review = sub.add_parser(
        "review-grid",
        help="Batch-classify traces and generate a visual review grid",
    )
    review.add_argument("--input-dir", type=str, required=True, help="Directory of trace files")
    review.add_argument("--output", type=str, required=True, help="Output PNG path for the review grid")
    review.add_argument("--output-dir", type=str, default=None, help="Optional directory for classified CSV outputs")
    _add_fit_arguments(review)
    review.add_argument("--rows", type=int, default=4, help="Number of panel rows per review page")
    review.add_argument("--cols", type=int, default=4, help="Number of panels per row in the review grid")

    events = sub.add_parser(
        "events",
        help="Extract ON/OFF events from *_classified.csv files",
    )
    events.add_argument(
        "--tail-off-threshold-seconds",
        type=float,
        default=100.0,
        help=(
            "Exclude the final event when it is OFF and lasts at least this "
            "many seconds (default: 100.0)"
        ),
    )
    events.add_argument("--output-dir", type=str, required=True, help="Output directory")
    events_inp = events.add_mutually_exclusive_group(required=True)
    events_inp.add_argument("--input-dir", type=str, help="Directory of *_classified.csv files")
    events_inp.add_argument("--files", nargs="+", type=str, help="Individual *_classified.csv files")

    dwell = sub.add_parser(
        "dwell-stats",
        help="Descriptive dwell-time statistics + exponential rate-constant fit",
    )
    dwell.add_argument("--input", type=str, required=True, help="Path to event_details.csv (output of `frethmm events`)")
    dwell.add_argument("--output-dir", type=str, required=True, help="Output directory")
    dwell.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Histogram bin count for the exponential fit (default: max(10, n_events // 3))",
    )
    dwell.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip the exponential fit; emit descriptive statistics only",
    )

    sub.add_parser("gui", help="Launch the FretHMM GUI")
    return parser


def _classification_output_paths(
    results,
    output_dir: Path | None,
    *,
    classified_only: bool,
) -> list[Path]:
    """Return output paths that were actually written for successful fits."""
    paths: list[Path] = []
    suffixes = ["_classified.csv"]
    if not classified_only:
        suffixes.extend(["_summary.json", "report.dat", "path.dat", "dwell.dat"])
    for result in results:
        if result.filepath is None:
            continue
        directory = output_dir if output_dir is not None else result.filepath.parent
        for suffix in suffixes:
            path = directory / f"{result.filepath.stem}{suffix}"
            if path.exists() and path not in paths:
                paths.append(path)
    return paths


def _manifest_output_dir(
    configured_output_dir: Path | None,
    input_dir: str | None,
    input_paths: list[Path],
) -> Path:
    if configured_output_dir is not None:
        return configured_output_dir
    if input_dir is not None:
        return Path(input_dir)
    if input_paths:
        return input_paths[0].parent
    return Path.cwd()


def cmd_run(args: argparse.Namespace) -> None:
    import warnings

    if args.verbose:
        warnings.simplefilter("always")
    guesses = [float(value) for value in args.guesses.split(",")] if args.guesses else None
    config = ClassificationConfig(
        n_states=args.states,
        max_iter=args.max_iter,
        tol=args.tol,
        guesses=guesses,
        workers=args.workers,
        data_mode=args.mode,
        signal_column=args.signal_column,
        low_state_tail_trim_seconds=args.low_state_tail_trim_seconds,
        n_init=args.n_init,
        min_states=args.min_states,
        max_states=args.max_states,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    input_paths = (
        find_trace_files(Path(args.input_dir))
        if args.input_dir
        else [Path(path) for path in args.files]
    )
    results = (
        process_batch(
            Path(args.input_dir),
            config,
            output_dir,
            classified_only=args.classified_only,
        )
        if args.input_dir
        else process_files(
            [Path(path) for path in args.files],
            config,
            output_dir,
            classified_only=args.classified_only,
        )
    )
    manifest_path = write_run_manifest(
        command="run",
        parameters={
            "fit": asdict(config),
            "classified_only": args.classified_only,
        },
        input_paths=input_paths,
        output_paths=_classification_output_paths(
            results,
            output_dir,
            classified_only=args.classified_only,
        ),
        output_dir=_manifest_output_dir(output_dir, args.input_dir, input_paths),
    )
    print(f"\nDone. Processed {len(results)} file(s).")
    for result in results:
        stem = result.filepath.stem if result.filepath else "output"
        print(
            f"  {result.filepath.name}: {result.n_states} states, "
            f"log_prob={result.log_prob:.2f}, means={result.state_means}"
        )
        if args.classified_only:
            print(f"    outputs: {stem}_classified.csv")
        else:
            print(f"    outputs: {stem}_classified.csv, {stem}_summary.json")
        for warning in result.warnings:
            print(f"    WARNING: {warning}")
    print(f"  run manifest: {manifest_path}")


def cmd_tdp(args: argparse.Namespace) -> None:
    from frethmm.viz.tdp import generate_tdp

    generate_tdp(
        input_dir=Path(args.input_dir),
        exposure=args.exposure,
        n_display_states=args.states,
        output=args.output,
    )


def cmd_review_grid(args: argparse.Namespace) -> None:
    from frethmm.viz.review_grid import generate_review_grid

    guesses = [float(value) for value in args.guesses.split(",")] if args.guesses else None
    config = ClassificationConfig(
        n_states=args.states,
        max_iter=args.max_iter,
        tol=args.tol,
        guesses=guesses,
        workers=args.workers,
        data_mode=args.mode,
        signal_column=args.signal_column,
        low_state_tail_trim_seconds=args.low_state_tail_trim_seconds,
        n_init=args.n_init,
        min_states=args.min_states,
        max_states=args.max_states,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    input_paths = find_trace_files(Path(args.input_dir))
    results, image_paths = generate_review_grid(
        input_dir=Path(args.input_dir),
        config=config,
        output=Path(args.output),
        results_dir=output_dir,
        rows=args.rows,
        cols=args.cols,
    )
    print("\nReview grid page(s) saved to:")
    for image_path in image_paths:
        print(f"  {image_path}")
    print(f"Rendered {len(results)} file(s).")
    manifest_path = write_run_manifest(
        command="review-grid",
        parameters={
            "fit": asdict(config),
            "rows": args.rows,
            "cols": args.cols,
            "classified_only": True,
        },
        input_paths=input_paths,
        output_paths=[
            *_classification_output_paths(results, output_dir, classified_only=True),
            *image_paths,
        ],
        output_dir=Path(args.output).parent,
    )
    print(f"  run manifest: {manifest_path}")


def cmd_events(args: argparse.Namespace) -> None:
    """Extract ON/OFF events from ``*_classified.csv`` files.

    Each classified file is reverse-parsed into a state path, segmented into
    ON/OFF events (highest-mean state = ON, all others = OFF), and three CSV
    tables are written: per-event details, per-file summaries, and an overall
    aggregate. Single-process; per-file errors are printed but do not abort.
    """
    import csv as csv_module

    from frethmm.core.events import (
        DETAIL_FIELDS,
        SUMMARY_FIELDS,
        event_to_detail_row,
        extract_events,
        overall_fields,
        summarize_events,
        summarize_overall,
    )
    from frethmm.core.io import find_classified_files
    from frethmm.formats.classified_parser import read_classified_csv

    if args.input_dir:
        files = find_classified_files(Path(args.input_dir))
        if not files:
            raise SystemExit(f"No *_classified.csv files found in {args.input_dir}")
    else:
        files = [Path(path) for path in args.files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    per_file_summaries = []
    for path in files:
        try:
            state_path, state_means, times = read_classified_csv(path)
            events = extract_events(
                state_path,
                state_means,
                times,
                path.name,
                tail_off_threshold_seconds=args.tail_off_threshold_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - surface and continue, like batch.py
            print(f"  ERROR {path.name}: {exc}")
            continue
        all_events.extend(events)
        per_file_summaries.append(summarize_events(path.name, events))
        print(f"  {path.name}: {len(events)} event(s)")

    detail_rows = [event_to_detail_row(event) for event in all_events]
    overall = summarize_overall(all_events, len(files))

    def _write_csv(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
        out_path = output_dir / filename
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv("event_details.csv", DETAIL_FIELDS, detail_rows)
    _write_csv("event_summary.csv", SUMMARY_FIELDS, per_file_summaries)
    _write_csv("event_stats_overall.csv", overall_fields(overall), [overall])

    print(f"\nProcessed {len(files)} file(s).")
    print(f"Wrote {len(detail_rows)} event rows to {output_dir / 'event_details.csv'}")
    print(f"Wrote {len(per_file_summaries)} per-file summaries to {output_dir / 'event_summary.csv'}")
    print(f"Wrote overall summary to {output_dir / 'event_stats_overall.csv'}")
    manifest_path = write_run_manifest(
        command="events",
        parameters={"tail_off_threshold_seconds": args.tail_off_threshold_seconds},
        input_paths=files,
        output_paths=[
            output_dir / "event_details.csv",
            output_dir / "event_summary.csv",
            output_dir / "event_stats_overall.csv",
        ],
        output_dir=output_dir,
    )
    print(f"Run manifest: {manifest_path}")


def cmd_dwell_stats(args: argparse.Namespace) -> None:
    """Descriptive dwell-time statistics + optional exponential rate-constant fit.

    Consumes an ``event_details.csv`` (the per-event output of ``frethmm
    events``) and writes two tables: a single-row overall summary (with the
    full descriptive-stat block and, unless ``--no-fit``, the rate constants),
    and a per-file breakdown. Excluded (terminal-off) events are omitted from
    the statistics, mirroring ``frethmm events``.
    """
    import csv as csv_module

    from frethmm.core.dwell_stats import (
        describe_durations,
        fit_dict_to_columns,
        fit_exponential_dwell,
        summarize_events_extended,
        summarize_overall_extended,
    )
    from frethmm.formats.event_details_parser import read_event_details

    events = read_event_details(Path(args.input))
    if not events:
        raise SystemExit(f"No events found in {args.input}")

    included = [event for event in events if not event.excluded]
    on_durations = [e.duration_seconds for e in included if e.event_type == "ON"]
    off_durations = [e.duration_seconds for e in included if e.event_type == "OFF"]

    source_files = {event.source_file for event in events}
    overall = summarize_overall_extended(events, len(source_files))
    # Carry the raw counts the extended summary does not expose directly.
    overall["file_count"] = len(source_files)
    overall["total_events"] = len(events)
    overall["included_events"] = len(included)

    if not args.no_fit:
        on_fit = fit_exponential_dwell(on_durations, n_bins=args.bins)
        off_fit = fit_exponential_dwell(off_durations, n_bins=args.bins)
    else:
        on_fit = off_fit = None
    overall.update(fit_dict_to_columns(on_fit, prefix="on"))
    overall.update(fit_dict_to_columns(off_fit, prefix="off"))

    # Per-file breakdown (one row per source file), descriptive only.
    per_file_rows = []
    for source in sorted(source_files):
        file_events = [e for e in events if e.source_file == source]
        per_file_rows.append(summarize_events_extended(source, file_events))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "dwell_stats_summary.csv"
    per_file_path = output_dir / "dwell_stats_per_file.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(overall.keys()))
        writer.writeheader()
        writer.writerow(overall)

    with per_file_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(per_file_rows[0].keys()) if per_file_rows else []
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_file_rows)

    print(f"Read {len(events)} event(s) from {args.input} ({len(source_files)} file(s)).")
    print(
        f"ON: {len(on_durations)} events, mean={overall['on_mean_seconds']}s; "
        f"OFF: {len(off_durations)} events, mean={overall['off_mean_seconds']}s"
    )
    if not args.no_fit:
        on_rate = overall["on_rate_constant"]
        off_rate = overall["off_rate_constant"]
        print(
            f"Rate constants: on={on_rate} (leaves ON, ~k_off), "
            f"off={off_rate} (leaves OFF, ~k_on)"
        )
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote per-file breakdown to {per_file_path}")
    manifest_path = write_run_manifest(
        command="dwell-stats",
        parameters={"bins": args.bins, "no_fit": args.no_fit},
        input_paths=[Path(args.input)],
        output_paths=[summary_path, per_file_path],
        output_dir=output_dir,
    )
    print(f"Run manifest: {manifest_path}")


def cmd_gui(_args: argparse.Namespace) -> None:
    from frethmm.app.gui import run_gui

    run_gui()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "tdp":
        cmd_tdp(args)
    elif args.command == "review-grid":
        cmd_review_grid(args)
    elif args.command == "events":
        cmd_events(args)
    elif args.command == "dwell-stats":
        cmd_dwell_stats(args)
    elif args.command == "gui":
        cmd_gui(args)


if __name__ == "__main__":
    main()
