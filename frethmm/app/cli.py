"""Primary CLI entry point for FretHMM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frethmm.core.batch import process_batch, process_files
from frethmm.domain.models import ClassificationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frethmm",
        description="FretHMM: Hidden Markov Model state classification for single-molecule trajectories",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run = sub.add_parser("run", help="Run HMM state classification on trace files")
    run.add_argument("--states", type=int, default=2, help="Number of HMM states (default: 2)")
    run.add_argument("--guesses", type=str, default=None, help="Comma-separated initial signal guesses")
    run.add_argument("--max-iter", type=int, default=500, help="Max Baum-Welch iterations (default: 500)")
    run.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    run.add_argument("--workers", type=int, default=1, help="Parallel workers for batch mode (default: 1)")
    run.add_argument("--mode", choices=["auto", "paired_channel", "single_channel"], default="auto")
    run.add_argument(
        "--signal-column",
        type=int,
        default=1,
        help="1-based signal column index after Time for single_channel mode (default: 1)",
    )
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
    review.add_argument("--states", type=int, default=2, help="Number of HMM states (default: 2)")
    review.add_argument("--guesses", type=str, default=None, help="Comma-separated initial signal guesses")
    review.add_argument("--max-iter", type=int, default=500, help="Max Baum-Welch iterations (default: 500)")
    review.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    review.add_argument("--workers", type=int, default=1, help="Parallel workers for batch mode (default: 1)")
    review.add_argument("--mode", choices=["auto", "paired_channel", "single_channel"], default="auto")
    review.add_argument(
        "--signal-column",
        type=int,
        default=1,
        help="1-based signal column index after Time for single_channel mode (default: 1)",
    )
    review.add_argument("--rows", type=int, default=4, help="Number of panel rows per review page")
    review.add_argument("--cols", type=int, default=4, help="Number of panels per row in the review grid")

    sub.add_parser("gui", help="Launch the FretHMM GUI")
    return parser


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
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
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
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
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
    elif args.command == "gui":
        cmd_gui(args)


if __name__ == "__main__":
    main()
