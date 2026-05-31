"""CLI entry point for pyHaMMy."""

import argparse
import sys
from pathlib import Path

from pyhammi.config import HMMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyhammi",
        description="pyHaMMy: Hidden Markov Model analysis for single-molecule FRET trajectories",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run = sub.add_parser("run", help="Run HMM analysis on trace files")
    run.add_argument("--states", type=int, default=2, help="Number of HMM states (default: 2)")
    run.add_argument("--guesses", type=str, default=None,
                     help="Comma-separated initial FRET/signal guesses, e.g. '0.3,0.7'")
    run.add_argument("--max-iter", type=int, default=500, help="Max Baum-Welch iterations (default: 500)")
    run.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (default: 1e-4)")
    run.add_argument("--workers", type=int, default=1, help="Parallel workers for batch mode (default: 1)")
    run.add_argument("--mode", choices=["auto", "fret", "donor_acceptor", "single_channel"],
                     default="auto", help="Data mode (default: auto)")
    run.add_argument("--signal-column", type=int, default=1,
                     help="Column index for single_channel mode (default: 1)")
    run.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    inp = run.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input-dir", type=str, help="Directory of trace files")
    inp.add_argument("--files", nargs="+", type=str, help="Individual trace files")
    run.add_argument("--output-dir", type=str, default=None, help="Output directory")

    tdp = sub.add_parser("tdp", help="Generate Transition Density Plot from report files")
    tdp.add_argument("--input-dir", type=str, required=True,
                     help="Directory containing *report.dat files")
    tdp.add_argument("--exposure", type=float, default=0.1,
                     help="Exposure time in seconds (default: 0.1)")
    tdp.add_argument("--states", type=int, default=None,
                     help="Number of states to display (default: auto)")
    tdp.add_argument("--output", type=str, default=None,
                     help="Output image file (default: show interactively)")

    gui = sub.add_parser("gui", help="Launch the pyHaMMy GUI")

    return parser


def cmd_run(args):
    import warnings
    from pyhammi.batch import process_batch, process_files

    if args.verbose:
        warnings.simplefilter("always")

    guesses = None
    if args.guesses:
        guesses = [float(g) for g in args.guesses.split(",")]

    config = HMMConfig(
        n_states=args.states,
        max_iter=args.max_iter,
        tol=args.tol,
        guesses=guesses,
        workers=args.workers,
        data_mode=args.mode,
        signal_column=args.signal_column,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.verbose:
        print(f"Config: states={config.n_states}, max_iter={config.max_iter}, "
              f"tol={config.tol}, mode={config.data_mode}, workers={config.workers}")

    if args.input_dir:
        results = process_batch(Path(args.input_dir), config, output_dir)
    else:
        files = [Path(f) for f in args.files]
        results = process_files(files, config, output_dir)

    print(f"\nDone. Processed {len(results)} file(s).")
    if results:
        for r in results:
            fp = r.filepath.name if r.filepath else "?"
            print(f"  {fp}: {r.n_states} states, log_prob={r.log_prob:.2f}, "
                  f"means={r.means}")
            for w in r.warnings:
                print(f"    WARNING: {w}")


def cmd_tdp(args):
    from pyhammi.tdp import generate_tdp

    generate_tdp(
        input_dir=Path(args.input_dir),
        exposure=args.exposure,
        n_display_states=args.states,
        output=args.output,
    )


def cmd_gui(args):
    from pyhammi.gui import run_gui
    run_gui()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "tdp":
        cmd_tdp(args)
    elif args.command == "gui":
        cmd_gui(args)


if __name__ == "__main__":
    main()
