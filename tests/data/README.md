# Reproducibility Fixtures

These files are small, synthetic or de-identified fixtures that may be
committed with the source tree. They contain no raw microscopy images, ND2
files, participant data, or sample identifiers.

- `single_channel_trace.csv` exercises the documented `time,signal` workflow.
- `paired_channel_trace.csv` exercises donor/acceptor ratio handling.
- `legacy_reports/` contains compact HaMMy-compatible report files for parser
  and transition-density compatibility checks.

The fixtures are intentionally small. They make the core regression suite
fully runnable in CI; they are not representative experimental datasets.
