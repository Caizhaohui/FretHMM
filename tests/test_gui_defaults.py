from frethmm.app.cli import build_parser
from frethmm.app.gui import (
    DEFAULT_GUI_WORKERS,
    DEFAULT_LOW_STATE_TAIL_TRIM_SECONDS,
    resolve_event_classified_paths,
)
from frethmm.domain.models import ClassificationConfig


def test_gui_filter_default_matches_cli() -> None:
    args = build_parser().parse_args(["run", "--files", "trace.csv"])

    assert DEFAULT_LOW_STATE_TAIL_TRIM_SECONDS == args.low_state_tail_trim_seconds


def test_only_gui_workers_default_to_two() -> None:
    # Given / When
    args = build_parser().parse_args(["run", "--files", "trace.csv"])

    # Then
    assert DEFAULT_GUI_WORKERS == 2
    assert args.workers == 1
    assert ClassificationConfig().workers == 1


def test_gui_events_find_saved_classified_files_when_session_is_empty(tmp_path) -> None:
    # Given: a prior review-grid run wrote a classified CSV to the output folder.
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    classified = output_dir / "trace_classified.csv"
    classified.write_text("time,classified_mean\n0,1\n", encoding="utf-8")

    # When: the GUI has no in-memory results (for example after a review-grid run).
    paths = resolve_event_classified_paths({}, [output_dir])

    # Then: event extraction can use the saved classified result.
    assert paths == [classified]
