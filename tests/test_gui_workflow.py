from pathlib import Path
from unittest.mock import patch

from frethmm.app.workflow import (
    default_onoff_output_dir,
    default_review_output_dir,
    versioned_output_dir,
)
from frethmm.app.gui import _App, _FolderBatchJob


def test_review_and_onoff_outputs_follow_selected_folder(tmp_path: Path):
    raw_folder = tmp_path / "experiment_a"
    raw_folder.mkdir()

    review_output = default_review_output_dir(raw_folder)

    assert review_output == tmp_path / "experiment_a_output"
    assert default_onoff_output_dir(review_output) == tmp_path / "experiment_a_output_ONOFF"


def test_versioned_output_uses_first_available_suffix(tmp_path: Path):
    output = tmp_path / "experiment_a_output"
    output.mkdir()
    (tmp_path / "experiment_a_output_v2").mkdir()

    assert versioned_output_dir(output) == tmp_path / "experiment_a_output_v3"


def test_folder_output_planning_cancels_without_removing_prior_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_output = default_review_output_dir(first)
    second_output = default_review_output_dir(second)
    first_output.mkdir()
    second_output.mkdir()

    app = object.__new__(_App)
    app.folder_jobs = [
        _FolderBatchJob(str(first), 2, 100, 1e-3, 1, "auto", 1),
        _FolderBatchJob(str(second), 2, 100, 1e-3, 1, "auto", 1),
    ]
    app._t = lambda key, **kwargs: key

    with (
        patch("frethmm.core.io.find_trace_files", return_value=[first / "trace.csv"]),
        patch("frethmm.app.gui.messagebox.askyesnocancel", side_effect=[True, None]),
    ):
        assert app._plan_folder_output_dirs() is None

    assert first_output.is_dir()
    assert second_output.is_dir()
