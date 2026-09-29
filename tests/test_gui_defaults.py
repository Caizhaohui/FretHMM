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


def test_gui_clean_options_defaults() -> None:
    import customtkinter as ctk
    from frethmm.app.gui import _App, _detect_fonts

    root = ctk.CTk()
    root.withdraw()
    try:
        app = _App(root, _detect_fonts())
        assert app._remove_spikes_var.get() is False
        assert app._spike_threshold_sigma_var.get() == 5.0
        assert app._trim_initial_artifacts_var.get() is False
        assert app._max_initial_artifact_frames_var.get() == 5
        assert app._smooth_window_var.get() == ""
        assert app._min_dwell_frames_var.get() == 1
        assert app._merge_state_threshold_var.get() == ""
        assert app._merge_state_sigma_factor_var.get() == ""
    finally:
        root.destroy()


def test_gui_build_config_with_custom_clean_options() -> None:
    import customtkinter as ctk
    from frethmm.app.gui import _App, _detect_fonts

    root = ctk.CTk()
    root.withdraw()
    try:
        app = _App(root, _detect_fonts())
        app.build()

        app._remove_spikes_var.set(True)
        app._spike_threshold_sigma_var.set(4.0)
        app._trim_initial_artifacts_var.set(True)
        app._max_initial_artifact_frames_var.set(10)
        app._smooth_window_var.set("3")
        app._min_dwell_frames_var.set(2)
        app._merge_state_threshold_var.set("0.05")
        app._merge_state_sigma_factor_var.set("1.5")

        cfg = app._build_config()
        assert cfg.remove_spikes is True
        assert cfg.spike_threshold_sigma == 4.0
        assert cfg.trim_initial_artifacts is True
        assert cfg.max_initial_artifact_frames == 10
        assert cfg.smooth_window == 3
        assert cfg.min_dwell_frames == 2
        assert cfg.merge_state_threshold == 0.05
        assert cfg.merge_state_sigma_factor == 1.5
        assert cfg.has_preprocessing is True
        assert cfg.has_postprocessing is True
    finally:
        root.destroy()


def test_gui_parse_clean_options_validation() -> None:
    import customtkinter as ctk
    import pytest
    from frethmm.app.gui import _App, _detect_fonts

    root = ctk.CTk()
    root.withdraw()
    try:
        app = _App(root, _detect_fonts())

        # Valid options
        parsed = app._parse_clean_options(
            remove_spikes=True,
            spike_threshold_sigma_val=4.5,
            trim_initial_artifacts=False,
            max_initial_artifact_frames_val=5,
            smooth_window_val="5",
            min_dwell_frames_val=3,
            merge_state_threshold_val="0.1",
            merge_state_sigma_factor_val="2.0",
        )
        assert parsed == (True, 4.5, False, 5, 5, 3, 0.1, 2.0)

        # Invalid spike threshold
        with pytest.raises(ValueError):
            app._parse_clean_options(
                remove_spikes=True,
                spike_threshold_sigma_val=-1.0,
                trim_initial_artifacts=False,
                max_initial_artifact_frames_val=5,
                smooth_window_val="",
                min_dwell_frames_val=1,
                merge_state_threshold_val="",
                merge_state_sigma_factor_val="",
            )

        # Invalid smooth window (even)
        with pytest.raises(ValueError):
            app._parse_clean_options(
                remove_spikes=False,
                spike_threshold_sigma_val=5.0,
                trim_initial_artifacts=False,
                max_initial_artifact_frames_val=5,
                smooth_window_val="4",
                min_dwell_frames_val=1,
                merge_state_threshold_val="",
                merge_state_sigma_factor_val="",
            )

        # Invalid min dwell (< 1)
        with pytest.raises(ValueError):
            app._parse_clean_options(
                remove_spikes=False,
                spike_threshold_sigma_val=5.0,
                trim_initial_artifacts=False,
                max_initial_artifact_frames_val=5,
                smooth_window_val="",
                min_dwell_frames_val=0,
                merge_state_threshold_val="",
                merge_state_sigma_factor_val="",
            )
    finally:
        root.destroy()


def test_gui_folder_job_config_propagates_clean_options() -> None:
    import customtkinter as ctk
    from frethmm.app.gui import _App, _FolderBatchJob, _detect_fonts

    root = ctk.CTk()
    root.withdraw()
    try:
        app = _App(root, _detect_fonts())
        app.build()

        job = _FolderBatchJob(
            folder="test_folder",
            n_states=2,
            max_iter=100,
            tol=1e-4,
            workers=1,
            data_mode="single_channel",
            signal_column=1,
            remove_spikes=True,
            spike_threshold_sigma=4.5,
            trim_initial_artifacts=True,
            max_initial_artifact_frames=8,
            smooth_window=5,
            min_dwell_frames=3,
            merge_state_threshold=0.1,
            merge_state_sigma_factor=2.0,
        )

        cfg = app._build_folder_job_config(job)
        assert cfg.remove_spikes is True
        assert cfg.spike_threshold_sigma == 4.5
        assert cfg.trim_initial_artifacts is True
        assert cfg.max_initial_artifact_frames == 8
        assert cfg.smooth_window == 5
        assert cfg.min_dwell_frames == 3
        assert cfg.merge_state_threshold == 0.1
        assert cfg.merge_state_sigma_factor == 2.0
        assert cfg.has_preprocessing is True
        assert cfg.has_postprocessing is True
    finally:
        root.destroy()
