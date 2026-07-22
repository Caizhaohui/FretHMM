from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from frethmm.domain.models import ClassificationConfig, ClassificationResult
from frethmm.viz.review_grid import plot_review_grid


def test_plot_review_grid_writes_png(tmp_path: Path):
    trace_file = tmp_path / "trace.csv"
    trace_file.write_text(
        "Time,signal\n"
        "0,0.1\n"
        "1,0.2\n"
        "2,0.8\n"
        "3,0.9\n",
        encoding="utf-8",
    )
    result = ClassificationResult(
        n_states=2,
        log_prob=-10.5,
        state_means=np.array([0.15, 0.85]),
        state_sigma=0.05,
        signal_sigma=0.3,
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
        state_path=np.array([0, 0, 1, 1]),
        classified_signal=np.array([0.15, 0.15, 0.85, 0.85]),
        fraction_spent=np.zeros((2, 2)),
        transitions_found=np.array([[0, 1], [0, 0]], dtype=int),
        filepath=trace_file,
    )

    output = tmp_path / "review.png"
    rendered = plot_review_grid(
        [result],
        ClassificationConfig(n_states=2, data_mode="single_channel"),
        output,
        rows=1,
        cols=1,
    )

    assert rendered == [output]
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_review_grid_uses_neutral_frame_when_result_has_warning(tmp_path: Path):
    trace_file = tmp_path / "trace.csv"
    trace_file.write_text(
        "Time,signal\n"
        "0,0.1\n"
        "1,0.2\n"
        "2,0.8\n"
        "3,0.9\n",
        encoding="utf-8",
    )
    result = ClassificationResult(
        n_states=2,
        log_prob=-10.5,
        state_means=np.array([0.15, 0.85]),
        state_sigma=0.05,
        signal_sigma=0.3,
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
        state_path=np.array([0, 0, 1, 1]),
        classified_signal=np.array([0.15, 0.15, 0.85, 0.85]),
        fraction_spent=np.zeros((2, 2)),
        transitions_found=np.array([[0, 1], [0, 0]], dtype=int),
        filepath=trace_file,
        warnings=["Applied low-state tail trim after 250s"],
    )

    output = tmp_path / "warning_review.png"
    plot_review_grid(
        [result],
        ClassificationConfig(n_states=2, data_mode="single_channel"),
        output,
        rows=1,
        cols=1,
    )

    image = mpimg.imread(output)
    orange = np.array([245, 124, 0], dtype=np.float32) / 255
    assert not np.any(np.all(np.isclose(image[:, :, :3], orange, atol=0.001), axis=2))


def test_plot_review_grid_paginates_outputs(tmp_path: Path):
    results = []
    for index in range(3):
        trace_file = tmp_path / f"trace_{index}.csv"
        trace_file.write_text(
            "Time,signal\n"
            "0,0.1\n"
            "1,0.2\n"
            "2,0.8\n"
            "3,0.9\n",
            encoding="utf-8",
        )
        results.append(
            ClassificationResult(
                n_states=2,
                log_prob=-10.5 - index,
                state_means=np.array([0.15, 0.85]),
                state_sigma=0.05,
                signal_sigma=0.3,
                transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
                state_path=np.array([0, 0, 1, 1]),
                classified_signal=np.array([0.15, 0.15, 0.85, 0.85]),
                fraction_spent=np.zeros((2, 2)),
                transitions_found=np.array([[0, 1], [0, 0]], dtype=int),
                filepath=trace_file,
            )
        )

    outputs = plot_review_grid(
        results,
        ClassificationConfig(n_states=2, data_mode="single_channel"),
        tmp_path / "review.png",
        rows=1,
        cols=2,
    )

    assert len(outputs) == 2
    assert outputs[0].name == "review_page_01.png"
    assert outputs[1].name == "review_page_02.png"
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
