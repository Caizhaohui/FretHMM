from __future__ import annotations

import inspect
import multiprocessing
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from frethmm.app import gui, gui_pool
from frethmm.domain.models import ClassificationConfig, ExportOptions


@dataclass(frozen=True, slots=True)
class _ControlledTask:
    value: int
    started_path: Path
    wait_path: Path | None
    release_path: Path | None
    deadline: float


@dataclass(frozen=True, slots=True)
class _ControlledOutcome:
    value: int
    worker_pid: int


def _controlled_worker(task: _ControlledTask) -> _ControlledOutcome:
    task.started_path.write_text(str(os.getpid()), encoding="utf-8")
    for gate_path in (task.wait_path, task.release_path):
        while gate_path is not None and not gate_path.exists():
            if time.monotonic() >= task.deadline:
                raise TimeoutError(task.value)
    return _ControlledOutcome(value=task.value, worker_pid=os.getpid())


def _raising_worker(task: _ControlledTask) -> _ControlledOutcome:
    task.started_path.write_text(str(os.getpid()), encoding="utf-8")
    raise RuntimeError(task.value)


def _write_valid_trace(path: Path) -> None:
    rows = ["Time,signal"]
    rows.extend(
        f"{index},{0.1 if (index // 10) % 2 == 0 else 0.9}"
        for index in range(40)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _classification_task(path: Path, output_dir: Path) -> gui_pool.ClassificationTask:
    return gui_pool.ClassificationTask(
        filepath=path,
        config=ClassificationConfig(
            n_states=2,
            max_iter=20,
            n_init=1,
            data_mode="single_channel",
        ),
        output_dir=output_dir,
        export_options=ExportOptions.classified_only(),
    )


def _controlled_tasks(tmp_path: Path, count: int) -> tuple[_ControlledTask, ...]:
    deadline = time.monotonic() + 15.0
    return tuple(
        _ControlledTask(
            value=index,
            started_path=tmp_path / f"started-{index}",
            wait_path=tmp_path / "started-0" if index == 1 else None,
            release_path=tmp_path / "release-0" if index == 0 else None,
            deadline=deadline,
        )
        for index in range(count)
    )


def _active_pool_pids() -> set[int]:
    return {child.pid for child in multiprocessing.active_children() if child.pid}


def test_classification_child_is_spawn_pickle_safe_and_isolates_file_errors(
    tmp_path: Path,
) -> None:
    # Given
    first = tmp_path / "first.csv"
    malformed = tmp_path / "malformed.csv"
    last = tmp_path / "last.csv"
    _write_valid_trace(first)
    malformed.write_text("Time,signal\nbad,data\n", encoding="utf-8")
    _write_valid_trace(last)
    output_dir = tmp_path / "output"
    tasks = tuple(
        _classification_task(path, output_dir)
        for path in (first, malformed, last)
    )
    cancellation = threading.Event()

    # When
    events = list(gui_pool.iter_classification_pool(tasks, 2, cancellation))

    # Then
    outcome_events = [
        event for event in events if isinstance(event, gui_pool.PoolOutcome)
    ]
    outcomes = [event.outcome for event in outcome_events]
    assert [event.index for event in outcome_events] == [0, 1, 2]
    assert isinstance(outcomes[0], gui_pool.ClassificationSuccess)
    assert isinstance(outcomes[1], gui_pool.ClassificationFailure)
    assert isinstance(outcomes[2], gui_pool.ClassificationSuccess)
    assert outcomes[1].filepath == malformed
    assert "could not convert string" in outcomes[1].message
    restored_outcomes = pickle.loads(pickle.dumps(outcomes))
    assert [type(outcome) for outcome in restored_outcomes] == [
        type(outcome) for outcome in outcomes
    ]
    assert [outcome.filepath for outcome in restored_outcomes] == [
        outcome.filepath for outcome in outcomes
    ]
    assert pickle.loads(pickle.dumps(tasks)) == tasks
    assert pickle.loads(pickle.dumps(gui_pool.classify_trace_task)) is (
        gui_pool.classify_trace_task
    )
    assert not inspect.ismethod(gui_pool.classify_trace_task)
    assert "<locals>" not in gui_pool.classify_trace_task.__qualname__
    assert (output_dir / "first_classified.csv").is_file()
    assert (output_dir / "last_classified.csv").is_file()
    assert _active_pool_pids().isdisjoint(
        {outcome.worker_pid for outcome in outcomes}
    )


def test_bounded_pool_emits_outcomes_in_task_order_after_inverse_completion(
    tmp_path: Path,
) -> None:
    # Given
    tasks = _controlled_tasks(tmp_path, count=4)
    request = gui_pool.PoolRequest(
        tasks=tasks,
        workers=2,
        worker=_controlled_worker,
    )
    cancellation = threading.Event()
    events = gui_pool.iter_bounded_pool(request, cancellation)

    # When
    first_event = next(events)
    started_before_release = sorted(tmp_path.glob("started-*"))
    (tmp_path / "release-0").touch()
    remaining_events = list(events)

    # Then
    assert first_event == gui_pool.PoolProgress(completed=1, total=4)
    assert len(started_before_release) == 2
    outcome_events = [
        event
        for event in remaining_events
        if isinstance(event, gui_pool.PoolOutcome)
    ]
    assert [event.index for event in outcome_events] == [0, 1, 2, 3]
    assert [event.outcome.value for event in outcome_events] == [0, 1, 2, 3]
    finished = remaining_events[-1]
    assert finished == gui_pool.PoolFinished(
        cancelled=False,
        submitted=4,
        completed=4,
        total=4,
        peak_in_flight=2,
    )


def test_bounded_pool_workers_one_uses_one_child(tmp_path: Path) -> None:
    # Given
    tasks = tuple(
        _ControlledTask(
            value=index,
            started_path=tmp_path / f"started-{index}",
            wait_path=None,
            release_path=None,
            deadline=time.monotonic() + 15.0,
        )
        for index in range(3)
    )
    request = gui_pool.PoolRequest(tasks=tasks, workers=1, worker=_controlled_worker)

    # When
    events = list(gui_pool.iter_bounded_pool(request, threading.Event()))

    # Then
    outcomes = [
        event.outcome
        for event in events
        if isinstance(event, gui_pool.PoolOutcome)
    ]
    assert len({outcome.worker_pid for outcome in outcomes}) == 1
    assert events[-1].peak_in_flight == 1


def test_cancellation_drains_active_children_without_new_submissions(
    tmp_path: Path,
) -> None:
    # Given
    tasks = _controlled_tasks(tmp_path, count=4)
    request = gui_pool.PoolRequest(tasks=tasks, workers=2, worker=_controlled_worker)
    cancellation = threading.Event()
    events = gui_pool.iter_bounded_pool(request, cancellation)

    # When
    first_event = next(events)
    cancellation.set()
    (tmp_path / "release-0").touch()
    remaining_events = list(events)

    # Then
    assert first_event == gui_pool.PoolProgress(completed=1, total=4)
    assert sorted(path.name for path in tmp_path.glob("started-*")) == [
        "started-0",
        "started-1",
    ]
    assert [
        event.index
        for event in remaining_events
        if isinstance(event, gui_pool.PoolOutcome)
    ] == [0, 1]
    finished = remaining_events[-1]
    assert finished.cancelled is True
    assert finished.submitted == 2
    assert finished.completed == 2
    worker_pids = {
        int(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("started-*")
    }
    assert _active_pool_pids().isdisjoint(worker_pids)


def test_worker_exception_still_shuts_down_pool_children(tmp_path: Path) -> None:
    # Given
    task = _ControlledTask(
        value=7,
        started_path=tmp_path / "started-error",
        wait_path=None,
        release_path=None,
        deadline=time.monotonic() + 15.0,
    )
    request = gui_pool.PoolRequest(
        tasks=(task,),
        workers=1,
        worker=_raising_worker,
    )

    # When
    with pytest.raises(RuntimeError, match="7"):
        list(gui_pool.iter_bounded_pool(request, threading.Event()))

    # Then
    worker_pid = int(task.started_path.read_text(encoding="utf-8"))
    assert worker_pid not in _active_pool_pids()


@pytest.mark.parametrize(
    ("requested", "tasks", "cpus", "expected"),
    [(4, 8, 2, 2), (4, 2, 8, 2), (1, 8, 8, 1), (2, 8, None, 1)],
)
def test_gui_effective_workers_respects_every_resource_limit(
    requested: int,
    tasks: int,
    cpus: int | None,
    expected: int,
) -> None:
    # Given / When
    effective = gui._effective_gui_workers(requested, tasks, cpus)

    # Then
    assert effective == expected


@pytest.mark.parametrize("raw_value", ["0", "-1", "2.5", "5"])
def test_gui_rejects_workers_outside_integer_range(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    # Given
    shown_errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui.messagebox,
        "showerror",
        lambda title, message: shown_errors.append((title, message)),
    )
    app = SimpleNamespace(
        _workers_entry=SimpleNamespace(get=lambda: raw_value),
        _t=lambda key, **_kwargs: key,
    )

    # When
    requested = gui._App._confirm_requested_workers(app)

    # Then
    assert requested is None
    assert len(shown_errors) == 1


@pytest.mark.parametrize("requested", [3, 4])
def test_gui_confirms_high_worker_counts(
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
) -> None:
    # Given
    confirmations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui.messagebox,
        "askyesno",
        lambda title, message: confirmations.append((title, message)) or False,
    )
    app = SimpleNamespace(
        _workers_entry=SimpleNamespace(get=lambda: str(requested)),
        _t=lambda key, **_kwargs: key,
    )

    # When
    accepted = gui._App._confirm_requested_workers(app)

    # Then
    assert accepted is None
    assert len(confirmations) == 1


@pytest.mark.parametrize("requested", [1, 2])
def test_gui_accepts_conservative_worker_counts_silently(
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
) -> None:
    # Given
    monkeypatch.setattr(
        gui.messagebox,
        "askyesno",
        lambda *_args: pytest.fail("confirmation must not be shown"),
    )
    app = SimpleNamespace(
        _workers_entry=SimpleNamespace(get=lambda: str(requested)),
        _t=lambda key, **_kwargs: key,
    )

    # When
    accepted = gui._App._confirm_requested_workers(app)

    # Then
    assert accepted == requested


def test_gui_analysis_worker_uses_pool_and_emits_cancelled_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    task = _classification_task(tmp_path / "trace.csv", tmp_path / "output")
    cancellation = threading.Event()
    result_queue: queue.Queue[gui._Msg] = queue.Queue()
    observed_workers: list[int] = []

    def fake_pool(tasks, workers, cancel_signal):
        observed_workers.append(workers)
        assert tuple(tasks) == (task,)
        assert cancel_signal is cancellation
        yield gui_pool.PoolFinished(
            cancelled=True,
            submitted=1,
            completed=1,
            total=1,
            peak_in_flight=1,
        )

    monkeypatch.setattr(gui_pool, "iter_classification_pool", fake_pool)

    # When
    gui._worker([task], 1, cancellation, result_queue)

    # Then
    assert observed_workers == [1]
    assert result_queue.get_nowait().type == gui._CANCELLED
    assert result_queue.empty()


def test_gui_analysis_worker_runs_real_spawn_pool_in_selection_order(
    tmp_path: Path,
) -> None:
    # Given
    first = tmp_path / "first.csv"
    malformed = tmp_path / "malformed.csv"
    last = tmp_path / "last.csv"
    _write_valid_trace(first)
    malformed.write_text("Time,signal\nbad,data\n", encoding="utf-8")
    _write_valid_trace(last)
    output_dir = tmp_path / "output"
    tasks = [
        _classification_task(path, output_dir)
        for path in (first, malformed, last)
    ]
    result_queue: queue.Queue[gui._Msg] = queue.Queue()

    # When
    gui._worker(tasks, 2, threading.Event(), result_queue)
    messages: list[gui._Msg] = []
    while not result_queue.empty():
        messages.append(result_queue.get_nowait())

    # Then
    result_paths = [
        Path(message.payload["filepath"])
        for message in messages
        if message.type == gui._RESULT
    ]
    progress = [
        message.payload
        for message in messages
        if message.type == gui._PROGRESS
    ]
    assert result_paths == [first, malformed, last]
    assert progress[-1] == {"current": 3, "total": 3}
    assert messages[-1].type == gui._DONE
    assert (output_dir / "first_classified.csv").is_file()
    assert (output_dir / "last_classified.csv").is_file()


def test_active_window_close_requests_cancel_before_destroy() -> None:
    # Given
    actions: list[str] = []
    app = SimpleNamespace(
        _worker_thread=object(),
        _closing=False,
        _cancel=lambda: actions.append("cancel"),
        root=SimpleNamespace(destroy=lambda: actions.append("destroy")),
    )

    # When
    gui._App._on_close(app)

    # Then
    assert app._closing is True
    assert actions == ["cancel"]


def test_cancelled_terminal_never_reports_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    actions: list[tuple[str, str | bool]] = []
    root = SimpleNamespace(
        after_cancel=lambda after_id: actions.append(("after_cancel", after_id)),
        destroy=lambda: actions.append(("root", "destroy")),
    )
    app = SimpleNamespace(
        _log=lambda message, _tag: actions.append(("log", message)),
        _t=lambda key, **_kwargs: key,
        _set_ui_running=lambda running: actions.append(("running", running)),
        _set_status=lambda key: actions.append(("status", key)),
        _worker_thread=object(),
        _after_id="poll-1",
        _closing=True,
        _result_stats={"ok": 2, "warnings": 0, "errors": 0},
        _runtime_summary_value=SimpleNamespace(configure=lambda **_kwargs: None),
        _last_output_path=None,
        root=root,
    )
    monkeypatch.setattr(
        gui.messagebox,
        "showinfo",
        lambda *_args: pytest.fail("completion dialog must not be shown"),
    )

    # When
    gui._App._finish_worker(app, cancelled=True)

    # Then
    assert ("status", "status_cancelled") in actions
    assert ("log", "log_complete") not in actions
    assert ("root", "destroy") in actions


def test_cancelled_review_skips_plotting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    task = _classification_task(tmp_path / "trace.csv", tmp_path / "output")
    cancellation = threading.Event()
    result_queue: queue.Queue[gui._Msg] = queue.Queue()
    monkeypatch.setattr(
        gui_pool,
        "iter_classification_pool",
        lambda *_args: iter(
            [
                gui_pool.PoolFinished(
                    cancelled=True,
                    submitted=1,
                    completed=1,
                    total=1,
                    peak_in_flight=1,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "frethmm.viz.review_grid.plot_review_grid",
        lambda *_args, **_kwargs: pytest.fail("plotting must be skipped"),
    )

    # When
    gui._review_worker(
        [task],
        1,
        "review_grid.png",
        2,
        2,
        gui._ReviewPublication(),
        cancellation,
        result_queue,
    )

    # Then
    assert result_queue.get_nowait().type == gui._CANCELLED
    assert result_queue.empty()


def test_review_cancelled_during_plot_discards_new_png_and_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_path = output_dir / "review_grid.png"
    output_path.write_bytes(b"pre-existing")
    task = _classification_task(tmp_path / "trace.csv", output_dir)
    cancellation = threading.Event()
    result_queue: queue.Queue[gui._Msg] = queue.Queue()
    result = SimpleNamespace(warnings=[])

    def fake_pool(tasks, workers, cancel_signal):
        assert tuple(tasks) == (task,)
        assert workers == 1
        assert cancel_signal is cancellation
        yield gui_pool.PoolOutcome(
            index=0,
            outcome=gui_pool.ClassificationSuccess(
                filepath=task.filepath,
                output_dir=output_dir,
                result=result,
                worker_pid=os.getpid(),
            ),
        )
        yield gui_pool.PoolFinished(
            cancelled=False,
            submitted=1,
            completed=1,
            total=1,
            peak_in_flight=1,
        )

    def fake_plot(results, config, output, **kwargs):
        del results, config, kwargs
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"newly rendered")
        cancellation.set()
        return [output]

    monkeypatch.setattr(gui_pool, "iter_classification_pool", fake_pool)
    monkeypatch.setattr("frethmm.viz.review_grid.plot_review_grid", fake_plot)

    # When
    gui._review_worker(
        [task],
        1,
        "review_grid.png",
        2,
        2,
        gui._ReviewPublication(),
        cancellation,
        result_queue,
    )

    # Then
    messages = []
    while not result_queue.empty():
        messages.append(result_queue.get_nowait())
    message_types = [message.type for message in messages]
    assert message_types[-1] == gui._CANCELLED
    assert gui._REVIEW_DONE not in message_types
    assert gui._DONE not in message_types
    assert output_path.read_bytes() == b"pre-existing"


def test_review_cancel_after_pre_publish_check_prevents_png_and_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    output_dir = tmp_path / "output"
    output_path = output_dir / "review_grid.png"
    task = _classification_task(tmp_path / "trace.csv", output_dir)
    cancellation = threading.Event()
    result_queue: queue.Queue[gui._Msg] = queue.Queue()
    publication = gui._ReviewPublication()
    after_pre_publish_check = threading.Event()
    release_worker = threading.Event()
    result = SimpleNamespace(warnings=[])

    def fake_pool(tasks, workers, cancel_signal):
        assert tuple(tasks) == (task,)
        assert workers == 1
        assert cancel_signal is cancellation
        yield gui_pool.PoolOutcome(
            index=0,
            outcome=gui_pool.ClassificationSuccess(
                filepath=task.filepath,
                output_dir=output_dir,
                result=result,
                worker_pid=os.getpid(),
            ),
        )
        yield gui_pool.PoolFinished(
            cancelled=False,
            submitted=1,
            completed=1,
            total=1,
            peak_in_flight=1,
        )

    def fake_plot(results, config, output, **kwargs):
        del results, config, kwargs
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"newly rendered")
        return [output]

    original_is_set = cancellation.is_set
    worker_thread: threading.Thread
    worker_checks = 0

    def gated_is_set() -> bool:
        nonlocal worker_checks
        value = original_is_set()
        if threading.current_thread() is worker_thread:
            worker_checks += 1
            if worker_checks == 2:
                after_pre_publish_check.set()
                assert release_worker.wait(timeout=5)
        return value

    monkeypatch.setattr(gui_pool, "iter_classification_pool", fake_pool)
    monkeypatch.setattr("frethmm.viz.review_grid.plot_review_grid", fake_plot)
    monkeypatch.setattr(cancellation, "is_set", gated_is_set)
    worker_thread = threading.Thread(
        target=gui._review_worker,
        args=(
            [task],
            1,
            "review_grid.png",
            2,
            2,
            publication,
            cancellation,
            result_queue,
        ),
    )
    app = SimpleNamespace(
        _cancel_event=cancellation,
        _review_publication=publication,
        _cancel_btn=SimpleNamespace(configure=lambda **_kwargs: None),
        _log=lambda *_args: None,
        _set_status=lambda *_args: None,
        _t=lambda key: key,
    )
    worker_thread.start()
    assert after_pre_publish_check.wait(timeout=5)

    # When
    gui._App._cancel(app)
    release_worker.set()
    worker_thread.join(timeout=5)

    # Then
    assert not worker_thread.is_alive()
    messages = []
    while not result_queue.empty():
        messages.append(result_queue.get_nowait())
    message_types = [message.type for message in messages]
    assert not output_path.exists()
    assert gui._REVIEW_DONE not in message_types
    assert message_types[-1] == gui._CANCELLED


def test_cancel_waiting_on_review_publication_commit_keeps_completed_terminal() -> None:
    # Given
    actions: list[tuple[str, bool]] = []
    cancellation = threading.Event()
    publication = gui._ReviewPublication()
    cancel_started = threading.Event()
    app = SimpleNamespace(
        _cancel_event=cancellation,
        _review_publication=publication,
        _cancel_btn=SimpleNamespace(configure=lambda **_kwargs: None),
        _log=lambda *_args: None,
        _set_status=lambda *_args: None,
        _t=lambda key: key,
        _closing=False,
        _finish_worker=lambda cancelled: actions.append(("finish", cancelled)),
    )

    def click_cancel() -> None:
        cancel_started.set()
        gui._App._cancel(app)

    # When
    with publication.lock:
        cancel_thread = threading.Thread(target=click_cancel)
        cancel_thread.start()
        assert cancel_started.wait(timeout=5)
        publication.committed = True
    cancel_thread.join(timeout=5)
    gui._App._handle_msg(app, gui._Msg(gui._DONE))

    # Then
    assert not cancel_thread.is_alive()
    assert not cancellation.is_set()
    assert actions == [("finish", False)]


def test_close_after_review_publication_commit_keeps_completed_terminal() -> None:
    # Given
    actions: list[tuple[str, bool]] = []
    publication = gui._ReviewPublication()
    publication.committed = True
    app = SimpleNamespace(
        _cancel_event=threading.Event(),
        _review_publication=publication,
        _closing=True,
        _finish_worker=lambda cancelled: actions.append(("finish", cancelled)),
    )

    # When
    gui._App._handle_msg(app, gui._Msg(gui._DONE))

    # Then
    assert actions == [("finish", False)]
