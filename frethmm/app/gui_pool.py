from __future__ import annotations

import multiprocessing
import os
import traceback
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeAlias, TypeVar

from frethmm.domain.models import (
    ClassificationConfig,
    ClassificationResult,
    ExportOptions,
)


TaskT = TypeVar("TaskT")
OutcomeT = TypeVar("OutcomeT")


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class InvalidWorkerCountError(ValueError):
    workers: int

    def __str__(self) -> str:
        return f"workers must be at least 1, got {self.workers}"


@dataclass(frozen=True, slots=True)
class ClassificationTask:
    filepath: Path
    config: ClassificationConfig
    output_dir: Path | None
    export_options: ExportOptions


@dataclass(frozen=True, slots=True)
class ClassificationSuccess:
    filepath: Path
    output_dir: Path | None
    result: ClassificationResult
    worker_pid: int


@dataclass(frozen=True, slots=True)
class ClassificationFailure:
    filepath: Path
    output_dir: Path | None
    error_type: str
    message: str
    traceback: str
    worker_pid: int


ClassificationOutcome: TypeAlias = ClassificationSuccess | ClassificationFailure


@dataclass(frozen=True, slots=True)
class PoolRequest(Generic[TaskT, OutcomeT]):
    tasks: tuple[TaskT, ...]
    workers: int
    worker: Callable[[TaskT], OutcomeT]

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise InvalidWorkerCountError(workers=self.workers)


@dataclass(frozen=True, slots=True)
class PoolProgress:
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class PoolOutcome(Generic[OutcomeT]):
    index: int
    outcome: OutcomeT


@dataclass(frozen=True, slots=True)
class PoolFinished:
    cancelled: bool
    submitted: int
    completed: int
    total: int
    peak_in_flight: int


PoolEvent: TypeAlias = PoolProgress | PoolOutcome[OutcomeT] | PoolFinished


def classify_trace_task(task: ClassificationTask) -> ClassificationOutcome:
    from frethmm.core.model import process_trace_file

    try:
        result = process_trace_file(
            task.filepath,
            task.config,
            task.output_dir,
            export_options=task.export_options,
        )
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        return ClassificationFailure(
            filepath=task.filepath,
            output_dir=task.output_dir,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            worker_pid=os.getpid(),
        )
    return ClassificationSuccess(
        filepath=task.filepath,
        output_dir=task.output_dir,
        result=result,
        worker_pid=os.getpid(),
    )


def iter_bounded_pool(
    request: PoolRequest[TaskT, OutcomeT],
    cancellation: CancellationSignal,
) -> Iterator[PoolEvent[OutcomeT]]:
    """Yield completion progress and task-ordered outcomes from a spawn pool."""
    total = len(request.tasks)
    cancelled = cancellation.is_set()
    if total == 0 or cancelled:
        yield PoolFinished(
            cancelled=cancelled,
            submitted=0,
            completed=0,
            total=total,
            peak_in_flight=0,
        )
        return

    worker_limit = min(request.workers, total)
    submitted = 0
    completed = 0
    peak_in_flight = 0
    next_outcome_index = 0
    buffered_outcomes: dict[int, OutcomeT] = {}
    active: dict[Future[OutcomeT], int] = {}

    with ProcessPoolExecutor(
        max_workers=worker_limit,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        while submitted < worker_limit:
            if cancellation.is_set():
                cancelled = True
                break
            active[
                executor.submit(request.worker, request.tasks[submitted])
            ] = submitted
            submitted += 1
        peak_in_flight = len(active)

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=active.__getitem__):
                index = active.pop(future)
                buffered_outcomes[index] = future.result()
                completed += 1
                cancelled = cancelled or cancellation.is_set()
                yield PoolProgress(completed=completed, total=total)
                cancelled = cancelled or cancellation.is_set()

                while next_outcome_index in buffered_outcomes:
                    yield PoolOutcome(
                        index=next_outcome_index,
                        outcome=buffered_outcomes.pop(next_outcome_index),
                    )
                    next_outcome_index += 1
                    cancelled = cancelled or cancellation.is_set()

                if not cancelled and submitted < total:
                    if cancellation.is_set():
                        cancelled = True
                    else:
                        active[
                            executor.submit(
                                request.worker,
                                request.tasks[submitted],
                            )
                        ] = submitted
                        submitted += 1
                        peak_in_flight = max(peak_in_flight, len(active))

    cancelled = cancelled or cancellation.is_set() or submitted < total
    yield PoolFinished(
        cancelled=cancelled,
        submitted=submitted,
        completed=completed,
        total=total,
        peak_in_flight=peak_in_flight,
    )


def iter_classification_pool(
    tasks: Sequence[ClassificationTask],
    workers: int,
    cancellation: CancellationSignal,
) -> Iterator[PoolEvent[ClassificationOutcome]]:
    request = PoolRequest(
        tasks=tuple(tasks),
        workers=workers,
        worker=classify_trace_task,
    )
    yield from iter_bounded_pool(request, cancellation)
