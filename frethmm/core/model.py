"""Core HMM fitting workflow for signal classification."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from frethmm.core.io import (
    read_signal_trace,
    write_classified_csv,
    write_dwell_report,
    write_state_path,
    write_state_report,
    write_summary_json,
)
from frethmm.core.postprocess import build_classified_signal, compute_transition_stats
from frethmm.domain.models import (
    ClassificationConfig,
    ClassificationResult,
    ExportOptions,
    SignalTrace,
)


def _resolve_export_options(
    export_options: Optional[ExportOptions],
    classified_only: Optional[bool],
) -> ExportOptions:
    if export_options is not None:
        return export_options
    if classified_only:
        return ExportOptions.classified_only()
    return ExportOptions()


def sort_state_outputs(
    state_means: np.ndarray,
    transition_matrix: np.ndarray,
    state_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(state_means)
    remap = np.empty_like(sorted_indices)
    remap[sorted_indices] = np.arange(len(sorted_indices))
    return (
        state_means[sorted_indices],
        transition_matrix[np.ix_(sorted_indices, sorted_indices)],
        remap[state_path],
    )


def fit_signal_hmm(
    trace: SignalTrace,
    config: ClassificationConfig,
) -> ClassificationResult:
    from hmmlearn import hmm

    observations = trace.observations.astype(np.float64)
    if np.any(np.isnan(observations)):
        observations = np.nan_to_num(observations, nan=0.0)
    if np.any(np.isinf(observations)):
        observations = np.nan_to_num(
            observations,
            nan=0.0,
            posinf=observations.max(),
            neginf=observations.min(),
        )
    obs_2d = observations.reshape(-1, 1)
    data_min = observations.min()
    data_max = observations.max()
    data_range = data_max - data_min
    if data_range < 1e-10:
        data_range = 1.0

    model = hmm.GaussianHMM(
        n_components=config.n_states,
        covariance_type="tied",
        n_iter=config.max_iter,
        tol=config.tol,
        params="stmc",
        init_params="",
    )
    model.means_ = config.default_state_means(data_min, data_max).reshape(-1, 1)
    model.covars_ = np.full((1, 1), (data_range / (2 * config.n_states)) ** 2)
    transition_init = np.full((config.n_states, config.n_states), 1.0 / config.n_states)
    np.fill_diagonal(transition_init, 0.5)
    transition_init /= transition_init.sum(axis=1, keepdims=True)
    model.transmat_ = transition_init
    model.startprob_ = np.ones(config.n_states) / config.n_states

    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(obs_2d)
        if hasattr(model, "monitor_") and not model.monitor_.converged:
            warnings.warn(
                f"Baum-Welch did not converge after {config.max_iter} iterations "
                f"(n_states={config.n_states}). Results may be unreliable.",
                stacklevel=2,
            )
        log_prob = model.score(obs_2d)
        state_path = model.predict(obs_2d)
        for warning in caught:
            if warning.category not in (DeprecationWarning, FutureWarning):
                captured_warnings.append(str(warning.message))

    state_means = model.means_.flatten()
    state_sigma = float(np.sqrt(model.covars_.flatten()[0]))
    signal_sigma = float(observations.std())
    state_means, transition_matrix, state_path = sort_state_outputs(
        state_means,
        model.transmat_,
        state_path,
    )
    classified_signal = build_classified_signal(state_path, state_means)
    fraction_spent, transitions_found = compute_transition_stats(
        state_path,
        config.n_states,
        trace.n_frames,
    )
    return ClassificationResult(
        n_states=config.n_states,
        log_prob=log_prob,
        state_means=state_means,
        state_sigma=state_sigma,
        signal_sigma=signal_sigma,
        transition_matrix=transition_matrix,
        state_path=state_path,
        classified_signal=classified_signal,
        fraction_spent=fraction_spent,
        transitions_found=transitions_found,
        filepath=trace.filepath,
        warnings=captured_warnings,
        trace_time=trace.time.copy(),
        trace_signal=trace.signal.copy(),
    )


def trim_trace_after_low_state_tail(
    trace: SignalTrace,
    first_pass_result: ClassificationResult,
    duration_seconds: float,
) -> tuple[SignalTrace, Optional[float]]:
    lowest_state = int(np.argmin(first_pass_result.state_means))
    run_start_index: Optional[int] = None
    cutoff_time: Optional[float] = None

    for index, state in enumerate(first_pass_result.state_path):
        if int(state) == lowest_state:
            if run_start_index is None:
                run_start_index = index
            elapsed = trace.time[index] - trace.time[run_start_index]
            if elapsed >= duration_seconds:
                cutoff_time = float(trace.time[run_start_index] + duration_seconds)
                break
        else:
            run_start_index = None

    if cutoff_time is None:
        return trace, None

    keep_mask = trace.time <= cutoff_time
    if np.all(keep_mask):
        return trace, cutoff_time

    return (
        SignalTrace(
            time=trace.time[keep_mask].copy(),
            signal=trace.signal[keep_mask].copy(),
            observations=trace.observations[keep_mask].copy(),
            filepath=trace.filepath,
            mode=trace.mode,
            channel_1=trace.channel_1[keep_mask].copy() if trace.channel_1 is not None else None,
            channel_2=trace.channel_2[keep_mask].copy() if trace.channel_2 is not None else None,
            derived_signal=(
                trace.derived_signal[keep_mask].copy()
                if trace.derived_signal is not None
                else None
            ),
        ),
        cutoff_time,
    )


def process_trace_file(
    filepath: Path,
    config: ClassificationConfig,
    output_dir: Optional[Path] = None,
    classified_only: Optional[bool] = None,
    export_options: Optional[ExportOptions] = None,
) -> ClassificationResult:
    trace = read_signal_trace(filepath, mode=config.data_mode, signal_column=config.signal_column)
    trim_seconds = config.low_state_tail_trim_seconds
    if trim_seconds is not None:
        first_pass_result = fit_signal_hmm(trace, config)
        trace, cutoff_time = trim_trace_after_low_state_tail(
            trace,
            first_pass_result,
            trim_seconds,
        )
        result = fit_signal_hmm(trace, config)
        result.low_state_tail_trim_seconds = trim_seconds
        result.low_state_tail_cutoff_time = cutoff_time
        result.low_state_tail_kept_frames = trace.n_frames
    else:
        result = fit_signal_hmm(trace, config)
    exports = _resolve_export_options(export_options, classified_only)
    if exports.classified_csv:
        write_classified_csv(trace, result, output_dir)
    if exports.summary_json:
        write_summary_json(result, output_dir)
    if exports.state_report:
        write_state_report(result, output_dir)
    if exports.state_path:
        write_state_path(trace, result, output_dir)
    if exports.dwell_report:
        write_dwell_report(result, output_dir)
    return result
