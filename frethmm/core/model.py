"""Core HMM fitting workflow for signal classification."""

from __future__ import annotations

import warnings
from dataclasses import replace as dc_replace
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
from frethmm.core.metrics import count_gaussian_hmm_params, compute_aic, compute_bic
from frethmm.core.postprocess import build_classified_signal, compute_transition_stats
from frethmm.domain.models import (
    AUTO_STATES,
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


def _stable_seed(n_states: int, n_init: int, start_index: int) -> int:
    """Deterministic 32-bit seed for multi-start initial means.

    The seed depends only on the configuration, never on wall-clock time, so
    repeated runs on the same input reproduce identical initial guesses and
    therefore identical results (Baum-Welch is deterministic given the init).
    """
    raw = abs(hash((int(n_states), int(n_init), int(start_index)))) % (2**32)
    return int(raw)


def _generate_init_means(
    data_min: float,
    data_max: float,
    n_states: int,
    n_init: int,
    base_means: np.ndarray,
) -> np.ndarray:
    """Return a ``(n_init, n_states)`` array of candidate initial means.

    Row 0 is always ``base_means`` verbatim, so ``n_init == 1`` reproduces the
    historical single-fit behaviour exactly. Rows 1..n_init-1 perturb the base
    means with deterministic jitter drawn from a fixed-seed RNG. Perturbations
    are scaled to a fraction of the data range and never swap the ordering of
    the (sorted) base means, which keeps each candidate well-conditioned.
    """
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")
    base = np.asarray(base_means, dtype=np.float64).reshape(n_states)
    candidates = np.tile(base, (n_init, 1))
    if n_init == 1:
        return candidates

    data_range = float(data_max - data_min)
    if data_range < 1e-12:
        data_range = 1.0
    # Perturb up to ~20% of inter-state spacing, kept well inside the data span.
    span = data_range / max(n_states, 1)
    for start_index in range(1, n_init):
        rng = np.random.default_rng(_stable_seed(n_states, n_init, start_index))
        jitter = rng.uniform(-0.2, 0.2, size=n_states) * span
        candidates[start_index] = base + jitter

    # Guard against any candidate leaving the observed range or reordering
    # states (the base means are sorted ascending by construction).
    candidates = np.clip(candidates, data_min, data_max)
    candidates = np.sort(candidates, axis=1)
    candidates[0] = base  # row 0 stays exactly the legacy default
    return candidates


def _build_hmm(n_states: int, init_means: np.ndarray, data_range: float, config: ClassificationConfig):
    """Construct a fresh GaussianHMM seeded with ``init_means``.

    Each multi-start run needs its own unfitted model; ``hmmlearn`` fits are
    stateful, so we cannot reuse a single instance across starts.
    """
    from hmmlearn import hmm

    if data_range < 1e-10:
        data_range = 1.0
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="tied",
        n_iter=config.max_iter,
        tol=config.tol,
        params="stmc",
        init_params="",
    )
    model.means_ = np.asarray(init_means, dtype=np.float64).reshape(-1, 1)
    model.covars_ = np.full((1, 1), (data_range / (2 * n_states)) ** 2)
    transition_init = np.full((n_states, n_states), 1.0 / n_states)
    np.fill_diagonal(transition_init, 0.5)
    transition_init /= transition_init.sum(axis=1, keepdims=True)
    model.transmat_ = transition_init
    model.startprob_ = np.ones(n_states) / n_states
    return model


def fit_signal_hmm(
    trace: SignalTrace,
    config: ClassificationConfig,
) -> ClassificationResult:
    """Fit a Gaussian HMM with deterministic multi-start initialization.

    Runs Baum-Welch ``config.n_init`` times from different initial means and
    keeps the result with the highest log-likelihood. ``n_init == 1`` (and the
    first start of any run) uses the legacy evenly-spaced default means, so the
    historical single-fit output is reproduced exactly.
    """
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
    data_min = float(observations.min())
    data_max = float(observations.max())
    data_range = data_max - data_min
    if data_range < 1e-10:
        data_range = 1.0

    n_states = int(config._resolved_n_states())
    base_means = config.default_state_means(data_min, data_max)
    init_candidates = _generate_init_means(
        data_min, data_max, n_states, config.n_init, base_means
    )

    n_samples = trace.n_frames
    n_params = count_gaussian_hmm_params(n_states)

    best: Optional[dict] = None
    all_unconverged = True
    captured_warnings: list[str] = []
    seen_warning_messages: set[str] = set()

    for start_index in range(config.n_init):
        model = _build_hmm(n_states, init_candidates[start_index], data_range, config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(obs_2d)
            converged = bool(getattr(model, "monitor_", None) and model.monitor_.converged)
            if converged:
                all_unconverged = False
            log_prob = float(model.score(obs_2d))
            state_path = model.predict(obs_2d)
            for warning in caught:
                if warning.category in (DeprecationWarning, FutureWarning):
                    continue
                message = str(warning.message)
                if message not in seen_warning_messages:
                    seen_warning_messages.add(message)
                    captured_warnings.append(message)

        # Only a strictly better log-likelihood replaces the incumbent. The
        # ``start_index == 0`` check guarantees ties keep the legacy default.
        if best is None or log_prob > best["log_prob"]:
            best = {
                "log_prob": log_prob,
                "state_means": model.means_.flatten().copy(),
                "state_sigma": float(np.sqrt(model.covars_.flatten()[0])),
                "transmat": model.transmat_.copy(),
                "state_path": state_path.copy(),
                "converged": converged,
                "start_index": start_index,
            }

    assert best is not None  # n_init >= 1 enforced by ClassificationConfig
    if all_unconverged:
        warn_msg = (
            f"Baum-Welch did not converge in any of {config.n_init} start(s) "
            f"after {config.max_iter} iterations (n_states={n_states}). "
            "Results may be unreliable."
        )
        captured_warnings.append(warn_msg)

    state_means, transition_matrix, state_path = sort_state_outputs(
        best["state_means"], best["transmat"], best["state_path"]
    )
    classified_signal = build_classified_signal(state_path, state_means)
    fraction_spent, transitions_found = compute_transition_stats(
        state_path, n_states, trace.n_frames
    )
    signal_sigma = float(observations.std())

    return ClassificationResult(
        n_states=n_states,
        log_prob=best["log_prob"],
        state_means=state_means,
        state_sigma=best["state_sigma"],
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
        n_init=config.n_init,
        n_init_used=config.n_init,
        best_start_index=best["start_index"],
        bic=compute_bic(n_params, best["log_prob"], n_samples),
        aic=compute_aic(n_params, best["log_prob"]),
    )


def select_best_state_count(
    trace: SignalTrace,
    config: ClassificationConfig,
) -> ClassificationResult:
    """Pick the state count with the lowest BIC over ``[min_states, max_states]``.

    Used when ``config.n_states == "auto"``. Each candidate state count is fit
    with the full multi-start procedure, scored by BIC, and the best result is
    returned with all candidates recorded in ``model_candidates`` for
    transparency.
    """
    if not config.is_auto_states:
        raise ValueError(
            "select_best_state_count requires n_states == 'auto', "
            f"got {config.n_states!r}"
        )

    candidates: list[dict] = []
    best_result: Optional[ClassificationResult] = None
    best_bic = float("inf")

    for n_states in range(config.min_states, config.max_states + 1):
        candidate_config = dc_replace(config, n_states=n_states)
        result = fit_signal_hmm(trace, candidate_config)
        bic = result.bic if result.bic is not None else float("inf")
        candidates.append(
            {
                "n_states": n_states,
                "log_prob": float(result.log_prob),
                "bic": float(bic),
                "aic": float(result.aic) if result.aic is not None else None,
            }
        )
        if bic < best_bic:
            best_bic = bic
            best_result = result

    assert best_result is not None  # range is non-empty by config validation
    best_result.model_candidates = candidates
    return best_result


def trim_trace_after_low_state_tail(
    trace: SignalTrace,
    first_pass_result: ClassificationResult,
    duration_seconds: float,
) -> tuple[SignalTrace, Optional[float]]:
    """Trim only a persistent *terminal* lowest-state run.

    A low state can occur naturally in the middle of a trace. Treating the
    first long low segment as a bleaching tail discards valid signal and can
    leave too little data for the second fit. The trim therefore applies only
    when the final Viterbi state is the lowest state and that terminal run has
    lasted at least ``duration_seconds``.
    """
    lowest_state = int(np.argmin(first_pass_result.state_means))
    state_path = first_pass_result.state_path
    if len(state_path) == 0 or int(state_path[-1]) != lowest_state:
        return trace, None

    run_start_index = len(state_path) - 1
    while run_start_index > 0 and int(state_path[run_start_index - 1]) == lowest_state:
        run_start_index -= 1

    cutoff_time = float(trace.time[run_start_index] + duration_seconds)
    if trace.time[-1] < cutoff_time:
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

    def _fit(current_trace: SignalTrace) -> ClassificationResult:
        """Run the appropriate fitter, honoring BIC auto-selection."""
        if config.is_auto_states:
            return select_best_state_count(current_trace, config)
        return fit_signal_hmm(current_trace, config)

    if trim_seconds is not None:
        first_pass_result = _fit(trace)
        trace, cutoff_time = trim_trace_after_low_state_tail(
            trace,
            first_pass_result,
            trim_seconds,
        )
        result = _fit(trace)
        result.low_state_tail_trim_seconds = trim_seconds
        result.low_state_tail_cutoff_time = cutoff_time
        result.low_state_tail_kept_frames = trace.n_frames
        if cutoff_time is not None:
            result.warnings.append(
                "Applied low-state tail trim after "
                f"{trim_seconds:g}s (cutoff={cutoff_time:g}s, kept {trace.n_frames} frames)."
            )
    else:
        result = _fit(trace)
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
