"""HMM engine: Baum-Welch training + Viterbi decoding using hmmlearn."""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from hmmlearn import hmm

from pyhammi.config import HMMConfig, HMMResult, TraceData
from pyhammi.postprocess import (
    build_idealized_fret,
    compute_transition_stats,
)


def fit_hmm(
    trace: TraceData,
    config: HMMConfig,
) -> HMMResult:
    obs = trace.observations.astype(np.float64)
    if np.any(np.isnan(obs)):
        obs = np.nan_to_num(obs, nan=0.0)
    if np.any(np.isinf(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=obs.max(), neginf=obs.min())

    obs_2d = obs.reshape(-1, 1)

    data_min = obs.min()
    data_max = obs.max()
    data_range = data_max - data_min
    if data_range < 1e-10:
        data_range = 1.0

    means_init = config.default_means(data_min, data_max)
    means_init = means_init.reshape(-1, 1).astype(np.float64)

    model = hmm.GaussianHMM(
        n_components=config.n_states,
        covariance_type="tied",
        n_iter=config.max_iter,
        tol=config.tol,
        params="stmc",
        init_params="",
    )
    model.means_ = means_init
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
                f"(n_states={config.n_states}). Results may be unreliable. "
                f"Consider increasing --max-iter.",
                stacklevel=2,
            )

        log_prob = model.score(obs_2d)
        viterbi_path = model.predict(obs_2d)

        for w in caught:
            if w.category in (DeprecationWarning, FutureWarning):
                continue
            captured_warnings.append(str(w.message))

    means = model.means_.flatten()
    sigma = float(np.sqrt(model.covars_.flatten()[0]))
    signal_sigma = float(obs.std())

    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    transmat_sorted = model.transmat_[np.ix_(sorted_indices, sorted_indices)]
    viterbi_path_sorted = np.array([np.searchsorted(sorted_indices, v) for v in viterbi_path])

    idealized = build_idealized_fret(viterbi_path_sorted, means)
    fraction_spent, transitions_found = compute_transition_stats(
        viterbi_path_sorted, config.n_states, trace.n_frames
    )

    return HMMResult(
        n_states=config.n_states,
        log_prob=log_prob,
        means=means,
        sigma=sigma,
        signal_sigma=signal_sigma,
        transmat=transmat_sorted,
        viterbi_path=viterbi_path_sorted,
        idealized_fret=idealized,
        fraction_spent=fraction_spent,
        transitions_found=transitions_found,
        filepath=trace.filepath,
        warnings=captured_warnings,
    )


def process_file(
    filepath: Path,
    config: HMMConfig,
    output_dir: Optional[Path] = None,
) -> HMMResult:
    from pyhammi.io import read_trace, write_report, write_path, write_dwell

    trace = read_trace(filepath, mode=config.data_mode, signal_column=config.signal_column)
    result = fit_hmm(trace, config)

    write_report(result, output_dir)
    write_path(trace, result, output_dir)
    write_dwell(result, output_dir)

    return result
