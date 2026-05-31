"""Transition Density Plot (TDP) visualization."""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_reports(input_dir: Path) -> list[dict]:
    from pyhammi.io import read_report

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    reports = []
    for fp in sorted(input_dir.iterdir()):
        if fp.is_file() and "report" in fp.name.lower():
            try:
                reports.append(read_report(fp))
            except Exception:
                continue
    return reports


def aggregate_transitions(reports: list[dict]):
    all_starts = []
    all_stops = []
    all_weights = []

    for rep in reports:
        n = rep["n_states"]
        means = rep["means"]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ntrans = rep["transitions_found"][i, j]
                if ntrans > 0:
                    all_starts.append(means[i])
                    all_stops.append(means[j])
                    all_weights.append(int(ntrans))

    if not all_starts:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(all_starts),
        np.array(all_stops),
        np.array(all_weights),
    )


def _filter_reports_by_states(reports: list[dict], n_states: int) -> list[dict]:
    """Keep only the top-N most-visited states per report."""
    filtered = []
    for rep in reports:
        n = rep["n_states"]
        if n <= n_states:
            filtered.append(rep)
            continue
        # Pick top-n states by total transitions found
        means = rep["means"]
        tf = rep["transitions_found"]
        total_per_state = tf.sum(axis=1) + tf.sum(axis=0)
        top_indices = np.argsort(total_per_state)[-n_states:]
        top_indices = np.sort(top_indices)
        # Build filtered report
        new_rep = dict(rep)
        new_rep["n_states"] = n_states
        new_rep["means"] = means[top_indices]
        new_rep["transmat"] = rep["transmat"][np.ix_(top_indices, top_indices)]
        new_rep["transitions_found"] = tf[np.ix_(top_indices, top_indices)]
        new_rep["fraction_spent"] = rep["fraction_spent"][np.ix_(top_indices, top_indices)]
        filtered.append(new_rep)
    return filtered


def generate_tdp(
    input_dir: Path,
    exposure: float = 0.1,
    n_display_states: Optional[int] = None,
    output: Optional[str] = None,
) -> None:
    if not HAS_MPL:
        print("matplotlib is required for TDP. Install with: pip install matplotlib")
        return

    reports = load_reports(input_dir)
    if not reports:
        print(f"No report files found in {input_dir}")
        return

    # If n_display_states is set, filter to top-N states by total transitions
    if n_display_states is not None:
        reports = _filter_reports_by_states(reports, n_display_states)
        if not reports:
            print(f"No transitions found with {n_display_states} states filter.")
            return

    starts, stops, weights = aggregate_transitions(reports)
    if len(starts) == 0:
        print("No transitions found in reports.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    scatter = ax.scatter(
        starts,
        stops,
        s=weights * 5,
        c=weights,
        cmap="hot",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Number of transitions")

    all_vals = np.concatenate([starts, stops])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.1
    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            "k--", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Start FRET (before transition)")
    ax.set_ylabel("Stop FRET (after transition)")
    ax.set_title("Transition Density Plot")
    ax.set_aspect("equal")

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"TDP saved to {output}")
    else:
        plt.show()


def fit_gaussian_to_rates(
    reports: list[dict],
    start_state: int,
    stop_state: int,
    exposure: float = 0.1,
) -> Optional[dict]:
    from scipy.optimize import curve_fit

    rates = []
    for rep in reports:
        means = rep["means"]
        n = rep["n_states"]
        sorted_means = np.sort(means)

        if start_state >= n or stop_state >= n:
            continue

        start_fret = sorted_means[start_state]
        stop_fret = sorted_means[stop_state]

        i = int(np.argmin(np.abs(means - start_fret)))
        j = int(np.argmin(np.abs(means - stop_fret)))

        tp = rep["transmat"][i, j]
        if tp > 0:
            rate = tp / exposure
            rates.append(rate)

    if len(rates) < 3:
        return None

    rates = np.array(rates)
    hist, bin_edges = np.histogram(rates, bins=max(10, len(rates) // 3))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gauss(x, amp, mu, sig):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    try:
        p0 = [hist.max(), rates.mean(), rates.std()]
        popt, _ = curve_fit(gauss, bin_centers, hist, p0=p0)
        return {
            "amplitude": popt[0],
            "rate": popt[1],
            "rate_std": abs(popt[2]),
            "n_transitions": len(rates),
        }
    except Exception:
        return None
