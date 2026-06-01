"""Report file parser for FretHMM outputs and archived reference files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def read_report_file(filepath: Union[str, Path]) -> dict:
    filepath = Path(filepath)
    text = filepath.read_text(encoding="ascii")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    n_states = None
    log_prob = None
    means = None
    sigma = None
    signal_sigma = None
    trans_entries = []
    for line in lines:
        if line.startswith("Number of states:"):
            parts = line.split("Max probability found:")
            n_states = int(parts[0].split(":")[1].strip())
            log_prob = float(parts[1].strip())
        elif line.startswith("State means:") or line.startswith("FRET peaks at:"):
            means = np.array([float(v) for v in line.split(":")[1].strip().split()])
        elif line.startswith("State sigma:") or line.startswith("FRET sigma:"):
            parts = line.split("Signal sigma:")
            sigma = float(parts[0].split(":")[1].strip())
            signal_sigma = float(parts[1].strip())
        elif line.startswith("Transition probability"):
            continue
        else:
            vals = line.split()
            if len(vals) == 5:
                try:
                    trans_entries.append([float(v) for v in vals])
                except ValueError:
                    continue
    if n_states is None or means is None:
        raise ValueError(f"Could not parse report file: {filepath}")
    transmat = np.zeros((n_states, n_states))
    fraction_spent = np.zeros((n_states, n_states))
    transitions_found = np.zeros((n_states, n_states), dtype=int)
    for start_mean, stop_mean, tp, frac, nf in trans_entries:
        i = int(np.argmin(np.abs(means - start_mean)))
        j = int(np.argmin(np.abs(means - stop_mean)))
        transmat[i, j] = tp
        fraction_spent[i, j] = frac
        transitions_found[i, j] = int(nf)
    return {
        "n_states": n_states,
        "log_prob": log_prob,
        "means": means,
        "sigma": sigma,
        "signal_sigma": signal_sigma,
        "transmat": transmat,
        "fraction_spent": fraction_spent,
        "transitions_found": transitions_found,
    }
