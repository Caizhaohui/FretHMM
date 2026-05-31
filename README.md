# pyHaMMy

Python rewrite of HaMMy: Hidden Markov Model analysis for single-molecule FRET trajectories.

## Overview

pyHaMMy treats single-molecule time-binned FRET trajectories (or single-channel intensity traces) as hidden Markov processes, determining the most likely state distributions and interconversion rates based on probability.

## Features

- **Baum-Welch** training (EM for HMMs) with tied-covariance Gaussian emissions
- **Viterbi** decoding for idealized state trajectories
- **Auto-detection** of data format (donor/acceptor pairs vs single-channel)
- **Batch processing** with multiprocessing support
- **TDP visualization** (Transition Density Plot) with Gaussian fitting
- Output files compatible with the original HaMMy tool

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Single file analysis (2 states)
pyhammi run --files trace.csv --states 2 --output-dir ./results/

# Batch analysis with parallel workers
pyhammi run --input-dir ./traces/ --states 5 --workers 4 --output-dir ./results/

# With initial guesses
pyhammi run --files data.csv --states 2 --guesses "0.3,0.7"

# Verbose mode
pyhammi run --files data.csv --states 3 -v

# TDP visualization
pyhammi tdp --input-dir ./results/ --exposure 0.1
```

## Input Format

ASCII text files with auto-detected format:

**Donor/Acceptor mode** (3+ columns, whitespace-separated):
```
<time>  <donor_intensity>  <acceptor_intensity>
```

**Single-channel mode** (CSV with header):
```csv
Time,channel1,channel2
0,2820,-5096
1,2820,1342
```

## Output Files

| File | Description |
|------|-------------|
| `*report.dat` | Model parameters: states, FRET peaks, sigma, transition matrix |
| `*path.dat` | Idealized trajectory per frame |
| `*dwell.dat` | Dwell times for each transition |

## Dependencies

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- hmmlearn >= 0.3.0
- matplotlib >= 3.7
