# FretHMM

A single-molecule time-series Hidden Markov Model (HMM) state classification tool. Inspired by [HaMMy](https://github.com/Ha-SingleMoleculeLab/HaMMy), rewritten from scratch in Python with cross-platform support, batch processing, and a full GUI.

**[中文文档](README_zh.md)**

## Features

| Feature | Description |
|---------|-------------|
| HMM Engine | Baum-Welch training + Viterbi decoding (via hmmlearn), with customizable initial guesses |
| Data Modes | Auto-detect / Single-channel signal / Dual-channel Donor-Acceptor (auto-computes FRET efficiency) |
| Batch Processing | Multi-file parallel processing (`ProcessPoolExecutor`), directory scanning with multi-worker support |
| Review Grid | Batch classification + paginated multi-panel PNG visual review for quick quality screening |
| Low-State Tail Trimming | Two-pass HMM fitting that automatically identifies and trims persistent low-signal tails (e.g., photobleached states) |
| CLI | Four subcommands: `run`, `tdp`, `review-grid`, `gui` |
| GUI | CustomTkinter interface with dark/light themes, English/Chinese switching, threaded background analysis, and batch review grid export |
| Output Formats | `*_classified.csv`, `*_summary.json`, `*report.dat`, `*path.dat`, `*dwell.dat` (selectable in GUI) |
| TDP | Transition Density Plot visualization + Gaussian rate fitting |
| Packaging | PyInstaller one-click build for Windows executables (directory mode / `--onefile` mode) |

## Installation

```bash
git clone https://github.com/Caizhaohui/FretHMM.git
cd FretHMM
pip install -e .
```

**Requirements:**

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- hmmlearn >= 0.3.0
- matplotlib >= 3.7 (required for TDP and Review Grid visualization)
- customtkinter >= 5.2.0 (required for GUI)

**Optional dependencies:**

```bash
pip install -e ".[dev]"    # Install pytest testing framework
pip install -e ".[gui]"    # Install PyInstaller packaging tool
```

## Usage

### CLI

FretHMM provides four subcommands: `run` (HMM analysis), `review-grid` (visual review), `tdp` (transition density plot), and `gui` (graphical interface).

#### run — HMM State Classification

```bash
# Single file analysis (2 states, auto-detect data format)
frethmm run --files trace.csv --states 2 --output-dir ./results/

# Batch process all trace files in a directory (4 parallel workers)
frethmm run --input-dir ./traces/ --states 5 --workers 4 --output-dir ./results/

# Process multiple files at once
frethmm run --files trace1.csv trace2.csv trace3.csv --states 3 --output-dir ./results/

# Provide initial guesses (useful when state spacing is small)
frethmm run --files data.csv --states 2 --guesses "0.3,0.7"

# Specify single-channel mode and signal column
frethmm run --files data.csv --states 2 --mode single_channel --signal-column 1

# Use low-state tail trimming (trim persistent low-signal tails >= 5 seconds, then re-classify)
frethmm run --files trace.csv --states 2 --low-state-tail-trim-seconds 5.0

# Output only classified.csv
frethmm run --files data.csv --states 2 --classified-only

# Verbose mode (show all warnings)
frethmm run --files data.csv --states 3 -v
```

**`run` subcommand parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--files` | — | One or more trace file paths (mutually exclusive with `--input-dir`, required) |
| `--input-dir` | — | Input directory to scan for trace files (mutually exclusive with `--files`, required) |
| `--output-dir` | — | Output directory (defaults to input file directory) |
| `--states` | 2 | Number of HMM states |
| `--guesses` | None | Comma-separated initial signal guesses; count must match `--states` |
| `--max-iter` | 500 | Maximum Baum-Welch iterations |
| `--tol` | 1e-4 | Convergence tolerance |
| `--workers` | 1 | Number of parallel workers (>1 enables multiprocessing) |
| `--mode` | auto | Data mode: `auto` / `paired_channel` / `single_channel` |
| `--signal-column` | 1 | 1-based signal column index after Time for single_channel mode |
| `--low-state-tail-trim-seconds` | None | Low-state tail trim threshold in seconds (see [Data Filtering](#data-filtering-low-state-tail-trimming)) |
| `--classified-only` | off | Output only `*_classified.csv`, skip summary/report/path/dwell |
| `-v` / `--verbose` | off | Verbose output, show all warnings |

**Batch processing notes:**

- `--input-dir` scans all `.csv`, `.dat`, `.txt`, `.tsv` files, automatically skipping output files (`*report.dat`, `*path.dat`, `*dwell.dat`, `*_classified.csv`, `*_summary.json`)
- `--workers N` enables multi-process parallelism; N should not exceed CPU core count
- Individual file errors do not interrupt the overall batch; errors are printed to the terminal

#### review-grid — Batch Visual Review

```bash
# Basic: generate a 4×4 grid of 2-state traces
frethmm review-grid --input-dir ./traces/ --output review.png --states 2

# Custom grid layout
frethmm review-grid --input-dir ./traces/ --output review.png --states 3 --rows 5 --cols 6

# With initial guesses and output directory for classified CSVs
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --guesses "0.2,0.8" --output-dir ./classified/

# Combined with low-state tail trimming
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --low-state-tail-trim-seconds 5.0

# Accelerate with 4 parallel workers
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --workers 4 --rows 4 --cols 8
```

**`review-grid` subcommand parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | — | Input trace file directory (required) |
| `--output` | — | Output PNG path, e.g. `review.png` (required) |
| `--output-dir` | None | Optional directory for classified CSV side outputs |
| `--states` | 2 | Number of HMM states |
| `--guesses` | None | Comma-separated initial signal guesses |
| `--max-iter` | 500 | Maximum Baum-Welch iterations |
| `--tol` | 1e-4 | Convergence tolerance |
| `--workers` | 1 | Number of parallel workers |
| `--mode` | auto | Data mode: auto / paired_channel / single_channel |
| `--signal-column` | 1 | Signal column index for single_channel mode |
| `--low-state-tail-trim-seconds` | None | Low-state tail trim threshold in seconds |
| `--rows` | 4 | Panel rows per page |
| `--cols` | 4 | Panels per row |

**Pagination:** When the number of traces exceeds `rows × cols`, multiple page images are generated automatically (e.g., `review_page_01.png`, `review_page_02.png`). Each panel overlays the raw signal (gray) with the HMM classified signal (red). The title shows filename, log-likelihood, and state means. Traces with fitting warnings are highlighted with orange borders.

#### tdp — Transition Density Plot

```bash
# Generate TDP from report files (interactive window)
frethmm tdp --input-dir ./results/ --exposure 0.1

# Save to file
frethmm tdp --input-dir ./results/ --exposure 0.1 --output tdp.png

# Show only top N states (sorted by transition frequency)
frethmm tdp --input-dir ./results/ --exposure 0.1 --states 3 --output tdp.png
```

**`tdp` subcommand parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | — | Directory containing `*report.dat` files (required) |
| `--exposure` | 0.1 | Frame exposure time in seconds, used for rate calculations |
| `--states` | None | Show only top N states (sorted by transition frequency) |
| `--output` | None | Output image path (e.g., `tdp.png`); opens interactive window if not specified |

#### gui — Graphical Interface

```bash
frethmm gui
```

GUI screenshot (v1.0.0, with batch review grid panel):

![FretHMM GUI v1.0.0](docs/images/gui-v1.0.0-review-grid.png)

### GUI Guide

- **Menu bar**:
  - **File**: Add files, add folder, clear all, exit
  - **Settings**: HMM parameters dialog, language switch (English / Chinese), appearance mode (Light / Dark / System)
  - **Help**: About dialog
- **File selection**: Add `.csv` / `.dat` trace files via buttons or menu, or specify an input directory for batch processing
- **State folder batches**: Per-folder batch processing panel — add multiple folders, each with its own state count, data mode, and signal column
- **Parameters panel**: States, initial guesses, max iterations, tolerance, workers, data mode, signal column (displayed alongside output panel)
- **Output options**: Checkboxes to select output files — classified.csv / summary.json / report.dat / path.dat / dwell.dat
- **Review Grid section**: Dedicated area for setting rows, columns, and output filename. Click "Generate Review Grid" to produce the visual review image
- **Runtime panel**: Collapsible right sidebar (Show/Hide Runtime) showing real-time status, progress, run summary, and last output path
- **Result details**: Select a row in the results table to display full fitting metrics (states, log_prob, state means, sigma) and warnings in the right panel
- **Progress bar**: Real-time analysis task progress
- **Results table**: Fitting results for each file with color coding (green = OK, orange = warning, red = error)
- **Theme toggle**: Switch between Light / Dark / System via Settings menu or 🌓 button in the title bar
- **Bilingual support**: Real-time English / Chinese UI switching via Settings → Language
- **Threaded processing**: All analysis runs in a background thread with cancel support (Cancel button)
- **Log panel**: Color-coded log output (blue headers, orange warnings, red errors, green completion)
- **Status bar**: Current status and version number at the bottom

## Visualization

### Review Grid

Review Grid is a batch visualization tool for manual quality inspection. It classifies all traces in a directory via HMM and renders the results as a paginated multi-panel image.

**How it works:**

1. Scan all trace files in the input directory
2. Run HMM state classification on each file
3. Arrange results in a `rows × cols` grid of panels
4. Each panel overlays raw signal (gray thin line) with classified signal (red thick line)
5. Panel titles show filename, log-likelihood, and state means
6. Traces with fitting warnings are marked with orange borders for quick identification

**Output examples:**

```
review.png                     # Single page (trace count ≤ rows × cols)
review_page_01.png             # Auto-numbered pages when traces exceed one page
review_page_02.png
```

**Typical workflow:**

```bash
# 1. Quick quality review of all traces
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 --rows 4 --cols 8

# 2. Re-process problematic files individually
frethmm run --files traces/bad_trace.csv --states 3 --guesses "0.1,0.5,0.9" -v

# 3. After review passes, batch export full results
frethmm run --input-dir ./traces/ --states 2 --workers 4 --output-dir ./results/
```

### Transition Density Plot (TDP)

TDP aggregates transition information from all molecules (via `*report.dat` files) and renders a scatter density plot.

**Chart composition:**

- **X-axis**: Start state mean
- **Y-axis**: Stop state mean
- **Point size and color**: Encode transition count (using `hot` colormap; warmer = higher frequency)
- **Diagonal dashed line**: Self-transition reference

**`--states N` filtering**: When mixing datasets with different state counts, this parameter keeps only the top-N states per molecule (by total transition frequency) for cross-dataset comparison.

**Rate analysis**: Beyond visualization, FretHMM provides a `fit_gaussian_to_rates()` programming interface for Gaussian fitting on transition rate distributions between specific state pairs, extracting mean rate and standard deviation.

## Data Filtering

### Low-State Tail Trimming

**Background:** In single-molecule fluorescence experiments, trace tails often contain persistent low-signal segments (e.g., photobleached states, fluorophore inactivation). These tail artifacts are not genuine conformational states but are treated as an extra low-mean state by HMM, distorting the classification of real states.

**Two-pass fitting workflow:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1st Pass    │ ──→ │  Locate      │ ──→ │  Trim data   │
│  HMM on full │     │  lowest      │     │  at cutoff   │
│  trace       │     │  state run   │     │  point       │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  2nd Pass    │
                                          │  HMM on      │
                                          │  trimmed data│
                                          └──────────────┘
```

1. **First pass classification**: Fit HMM on the full trace to obtain the Viterbi state path
2. **Identify lowest state**: Find the state with the lowest mean value
3. **Detect persistent run**: Scan the time axis for the first contiguous segment of the lowest state lasting ≥ `--low-state-tail-trim-seconds`
4. **Trim data**: Truncate at that time point, discarding tail data
5. **Second pass classification**: Re-run HMM on the trimmed data for a cleaner classification

> **Note:** If the lowest state never persists beyond the threshold duration, no trimming is performed and the first-pass result is retained.

**CLI examples:**

```bash
# Single file: trim persistent low-signal tails >= 5 seconds
frethmm run --files trace.csv --states 2 --low-state-tail-trim-seconds 5.0

# Batch: 3-second threshold, 4 parallel workers
frethmm run --input-dir ./traces/ --states 3 --low-state-tail-trim-seconds 3.0 --workers 4

# Combined with Review Grid: trim then review
frethmm review-grid --input-dir ./traces/ --output review.png --states 2 \
    --low-state-tail-trim-seconds 5.0 --rows 4 --cols 8
```

**Output metadata:** When trimming is active, `*_summary.json` records these additional fields:

```json
{
  "low_state_tail_trim_seconds": 5.0,
  "low_state_tail_cutoff_time": 47.3,
  "low_state_tail_kept_frames": 473
}
```

- `low_state_tail_trim_seconds`: The configured trim threshold
- `low_state_tail_cutoff_time`: The actual cutoff time point (`null` if trimming was not triggered)
- `low_state_tail_kept_frames`: Number of frames retained after trimming

**GUI usage:** In the GUI output panel, find the "Low-state tail trim (seconds)" input field. Enter the threshold value before clicking Run Analysis. The trim setting is applied to all file and folder batch tasks.

## Input Format

The program auto-detects file format (header presence, delimiter type, column count). Two data modes are supported:

**Single-channel mode** (CSV, with header):

```csv
Time,channel1
0,2820
1,2884
2,2570
```

For multi-column signals, use `--signal-column` to select a specific column:

```csv
Time,channel1,channel2
0,2884,-5096
1,2884,1289
```

`--signal-column 1` uses the `channel1` column, `--signal-column 2` uses `channel2`.

**Dual-channel Donor/Acceptor mode** (whitespace/tab delimited, 3 columns, no header):

```
<time>  <donor>  <acceptor>
```

In this mode, FRET efficiency `A/(D+A)` is automatically computed as the HMM input signal.

## Output Files

Each input file generates the following outputs:

| File | Format | Description |
|------|--------|-------------|
| `*_classified.csv` | CSV | Primary output: `time, classified_mean` — the idealized trace |
| `*_summary.json` | JSON | State means, frame fractions, transition matrix, dwell statistics, trim metadata, warnings |
| `*report.dat` | Text | Model parameters (state count, means, sigma, transition probability matrix) |
| `*path.dat` | TSV | Raw signal channels + FRET signal + classified signal per frame |
| `*dwell.dat` | TSV | Dwell time table: `<start_mean> <stop_mean> <frames_lasted>` per dwell segment |

## Project Structure

```
FretHMM/
├── frethmm/
│   ├── __init__.py              # Version info
│   ├── app/
│   │   ├── cli.py               # CLI entry point (run / tdp / review-grid / gui)
│   │   ├── gui.py               # CustomTkinter GUI
│   │   └── i18n.py              # Internationalization (English / Chinese, 138 keys)
│   ├── assets/
│   │   ├── frethmm.ico          # Application icon
│   │   └── frethmm_logo.png     # Application logo
│   ├── core/
│   │   ├── io.py                # File I/O (trace reading + report output)
│   │   ├── model.py             # HMM engine (Baum-Welch + Viterbi + tail trimming)
│   │   ├── batch.py             # Multi-process batch processor
│   │   └── postprocess.py       # Classified signal + dwell segments + transition stats
│   ├── domain/
│   │   └── models.py            # Data models (Config / Trace / Result / ExportOptions)
│   ├── formats/
│   │   └── report_parser.py     # report.dat parser
│   ├── legacy/
│   │   └── report_parser.py     # Legacy report format parser
│   └── viz/
│       ├── review_grid.py       # Review Grid batch visualization (paginated layout)
│       └── tdp.py               # Transition Density Plot + Gaussian rate fitting
├── tests/
│   ├── fixtures/                # Regression test reference data
│   ├── test_io.py               # I/O and report parsing tests
│   ├── test_review_grid.py      # Review Grid visualization tests
│   └── test_golden.py           # CLI regression tests
├── docs/
│   ├── images/                  # Screenshots
│   └── FretHMM-refactor-plan.md # Development roadmap
├── pyproject.toml               # Project configuration
├── build_exe.py                 # PyInstaller build script
├── frethmm.spec                 # PyInstaller spec file
├── LICENSE                      # MIT License
└── README.md
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Packaging as Executable

```bash
# Directory mode (default, produces dist/FretHMM/ directory)
python build_exe.py

# Single-file mode (produces dist/FretHMM.exe, easy to distribute)
python build_exe.py --onefile
```

The build produces a standalone Windows GUI executable — no Python installation required. Single-file mode has a larger size but is more convenient for distribution.

## Changelog

### v1.1.0 (2026-06-09)

Documentation and release infrastructure update:

- **Enhanced README** with detailed visualization docs (Review Grid, TDP) and data filtering workflow (low-state tail trimming)
- **Added LICENSE** file (MIT)
- **Version bump** to 1.1.0 in `__init__.py` and `pyproject.toml`
- **Updated `.gitignore`** with additional exclusion rules

### v1.0.0 (2026-06-04)

Batch visual review release:

- **Batch review grid CLI**: New `review-grid` subcommand for batch classification with paginated grid overview
- **GUI review grid**: Dedicated Review Grid section in GUI for generating paginated review images
- **Paginated layout**: Configurable `rows × cols` grid, suitable for quick visual screening of 2-state, 3-state batches
- **Visual review enhancements**: Each panel overlays raw signal with classified trace; shows filename, `log_prob`, and `state means`

### v0.6.0 (2026-06-01)

GUI layout optimization and packaging improvements:

- **Layout restructure**: Removed ScrollableFrame; parameters and output panels side-by-side
- **Collapsible runtime panel**: Right sidebar defaults hidden; toggle via Show/Hide Runtime button
- **Application icon**: Added `frethmm.ico` and `frethmm_logo.png`
- **Window sizing**: Default 1280×720, minimum 1180×660
- **PyInstaller slimming**: Reduced EXE size by excluding unused packages
- **`--onefile` mode**: Added single-file EXE build support

### v0.5.0 (2026-06-01)

GUI stability and export options:

- **ExportOptions**: Fine-grained control over output file types
- **GUI output checkboxes**: Select which files to generate
- **Worker error handling**: Full traceback logging
- **Global exception hook**: Error dialogs even in `console=False` EXE mode

### v0.4.0 (2026-06-01)

CustomTkinter migration and CLI enhancements:

- **CustomTkinter migration**: Full dark/light/system theme support
- **`--classified-only`**: CLI flag to output only `*_classified.csv`
- **Folder batch panel**: Per-folder state count, data mode, and signal column
- **Runtime panel**: Real-time status, progress, and result details

### v0.3.0 (2026-06-01)

Project restructured as FretHMM with modular architecture:

- Modular packages: `core` / `domain` / `app` / `formats` / `legacy` / `viz`
- CLI with single-file and directory batch modes, multiprocessing support
- Full GUI with menu bar, parameters dialog, bilingual support, threaded analysis
- Default outputs: `*_classified.csv` + `*_summary.json`
- Additional outputs: `report.dat`, `path.dat`, `dwell.dat`
- TDP visualization + Gaussian rate fitting
- PyInstaller Windows executable build
- Regression test coverage (I/O, report parsing, end-to-end CLI)

### v0.2.0 (2026-06-01)

GUI major update:

- Menu bar (File / Settings / Help) and parameters dialog
- Bilingual support (i18n): English/Chinese real-time switching
- Modern UI: platform-adaptive fonts, custom ttk theme, color-coded logs
- Startup optimization: lazy loading of heavy libraries
- Warning handling: captured and displayed with orange highlighting

### v0.1.0 (2026-05-30)

Initial release:

- Complete HMM analysis pipeline (Baum-Welch training + Viterbi decoding)
- CLI tool (`run` / `tdp` / `gui` subcommands)
- tkinter GUI (file selection, parameters panel, progress bar, results table, log panel)
- Multi-process batch processing
- TDP visualization
- PyInstaller GUI packaging script

## License

[MIT License](LICENSE)
