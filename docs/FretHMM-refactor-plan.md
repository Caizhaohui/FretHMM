# FretHMM Refactor Plan

This document turns the current `pyHaMMy` prototype into an executable refactor plan for the next product shape: `FretHMM`.

Target product definition:

- Input-first workflow for two-column single-molecule traces: `time, signal`
- Core output as two-column classified trace: `time, classified_mean`
- Optional extended outputs: state summary, dwell statistics, transition statistics, diagnostic plots
- Preserve useful HaMMy ideas and sample compatibility, but do not limit the product to HaMMy's legacy naming, UI, or file formats

## 1. Refactor Goals

### Product goals

- [ ] Make two-column signal traces the default workflow
- [ ] Treat HMM state classification as the core feature, not FRET-specific report compatibility
- [ ] Provide a clean single-file workflow for `Values1.csv`-style data
- [ ] Support batch processing for experiment directories
- [ ] Keep legacy HaMMy-style exports only as optional compatibility artifacts

### Technical goals

- [ ] Separate domain model, algorithms, export, and UI concerns
- [ ] Reduce `FRET`-specific naming in core code
- [ ] Add stable regression coverage for `Values1.csv`
- [ ] Make GUI reflect the actual single-channel workflow
- [ ] Define a future-proof module layout for `FretHMM`

## 2. Current Baseline

Current reusable assets in this repo:

- `frethmm/cli.py`: working CLI entry
- `frethmm/gui.py`: working tkinter GUI
- `frethmm/model.py`: `hmmlearn`-based Gaussian HMM fitting
- `frethmm/io.py`: trace reading and report/path/dwell writing
- `frethmm/postprocess.py`: dwell extraction and transition stats
- `tests/test_golden.py`: golden coverage for `Values1.csv` and HaMMy sample reports

Current structural problems:

- Core naming is still FRET-oriented even for single-channel workflows
- The most important output for real use, `time + classified_mean`, is not the primary export
- GUI is still parameter-driven rather than trace-analysis-driven
- CLI and exports are organized around HaMMy compatibility rather than user workflow

## 3. Phase Plan

### Phase 0: Rebrand and stabilize scope

Outcome:

- `FretHMM` becomes the project name in docs and planning artifacts
- Current codebase remains runnable while refactor work proceeds

Tasks:

- [ ] Add an explicit statement that the codebase is in active refactor from HaMMy-style tooling to a general single-molecule signal classification tool
- [ ] Keep existing package/import path stable until a dedicated rename phase

Files:

- [ ] `README.md`
- [ ] `pyproject.toml`
- [ ] `frethmm/__init__.py`

Acceptance:

- [ ] Repo docs consistently describe the product as `FretHMM`
- [ ] Existing tests still pass after rebranding text changes

### Phase 1: Redefine the domain model around signal traces

Outcome:

- Core objects describe generic signal traces and classified states
- FRET-specific terminology becomes optional or compatibility-only

Tasks:

- [ ] Introduce a new domain language:
  - `SignalTrace`
  - `ClassificationResult`
  - `StateSummary`
  - `TransitionSummary`
- [ ] Deprecate direct core reliance on names like `fret`, `idealized_fret`, `signal_sigma` where generic names are more accurate
- [ ] Preserve compatibility adapters for donor/acceptor workflows

Files to change:

- [ ] `frethmm/config.py`
- [ ] `frethmm/postprocess.py`
- [ ] `frethmm/model.py`
- [ ] `frethmm/io.py`

Suggested target structure:

```text
domain/
  trace.py
  result.py
  config.py
```

Acceptance:

- [ ] Single-channel workflow can be understood from code without mentally translating from FRET naming
- [ ] Legacy compatibility exports still function through adapters

### Phase 2: Make two-column classified output the primary export

Outcome:

- The main user-facing output becomes `time, classified_mean`

Tasks:

- [ ] Define the primary export format:
  - filename: `*_classified.csv`
  - columns: `time, classified_mean`
- [ ] Add optional auxiliary exports:
  - `*_summary.json`
  - `*_states.csv`
  - `*_transitions.csv`
- [ ] Keep `report/path/dwell` as optional compatibility output, not primary output
- [ ] Ensure all single-channel runs write the two-column classified output automatically

Files to change:

- [ ] `frethmm/io.py`
- [ ] `frethmm/model.py`
- [ ] `frethmm/cli.py`
- [ ] `frethmm/gui.py`

New files to add:

- [ ] `frethmm/export.py`

Acceptance:

- [ ] Running the tool on `Values1.csv` writes `Values1_classified.csv`
- [ ] That file contains exactly two columns: `time`, `classified_mean`
- [ ] Existing regression fixtures for `Values1.csv` are updated or expanded intentionally

### Phase 3: Build a real single-trace analysis workflow

Outcome:

- One command handles the common case cleanly

CLI target:

```bash
frethmm classify --files Values1.csv
```

Tasks:

- [ ] Add a dedicated `classify` command for time-series classification
- [ ] Keep `run` as a compatibility alias during migration
- [ ] Add clear options for:
  - state count
  - initialization guesses
  - output directory
  - auto model selection
  - single-channel column selection
- [ ] Write a concise run summary at the end of processing

Files to change:

- [ ] `frethmm/cli.py`
- [ ] `frethmm/batch.py`

Acceptance:

- [ ] A default command for `Values1.csv` is obvious and short
- [ ] CLI text reflects signal classification, not only HMM fitting internals

### Phase 4: Algorithm hardening beyond a thin HaMMy rewrite

Outcome:

- Results become more stable, interpretable, and suitable for actual experiments

Tasks:

- [ ] Add multi-start fitting to reduce local optimum sensitivity
- [ ] Add AIC/BIC support for choosing state count
- [ ] Add optional post-fit segment cleanup:
  - minimum dwell merging
  - merge nearly identical states
- [ ] Add fit quality diagnostics
- [ ] Add explicit handling for edge cases:
  - constant traces
  - very short traces
  - NaN/Inf values
  - outlier spikes

Files to change:

- [ ] `frethmm/model.py`
- [ ] `frethmm/postprocess.py`
- [ ] `frethmm/config.py`

New files to add:

- [ ] `frethmm/metrics.py`
- [ ] `frethmm/preprocess.py`

Acceptance:

- [ ] Repeated runs on the same input are stable within defined tolerance
- [ ] Model selection is no longer fully manual
- [ ] Failure modes produce actionable warnings, not silent poor outputs

### Phase 5: Rebuild the GUI around the actual workflow

Outcome:

- GUI becomes a trace-analysis station, not just a parameter form

Target GUI layout:

- Left: file list and settings
- Center: raw signal plot with classified trace overlay
- Right: state statistics and transitions
- Bottom: logs and export actions

Tasks:

- [ ] Add raw trace preview
- [ ] Add classified trace overlay preview
- [ ] Add a summary panel for state means, occupancy, and transitions
- [ ] Add one-click export for `*_classified.csv`
- [ ] Keep advanced parameters available but secondary
- [ ] Make single-channel mode the default presentation

Files to change:

- [ ] `frethmm/gui.py`
- [ ] `frethmm/i18n.py`

New files to add:

- [ ] `frethmm/plots.py`

Acceptance:

- [ ] User can load `Values1.csv`, classify it, inspect the result, and export the two-column file without leaving the GUI
- [ ] GUI defaults make sense for two-column signal traces

### Phase 6: Batch and experiment-level analysis

Outcome:

- The tool scales from one trace to a directory of traces

Tasks:

- [ ] Add batch output folder conventions
- [ ] Write one classified file per input trace
- [ ] Add experiment-level summary table
- [ ] Add per-file QC status:
  - success
  - warning
  - failed fit
  - skipped

Files to change:

- [ ] `frethmm/batch.py`
- [ ] `frethmm/cli.py`
- [ ] `frethmm/gui.py`

New files to add:

- [ ] `frethmm/summary.py`

Acceptance:

- [ ] One directory can be processed in one command
- [ ] Failed files do not stop the batch
- [ ] Results are easy to inspect in Excel or pandas

### Phase 7: Verification, samples, and maintainability

Outcome:

- The refactor becomes safe to continue

Tasks:

- [ ] Keep HaMMy sample report parsing tests for compatibility coverage
- [ ] Promote `Values1.csv` to first-class regression sample
- [ ] Add regression outputs for:
  - classified CSV
  - summary JSON
  - optional plots metadata
- [ ] Add synthetic trace tests with known state means
- [ ] Add documentation for expected file formats and workflow

Files to change:

- [ ] `tests/test_golden.py`
- [ ] `tests/test_io.py`
- [ ] `README.md`

New tests to add:

- [ ] `tests/test_classify_cli.py`
- [ ] `tests/test_exports.py`
- [ ] `tests/test_preprocess.py`
- [ ] `tests/test_model_selection.py`

Acceptance:

- [ ] `Values1.csv` remains a stable end-to-end regression sample
- [ ] Refactor phases can proceed without losing the main workflow

## 4. File-by-File Task Map

### Existing files

`frethmm/cli.py`

- [ ] Rename user-facing language from "run HMM analysis" to "classify signal traces"
- [ ] Add `classify` subcommand
- [ ] Keep compatibility alias for old command names

`frethmm/gui.py`

- [ ] Promote single-channel trace workflow to default
- [ ] Add export button for two-column classified output
- [ ] Add trace preview and classified overlay

`frethmm/io.py`

- [ ] Split compatibility exports from primary exports
- [ ] Add `write_classified_csv`
- [ ] Add JSON and state-summary exports

`frethmm/model.py`

- [ ] Separate fitting, model selection, and result packaging
- [ ] Add multi-start fitting
- [ ] Add state-count scoring helpers

`frethmm/postprocess.py`

- [ ] Rename generic signal concepts away from FRET naming
- [ ] Add segment cleanup helpers

`frethmm/config.py`

- [ ] Split runtime config from domain result objects
- [ ] Add export and preprocessing config blocks

`frethmm/batch.py`

- [ ] Add summary aggregation
- [ ] Add consistent failure accounting

`frethmm/tdp.py`

- [ ] Mark as optional legacy/advanced analysis module
- [ ] Decide whether it stays under compatibility mode or moves to a separate advanced package area

`README.md`

- [ ] Rewrite around `FretHMM`
- [ ] Make `Values1.csv` the example data path
- [ ] Document the new primary outputs

### New files recommended

- [ ] `frethmm/export.py`
- [ ] `frethmm/preprocess.py`
- [ ] `frethmm/metrics.py`
- [ ] `frethmm/plots.py`
- [ ] `frethmm/summary.py`
- [ ] `tests/test_classify_cli.py`
- [ ] `tests/test_exports.py`
- [ ] `tests/test_preprocess.py`
- [ ] `tests/test_model_selection.py`

## 5. Target Package Layout

Recommended end-state layout:

```text
FretHMM/
  app/
    cli.py
    gui.py
  core/
    io.py
    preprocess.py
    model_hmm.py
    postprocess.py
    export.py
    metrics.py
    summary.py
  domain/
    trace.py
    result.py
    config.py
  viz/
    plots.py
  tests/
    test_io.py
    test_golden.py
    test_classify_cli.py
    test_exports.py
    test_preprocess.py
    test_model_selection.py
  samples/
    Values1.csv
```

Migration rule:

- Do not jump to this structure in one rename-only pass
- Move one concern at a time while preserving runnable behavior and test coverage

## 6. Execution Order

Recommended implementation order:

1. [ ] Rebrand docs and README to `FretHMM`
2. [ ] Add primary `*_classified.csv` export for two-column workflows
3. [ ] Add CLI regression coverage for `Values1.csv`
4. [ ] Refactor domain naming from `FRET` to generic `signal` concepts
5. [ ] Add summary JSON and state statistics export
6. [ ] Rework GUI around preview + export
7. [ ] Add model selection and multi-start fitting
8. [ ] Expand batch processing and QC

## 7. Definition of Done

The refactor is successful when all items below are true:

- [ ] `Values1.csv` is a documented first-class sample
- [ ] Default workflow accepts two-column `time, signal` input
- [ ] Default workflow produces two-column `time, classified_mean` output
- [ ] GUI supports the same workflow cleanly
- [ ] HaMMy sample compatibility remains only where it still adds value
- [ ] Tests cover both compatibility mode and the new primary workflow
- [ ] The project is described and structured as `FretHMM`, not as a partial Python port of HaMMy

## 8. Immediate Next Sprint

These are the highest-value tasks for the next implementation sprint:

- [ ] Update `README.md` and package metadata to `FretHMM`
- [ ] Implement `*_classified.csv` as the default single-channel export
- [ ] Add `tests/test_classify_cli.py` using `Values1.csv`
- [ ] Add summary export for state means and dwell stats
- [ ] Update GUI to expose direct export of the classified two-column result

