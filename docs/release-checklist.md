# FretHMM Release Checklist

Use this checklist to release an already reviewed version. Build artifacts must
come from the exact commit that receives the version tag.

1. Confirm `frethmm.__version__` and `pyproject.toml` have the intended version.
2. Run `python -m pytest -q` from a clean environment after `pip install -e ".[dev]"`.
3. Install the GUI build dependency with `pip install -e ".[gui]"` and run
   `python build_exe.py`.
4. Verify `dist/FretHMM/FretHMM.exe --version` prints the intended FretHMM
   version without opening a window.
5. Verify `dist/FretHMM.zip.sha256` against `dist/FretHMM.zip` and inspect the
   adjacent JSON manifest.
6. Commit all source, fixture, documentation, and CI changes; do not commit
   `dist/`, temporary outputs, or local run manifests.
7. Create the annotated Git tag and upload the ZIP, SHA-256 sidecar, and JSON
   manifest to the release.
