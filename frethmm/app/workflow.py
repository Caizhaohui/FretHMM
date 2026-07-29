from __future__ import annotations

from pathlib import Path


def default_review_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / f"{input_dir.name}_output"


def default_onoff_output_dir(reviewed_dir: Path) -> Path:
    return reviewed_dir.parent / f"{reviewed_dir.name}_ONOFF"


def versioned_output_dir(output_dir: Path) -> Path:
    version = 2
    while True:
        candidate = output_dir.parent / f"{output_dir.name}_v{version}"
        if not candidate.exists():
            return candidate
        version += 1
