"""Write machine-readable provenance for command-line analysis runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from frethmm import __version__


_DEPENDENCIES = ("numpy", "scipy", "hmmlearn", "matplotlib", "customtkinter")


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _path_record(path: Path) -> dict[str, Any]:
    """Describe a path without reading or copying experimental data."""
    resolved = path.resolve()
    record: dict[str, Any] = {
        "name": path.name,
        "path": str(resolved),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["modified_at_utc"] = datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat()
    return record


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _next_manifest_path(output_dir: Path, created_at: datetime) -> Path:
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    candidate = output_dir / f"frethmm_run_manifest_{stamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"frethmm_run_manifest_{stamp}_{suffix}.json"
        suffix += 1
    return candidate


def write_run_manifest(
    *,
    command: str,
    parameters: Mapping[str, Any],
    input_paths: Iterable[Path],
    output_paths: Iterable[Path],
    output_dir: Path,
) -> Path:
    """Write a per-invocation manifest next to its generated outputs.

    The manifest records only file metadata, never input contents. It is kept
    separate from ``*_summary.json`` so legacy output compatibility remains
    unchanged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    manifest_path = _next_manifest_path(output_dir, created_at)
    payload = {
        "schema_version": 1,
        "created_at_utc": created_at.isoformat(),
        "application": {"name": "FretHMM", "version": __version__},
        "command": command,
        "parameters": _json_value(dict(parameters)),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": {
                package: _package_version(package) for package in _DEPENDENCIES
            },
        },
        "inputs": [_path_record(Path(path)) for path in input_paths],
        "outputs": [_path_record(Path(path)) for path in output_paths],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
