import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_release_metadata(project_root: Path, artifact: Path) -> tuple[Path, Path]:
    from frethmm import __version__

    checksum = _sha256(artifact)
    checksum_path = artifact.with_suffix(artifact.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {artifact.name}\n", encoding="ascii")
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "application": {"name": "FretHMM", "version": __version__},
                "artifact": {
                    "filename": artifact.name,
                    "sha256": checksum,
                    "size_bytes": artifact.stat().st_size,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checksum_path, manifest_path


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    onefile = "--onefile" in sys.argv[1:]

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "PyInstaller is required. Install build dependencies with "
            'pip install -e ".[gui]" before building.',
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building FretHMM GUI bundle...")
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "frethmm.spec",
        "--noconfirm",
        "--clean",
    ]
    if onefile:
        os.environ["FRETHMM_ONEFILE"] = "1"
        onefile_target = project_root / "dist" / "FretHMM.exe"
        if onefile_target.exists():
            try:
                onefile_target.unlink()
            except PermissionError:
                print(
                    f"Cannot overwrite locked file: {onefile_target}",
                    file=sys.stderr,
                )
                print(
                    "Close any running FretHMM.exe instance and rerun with --onefile.",
                    file=sys.stderr,
                )
                sys.exit(1)

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"PyInstaller failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)

    if onefile:
        bundle_path = project_root / "dist" / "FretHMM.exe"
        print(f"\nBuild successful. Single-file GUI created at: {bundle_path}")
        release_artifact = bundle_path
    else:
        bundle_dir = project_root / "dist" / "FretHMM"
        print(f"\nBuild successful. GUI bundle created at: {bundle_dir}")
        release_artifact = Path(
            shutil.make_archive(
                str(project_root / "dist" / "FretHMM"),
                "zip",
                root_dir=project_root / "dist",
                base_dir="FretHMM",
            )
        )

    checksum_path, manifest_path = _write_release_metadata(project_root, release_artifact)
    print(f"Release artifact: {release_artifact}")
    print(f"SHA-256: {checksum_path}")
    print(f"Release manifest: {manifest_path}")


if __name__ == "__main__":
    main()
