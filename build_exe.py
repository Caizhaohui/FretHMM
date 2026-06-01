import subprocess
import sys
import os
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    onefile = "--onefile" in sys.argv[1:]

    print("Installing PyInstaller...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pyinstaller"],
        check=True,
    )

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
    else:
        bundle_dir = project_root / "dist" / "FretHMM"
        print(f"\nBuild successful. GUI bundle created at: {bundle_dir}")


if __name__ == "__main__":
    main()
