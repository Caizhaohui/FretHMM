import subprocess
import sys
import os
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

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

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"PyInstaller failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)

    bundle_dir = project_root / "dist" / "FretHMM"
    print(f"\nBuild successful. GUI bundle created at: {bundle_dir}")


if __name__ == "__main__":
    main()
