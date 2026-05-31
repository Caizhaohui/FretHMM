import subprocess
import sys
import os


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Installing PyInstaller...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pyinstaller"],
        check=True,
    )

    print("Building pyHaMMy.exe...")
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--name",
        "pyHaMMy",
        "--add-data",
        "pyhammi;pyhammi",
        "--hidden-import",
        "hmmlearn",
        "--hidden-import",
        "sklearn",
        "--hidden-import",
        "sklearn.utils._cython_blas",
        "--hidden-import",
        "pyhammi.i18n",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter.test",
        "--exclude-module",
        "unittest",
        "--exclude-module",
        "pydoc",
        "--noupx",
        "pyhammi/gui.py",
    ]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"PyInstaller failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)

    dist_dir = os.path.abspath("dist")
    exe_path = os.path.join(dist_dir, "pyHaMMy.exe")
    print(f"\nBuild successful. pyHaMMy.exe created at: {exe_path}")


if __name__ == "__main__":
    main()
