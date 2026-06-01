import customtkinter
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

sys.path.insert(0, os.path.abspath("."))

frethmm_hidden = collect_submodules("frethmm")
unittest_hidden = collect_submodules("unittest")


def _collect_package(name):
    datas, binaries, hiddenimports = collect_all(name)
    return datas, binaries, hiddenimports


customtkinter_datas, customtkinter_binaries, customtkinter_hidden = _collect_package("customtkinter")
hmmlearn_datas, hmmlearn_binaries, hmmlearn_hidden = _collect_package("hmmlearn")
sklearn_datas, sklearn_binaries, sklearn_hidden = _collect_package("sklearn")
scipy_datas, scipy_binaries, scipy_hidden = _collect_package("scipy")
numpy_datas, numpy_binaries, numpy_hidden = _collect_package("numpy")
matplotlib_datas, matplotlib_binaries, matplotlib_hidden = _collect_package("matplotlib")

metadata_datas = (
    copy_metadata("frethmm")
    + copy_metadata("customtkinter")
    + copy_metadata("hmmlearn")
    + copy_metadata("scipy")
    + copy_metadata("numpy")
    + copy_metadata("matplotlib")
    + copy_metadata("scikit-learn")
)

a = Analysis(
    ["frethmm\\app\\gui.py"],
    pathex=[os.path.abspath(".")],
    binaries=(
        customtkinter_binaries
        + hmmlearn_binaries
        + sklearn_binaries
        + scipy_binaries
        + numpy_binaries
        + matplotlib_binaries
    ),
    datas=[
        (os.path.join(os.path.dirname(customtkinter.__file__), "assets"), "customtkinter/assets"),
        ("frethmm", "frethmm"),
    ]
    + customtkinter_datas
    + hmmlearn_datas
    + sklearn_datas
    + scipy_datas
    + numpy_datas
    + matplotlib_datas
    + metadata_datas,
    hiddenimports=[
        "customtkinter",
        "frethmm",
        "frethmm.app",
        "frethmm.app.cli",
        "frethmm.app.gui",
        "frethmm.app.i18n",
        "frethmm.core",
        "frethmm.core.io",
        "frethmm.core.model",
        "frethmm.core.batch",
        "frethmm.core.postprocess",
        "frethmm.domain",
        "frethmm.domain.models",
        "frethmm.formats",
        "frethmm.formats.report_parser",
        "frethmm.viz",
        "frethmm.viz.tdp",
    ]
    + frethmm_hidden
    + unittest_hidden
    + customtkinter_hidden
    + hmmlearn_hidden
    + sklearn_hidden
    + scipy_hidden
    + numpy_hidden
    + matplotlib_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FretHMM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="FretHMM",
)
