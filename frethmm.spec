import customtkinter
import os
import sys
from PyInstaller.utils.hooks import collect_submodules

sys.path.insert(0, os.path.abspath("."))

frethmm_hidden = collect_submodules("frethmm")

a = Analysis(
    ["frethmm\\app\\gui.py"],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[
        (os.path.join(os.path.dirname(customtkinter.__file__), "assets"), "customtkinter/assets"),
        ("frethmm", "frethmm")
    ],
    hiddenimports=[
        "hmmlearn",
        "sklearn",
        "sklearn.utils._cython_blas",
        "customtkinter",
        "frethmm",
        "frethmm.app",
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
    ] + frethmm_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "unittest", "pydoc"],
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
