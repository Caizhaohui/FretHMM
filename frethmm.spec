# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


frethmm_hidden = collect_submodules("frethmm")

a = Analysis(
    ["frethmm\\app\\gui.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "hmmlearn",
        "sklearn",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
    ] + frethmm_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter.test", "unittest", "pydoc"],
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
