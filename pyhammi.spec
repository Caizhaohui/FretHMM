# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Collect all submodules from the pyhammi package
pyhammi_hidden = collect_submodules('pyhammi')

a = Analysis(
    ['pyhammi\\gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'hmmlearn',
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'numpy',
        'scipy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
    ] + pyhammi_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='pyHaMMy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pyHaMMy',
)
