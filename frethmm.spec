import customtkinter
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

sys.path.insert(0, os.path.abspath("."))
ONEFILE = os.environ.get("FRETHMM_ONEFILE") == "1"

customtkinter_datas = collect_data_files("customtkinter")
hmmlearn_binaries = collect_dynamic_libs("hmmlearn")
sklearn_binaries = collect_dynamic_libs("sklearn")
scipy_binaries = collect_dynamic_libs("scipy")
numpy_binaries = collect_dynamic_libs("numpy")
app_asset_datas = [
    ("frethmm/assets/frethmm_logo.png", "frethmm/assets"),
    ("frethmm/assets/frethmm.ico", "frethmm/assets"),
]

metadata_datas = (
    copy_metadata("frethmm")
    + copy_metadata("customtkinter")
    + copy_metadata("hmmlearn")
    + copy_metadata("scipy")
    + copy_metadata("numpy")
    + copy_metadata("scikit-learn")
)

a = Analysis(
    ["frethmm\\app\\gui.py"],
    pathex=[os.path.abspath(".")],
    binaries=(
        hmmlearn_binaries
        + sklearn_binaries
        + scipy_binaries
        + numpy_binaries
    ),
    datas=[
        (os.path.join(os.path.dirname(customtkinter.__file__), "assets"), "customtkinter/assets"),
        *app_asset_datas,
    ]
    + customtkinter_datas
    + metadata_datas,
    hiddenimports=[
        "customtkinter",
        "frethmm",
        "frethmm.app.gui",
        "frethmm.app.i18n",
        "frethmm.core.io",
        "frethmm.core.model",
        "frethmm.core.postprocess",
        "frethmm.domain.models",
        "hmmlearn",
        "hmmlearn.hmm",
        "numpy",
        "pydoc",
        "scipy",
        "scipy.linalg",
        "scipy.special",
        "sklearn",
        "sklearn.base",
        "sklearn.cluster",
        "sklearn.mixture",
        "sklearn.utils",
        "sklearn.utils._cython_blas",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._vector_sentinel",
        "unittest",
        "unittest.mock",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PyQt5",
        "matplotlib",
        "notebook",
        "pandas",
        "pytest",
        "scipy.tests",
        "sklearn.tests",
        "tkinter.test",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries if ONEFILE else [],
    a.datas if ONEFILE else [],
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
    icon="frethmm\\assets\\frethmm.ico",
    exclude_binaries=not ONEFILE,
)

if not ONEFILE:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="FretHMM",
    )
