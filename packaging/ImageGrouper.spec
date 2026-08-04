# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build definition - see build.bat, or run:

    pyinstaller --noconfirm packaging/ImageGrouper.spec
"""

import os

# SPECPATH is the directory holding this file (<repo>/packaging).
ROOT = os.path.dirname(os.path.abspath(SPECPATH))
ASSETS = os.path.join(ROOT, "assets")

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=[(ASSETS, "assets")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trimming these keeps the single-file exe from ballooning; none of them
    # are imported by the app or by the parts of Pillow/numpy it uses.
    excludes=[
        "matplotlib", "scipy", "pandas", "pytest", "setuptools", "pydoc",
        "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
        "numpy.f2py", "numpy.distutils", "PIL.ImageQt",
    ],
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
    name="ImageGrouper",
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
    icon=os.path.join(ASSETS, "app.ico"),
)
