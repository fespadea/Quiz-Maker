# -*- mode: python ; coding: utf-8 -*-
import sys # added line
from os import path # added line
site_packages = next(p for p in sys.path if 'site-packages' in p) # added line


a = Analysis(
    ['quizMaker.py'],
    pathex=[],
    binaries=[],
    datas=[(path.join(site_packages,"pptx","templates"), "pptx/templates")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='quizMaker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
