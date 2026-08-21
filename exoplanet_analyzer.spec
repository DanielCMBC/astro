# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the corrected 2D application (roadmap section 3.11).

The bundled data files are generated from ``DECLARED_RESOURCES`` rather than
listed by hand, so declaring a new resource in
``astro_explorer.assets.manager`` is enough to get it packaged.  The old spec
shipped ``tables`` and ``atmospheric_signatures.json`` but the application
then looked for them relative to the process working directory, so a frozen
build launched from elsewhere could not find its own data.
"""

import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
sys.path.insert(0, str(SPEC_DIR / "src"))

from astro_explorer.assets.manager import DECLARED_RESOURCES  # noqa: E402


def _collect_datas():
    """(source, destination) pairs for every declared resource that exists."""
    datas = []
    for resource in DECLARED_RESOURCES:
        source = SPEC_DIR / resource.relative_path
        if not source.exists():
            if resource.required:
                raise SystemExit(
                    "required resource missing: {0}".format(resource.relative_path)
                )
            continue
        destination = resource.relative_path if resource.is_directory else "."
        # Shaders keep their package-relative location so ShaderLibrary
        # finds them inside the bundle.
        if resource.key == "shaders":
            destination = "astro_explorer/rendering/shaders"
        datas.append((str(source), destination))
    return datas


a = Analysis(
    ['exoplanet_analyzer.py'],
    pathex=[str(SPEC_DIR / "src")],
    binaries=[],
    datas=_collect_datas(),
    hiddenimports=[
        'astropy.table',
        'astropy.io.ascii',
        'astropy.coordinates',
        'astro_explorer.ui.main_window',
        'matplotlib.backends.backend_tkagg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['moderngl', 'pygame', 'PySide6', 'rebound'],
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
    name='exoplanet_analyzer',
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
)
