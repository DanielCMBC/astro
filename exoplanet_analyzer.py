# -*- coding: utf-8 -*-
"""Launcher for the corrected 2D Exoplanet Scientific Suite.

The application used to live entirely in this file.  Following roadmap
Phase 2 its scientific content now lives in ``src/astro_explorer`` as
reusable modules, so the same models can feed the 3D engine instead of being
reimplemented there.

The original single-file program is preserved verbatim at
``legacy/exoplanet_analyzer_original.py`` as the historical baseline, and its
behaviour is documented in ``docs/original-2d-behaviour.md``.

Run with::

    python exoplanet_analyzer.py

or, once installed, with the ``astro-explorer-2d`` console script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running straight from a source checkout without installing.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    from astro_explorer.app.main import main as run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
