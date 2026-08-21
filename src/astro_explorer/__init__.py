"""Astro Explorer - offline-first exoplanet scientific explorer.

The package is layered so that the renderer never owns scientific truth
(roadmap section 6).  The dependency direction is strictly::

    data -> physics/coordinates/spectroscopy -> app state -> ui / rendering

Nothing in :mod:`astro_explorer.rendering` may import from
:mod:`astro_explorer.data`; the renderer receives already-computed positions
and display parameters.
"""

__version__ = "0.2.1"

__all__ = ["__version__"]
