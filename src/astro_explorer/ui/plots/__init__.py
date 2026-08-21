"""Matplotlib plot builders shared by the 2D app and future Qt UI."""

from .atmosphere import draw_orbit, draw_spectra
from .blackbody import draw_blackbody
from .hr import draw_hr_diagram, draw_temperature_radius_diagram, population_arrays

__all__ = [
    "draw_blackbody",
    "draw_hr_diagram",
    "draw_orbit",
    "draw_spectra",
    "draw_temperature_radius_diagram",
    "population_arrays",
]
