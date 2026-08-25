"""Stellar physics and the two distinct diagram definitions.

Roadmap sections 3.6 and 14.  The original program plotted effective
temperature against stellar *radius* and called it an HR diagram.  That plot
is useful, but it is a stellar temperature-radius diagram.  Both are kept
here, each with its correct name and axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, derived, unknown
from .constants import L_SUN, SIGMA_SB, SOLAR_EFFECTIVE_TEMPERATURE

__all__ = [
    "DiagramKind",
    "luminosity_ratio_from_radius_and_teff",
    "luminosity_from_radius_and_teff",
    "absolute_magnitude_from_luminosity",
    "equilibrium_temperature",
    "insolation_earth_units",
    "habitable_zone_au",
    "HabitableZone",
]


class DiagramKind(str, Enum):
    """The two stellar diagrams, correctly distinguished (roadmap 3.6)."""

    HR_DIAGRAM = "HR_DIAGRAM"
    """Classical: luminosity (or absolute magnitude) against Teff."""

    TEMPERATURE_RADIUS = "TEMPERATURE_RADIUS"
    """The original program's plot: stellar radius against Teff."""

    @property
    def title(self) -> str:
        return {
            DiagramKind.HR_DIAGRAM: "Hertzsprung-Russell diagram",
            DiagramKind.TEMPERATURE_RADIUS: "Stellar temperature-radius diagram",
        }[self]

    @property
    def y_label(self) -> str:
        return {
            DiagramKind.HR_DIAGRAM: "Luminosity (L_sun)",
            DiagramKind.TEMPERATURE_RADIUS: "Radius (R_sun)",
        }[self]

    @property
    def x_label(self) -> str:
        return "Effective temperature (K)"


def luminosity_ratio_from_radius_and_teff(radius_solar, teff_k):
    """``L / L_sun`` from ``L = 4 pi R^2 sigma T^4``, on scalars or arrays.

    The bare arithmetic, with no provenance and no units bookkeeping, so
    that the two callers who need it use the *same* formula:

    * :func:`luminosity_from_radius_and_teff`, the provenance-aware wrapper
      the records and panels go through;
    * the HR diagram's background population, which has thousands of
      catalogue rows and cannot afford one :class:`Parameter` each.

    Before Explorer C4a the second of those carried its own inline
    ``R^2 (T/T_sun)^4``, which is the same identity written twice - and the
    plotting layer recomputing stellar physics is exactly what the golden
    rule forbids. The two agree now because there is one of them.

    Non-finite or non-positive inputs come back as NaN rather than as a
    plausible number, so a caller that forgets to filter gets NaN.
    """
    radius = np.asarray(radius_solar, dtype=np.float64)
    temperature = np.asarray(teff_k, dtype=np.float64)

    usable = (
        np.isfinite(radius)
        & np.isfinite(temperature)
        & (radius > 0.0)
        & (temperature > 0.0)
    )

    # Expressed against the solar reference rather than in SI, because the
    # answer wanted is a ratio and the constants then cancel exactly.
    solar_teff = float(SOLAR_EFFECTIVE_TEMPERATURE.to_value(u.K))
    ratio = np.where(
        usable,
        np.square(np.where(usable, radius, 1.0))
        * np.power(np.where(usable, temperature, solar_teff) / solar_teff, 4.0),
        np.nan,
    )
    return float(ratio) if ratio.ndim == 0 else ratio


def luminosity_from_radius_and_teff(
    radius_solar: Parameter | float | None,
    teff: Parameter | float | None,
) -> Parameter:
    """Derive ``L / L_sun`` from ``L = 4 pi R^2 sigma T^4`` (section 14.3).

    Always returns a DERIVED parameter: the value was computed here, not
    published.  Uncertainty is propagated as ``dL/L = 2 dR/R + 4 dT/T``.
    """
    radius_param = radius_solar if isinstance(radius_solar, Parameter) else None
    teff_param = teff if isinstance(teff, Parameter) else None

    radius = radius_param.value_in(u.R_sun) if radius_param else radius_solar
    temperature = teff_param.value_in(u.K) if teff_param else teff

    if radius is None or temperature is None:
        return unknown(u.L_sun, provenance="4 pi R^2 sigma T^4")
    if not np.isfinite(radius) or not np.isfinite(temperature) or radius <= 0 or temperature <= 0:
        return unknown(u.L_sun, provenance="4 pi R^2 sigma T^4")

    value = float(luminosity_ratio_from_radius_and_teff(radius, temperature))

    rel = 0.0
    if radius_param is not None and radius_param.error_plus is not None and radius:
        rel += 2.0 * radius_param.error_plus / abs(radius)
    if teff_param is not None and teff_param.error_plus is not None and temperature:
        rel += 4.0 * teff_param.error_plus / abs(temperature)
    error = value * rel if rel > 0.0 else None

    return derived(
        value,
        u.L_sun,
        error_plus=error,
        error_minus=error,
        provenance="stefan-boltzmann(st_rad, st_teff)",
        note="bolometric luminosity derived from radius and effective temperature",
    )


#: IAU 2015 nominal bolometric magnitude zero point.
M_BOL_SUN = 4.74


def absolute_magnitude_from_luminosity(luminosity_solar: Parameter | float | None) -> Parameter:
    """Absolute bolometric magnitude ``M = M_sun - 2.5 log10(L/L_sun)``."""
    value = (
        luminosity_solar.value_in(u.L_sun)
        if isinstance(luminosity_solar, Parameter)
        else luminosity_solar
    )
    if value is None or not np.isfinite(value) or value <= 0.0:
        return unknown(u.mag, provenance="M_bol")
    return derived(
        M_BOL_SUN - 2.5 * np.log10(value),
        u.mag,
        provenance="M_bol(luminosity)",
        note="absolute bolometric magnitude",
    )


def equilibrium_temperature(
    luminosity_solar: Parameter | float | None,
    semimajor_axis_au: Parameter | float | None,
    *,
    albedo: float = 0.3,
    redistribution: float = 1.0,
) -> Parameter:
    """Planetary equilibrium temperature.

    .. math::
        T_{eq} = T_{eff,\\odot} \\left(\\frac{L}{L_\\odot}\\right)^{1/4}
                 \\sqrt{\\frac{R_\\odot^{\\rm scale}}{2a}}
                 (1 - A)^{1/4}

    implemented directly from the energy balance
    ``T_eq = ((1-A) L / (16 pi f sigma a^2))^{1/4}``.

    ``albedo`` and ``redistribution`` are modelling choices, so the result is
    DERIVED and its note records the assumptions used.
    """
    lum = (
        luminosity_solar.value_in(u.L_sun)
        if isinstance(luminosity_solar, Parameter)
        else luminosity_solar
    )
    axis = (
        semimajor_axis_au.value_in(u.au)
        if isinstance(semimajor_axis_au, Parameter)
        else semimajor_axis_au
    )
    if lum is None or axis is None or not np.isfinite(lum) or not np.isfinite(axis):
        return unknown(u.K, provenance="T_eq")
    if lum <= 0 or axis <= 0:
        return unknown(u.K, provenance="T_eq")

    flux = (lum * L_SUN) / (16.0 * np.pi * redistribution * SIGMA_SB * (axis * u.au) ** 2)
    temperature = float((((1.0 - albedo) * flux) ** 0.25).to_value(u.K))
    return derived(
        temperature,
        u.K,
        provenance="equilibrium_temperature(luminosity, semimajor_axis)",
        note="assumes Bond albedo {0:g} and heat redistribution factor {1:g}".format(
            albedo, redistribution
        ),
    )


def insolation_earth_units(
    luminosity_solar: Parameter | float | None,
    semimajor_axis_au: Parameter | float | None,
) -> Parameter:
    """Stellar flux at the planet in Earth units, ``(L/L_sun)/(a/AU)^2``."""
    lum = (
        luminosity_solar.value_in(u.L_sun)
        if isinstance(luminosity_solar, Parameter)
        else luminosity_solar
    )
    axis = (
        semimajor_axis_au.value_in(u.au)
        if isinstance(semimajor_axis_au, Parameter)
        else semimajor_axis_au
    )
    if lum is None or axis is None or not np.isfinite(lum) or not np.isfinite(axis) or axis <= 0:
        return unknown(provenance="insolation")
    return derived(
        lum / axis**2,
        u.dimensionless_unscaled,
        provenance="insolation(luminosity, semimajor_axis)",
        note="stellar flux relative to Earth",
    )


@dataclass(frozen=True)
class HabitableZone:
    """Conservative habitable-zone boundaries in AU."""

    inner: Parameter
    outer: Parameter
    model: str = "Kopparapu et al. 2013 (runaway greenhouse / maximum greenhouse)"

    @property
    def is_known(self) -> bool:
        return self.inner.is_known and self.outer.is_known


# Kopparapu et al. (2013) coefficients for the conservative HZ, valid for
# 2600 K <= Teff <= 7200 K.  Order: S_eff_sun, a, b, c, d.
_HZ_RUNAWAY = (1.0512, 1.3242e-4, 1.5418e-8, -7.9895e-12, -1.8328e-15)
_HZ_MAX_GREENHOUSE = (0.3438, 5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16)


def _seff(coefficients, teff: float) -> float:
    s0, a, b, c, d = coefficients
    t = teff - 5780.0
    return s0 + a * t + b * t**2 + c * t**3 + d * t**4


def habitable_zone_au(
    luminosity_solar: Parameter | float | None,
    teff: Parameter | float | None,
) -> HabitableZone:
    """Conservative habitable zone (roadmap section 20, SYSTEM view).

    Returns UNKNOWN bounds outside the coefficients' validity range rather
    than extrapolating a polynomial fit into a regime it was never fitted
    for.
    """
    lum = (
        luminosity_solar.value_in(u.L_sun)
        if isinstance(luminosity_solar, Parameter)
        else luminosity_solar
    )
    temperature = teff.value_in(u.K) if isinstance(teff, Parameter) else teff

    if lum is None or temperature is None or not np.isfinite(lum) or not np.isfinite(temperature):
        return HabitableZone(unknown(u.au, provenance="hz"), unknown(u.au, provenance="hz"))
    if lum <= 0 or not (2600.0 <= temperature <= 7200.0):
        return HabitableZone(
            unknown(u.au, provenance="hz", note="outside the fitted Teff range"),
            unknown(u.au, provenance="hz", note="outside the fitted Teff range"),
        )

    inner = np.sqrt(lum / _seff(_HZ_RUNAWAY, temperature))
    outer = np.sqrt(lum / _seff(_HZ_MAX_GREENHOUSE, temperature))
    return HabitableZone(
        derived(inner, u.au, provenance="kopparapu2013:runaway_greenhouse"),
        derived(outer, u.au, provenance="kopparapu2013:maximum_greenhouse"),
    )


def solar_teff_kelvin() -> float:
    """Nominal solar effective temperature, for scaling relations."""
    return float(SOLAR_EFFECTIVE_TEMPERATURE.to_value(u.K))
