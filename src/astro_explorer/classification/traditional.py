"""The conventional size/temperature naming scheme.

This preserves the original program's ``classify_planet`` behaviour, with
two corrections: the boundaries are named constants with a cited basis, and
a planet whose radius is unknown is not quietly promoted to "gas giant" on
the strength of a mass alone without saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u

from ..provenance import Parameter

__all__ = ["SizeClass", "ThermalClass", "TraditionalClass", "classify"]


class SizeClass(str, Enum):
    """Radius-based class.  Boundaries in Earth radii."""

    TERRESTRIAL = "Terrestrial"
    SUPER_EARTH = "Super-Earth"
    SUB_NEPTUNE = "Sub-Neptune"
    NEPTUNE_LIKE = "Neptune-like"
    GAS_GIANT = "Gas giant"
    UNKNOWN = "Unknown"


#: Upper radius bounds in Earth radii.  The 1.5-2.0 gap is the observed
#: radius valley (Fulton et al. 2017); the others are conventional.
SIZE_BOUNDARIES = (
    (1.25, SizeClass.TERRESTRIAL),
    (2.0, SizeClass.SUPER_EARTH),
    (4.0, SizeClass.SUB_NEPTUNE),
    (8.0, SizeClass.NEPTUNE_LIKE),
)


class ThermalClass(str, Enum):
    """Equilibrium-temperature band."""

    COLD = "Cold"
    TEMPERATE = "Temperate"
    HOT = "Hot"
    ULTRA_HOT = "Ultra-hot"
    UNKNOWN = "Unknown"


#: Upper equilibrium-temperature bounds in kelvin.
THERMAL_BOUNDARIES = (
    (180.0, ThermalClass.COLD),
    (1000.0, ThermalClass.TEMPERATE),
    (2000.0, ThermalClass.HOT),
)


@dataclass(frozen=True)
class TraditionalClass:
    """A conventional class plus a record of what was actually known."""

    size: SizeClass = SizeClass.UNKNOWN
    thermal: ThermalClass = ThermalClass.UNKNOWN
    basis: str = ""

    @property
    def label(self) -> str:
        if self.size is SizeClass.UNKNOWN:
            return "Unknown"
        if self.thermal in (ThermalClass.UNKNOWN, ThermalClass.TEMPERATE):
            return self.size.value
        return "{0} {1}".format(self.thermal.value, self.size.value.lower())

    def describe(self) -> str:
        return "{0} ({1})".format(self.label, self.basis or "no basis recorded")


def _size_from_radius(radius_earth: float) -> SizeClass:
    for bound, size in SIZE_BOUNDARIES:
        if radius_earth < bound:
            return size
    return SizeClass.GAS_GIANT


def _thermal_from_teq(teq: float) -> ThermalClass:
    for bound, thermal in THERMAL_BOUNDARIES:
        if teq < bound:
            return thermal
    return ThermalClass.ULTRA_HOT


def classify(
    radius_earth: Parameter | float | None,
    equilibrium_temperature: Parameter | float | None = None,
    mass_jupiter: Parameter | float | None = None,
) -> TraditionalClass:
    """Classify by radius, falling back to mass only with an explicit basis."""

    def numeric(value, unit):
        if isinstance(value, Parameter):
            return value.value_in(unit)
        return value

    radius = numeric(radius_earth, u.R_earth)
    teq = numeric(equilibrium_temperature, u.K)
    mass_j = numeric(mass_jupiter, u.M_jup)

    if radius is not None:
        size = _size_from_radius(radius)
        basis = "radius {0:.3g} R_earth".format(radius)
    elif mass_j is not None and mass_j >= 0.3:
        size = SizeClass.GAS_GIANT
        basis = "mass {0:.3g} M_jup; radius unknown".format(mass_j)
    else:
        return TraditionalClass(basis="neither radius nor a giant-planet mass available")

    if teq is None:
        return TraditionalClass(size=size, basis=basis + "; equilibrium temperature unknown")
    return TraditionalClass(
        size=size,
        thermal=_thermal_from_teq(teq),
        basis="{0}; T_eq {1:.0f} K".format(basis, teq),
    )
