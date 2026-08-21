"""Multidimensional physical classification (roadmap section 18).

The roadmap is explicit that this is a *design concept*: the boundaries here
are provisional and must not be treated as settled science until a
literature review and a statistical analysis justify them.  Every code this
module emits therefore carries a version string and a confidence dimension,
so a future revision can be told apart from this one, and a planet with
almost no data cannot masquerade as a well-characterised one.

The classification is a vector, not a scalar.  Reducing planetary physics to
one number is exactly what the roadmap warns against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u

from ..provenance import Parameter, Status

__all__ = [
    "SCHEME_VERSION",
    "Structure",
    "Thermal",
    "OrbitalRegime",
    "AtmosphericEvidence",
    "Irradiation",
    "Confidence",
    "PhysicalVector",
    "classify_vector",
]

#: Bump this whenever a boundary moves, so stored codes stay interpretable.
SCHEME_VERSION = "0.1-draft"

_PROVISIONAL = (
    "Provisional draft scheme (roadmap section 18): boundaries are not yet "
    "justified by a literature review or statistical analysis."
)


class Structure(str, Enum):
    """Bulk structural regime."""

    ROCKY = "R"
    SUPER_EARTH = "S"
    NEPTUNIAN = "N"
    GIANT = "G"
    UNKNOWN = "?"


class Thermal(str, Enum):
    """Thermal regime."""

    COLD = "C"
    TEMPERATE = "T"
    HOT = "H"
    ULTRA_HOT = "U"
    UNKNOWN = "?"


class OrbitalRegime(str, Enum):
    """Orbital regime."""

    ULTRA_SHORT = "U"
    """Period below one day."""

    CLOSE = "C"
    """Period below ten days."""

    INTERMEDIATE = "I"
    WIDE = "W"
    ECCENTRIC = "E"
    """Eccentricity above 0.3, whatever the period."""

    UNKNOWN = "?"


class AtmosphericEvidence(str, Enum):
    """What is known about the atmosphere."""

    UNKNOWN = "?"
    DETECTED = "D"
    """At least one molecule robustly detected."""

    CONSTRAINED = "C"
    """Composition constrained by a retrieval."""

    ABSENT = "N"
    """Searched for and consistent with no thick atmosphere."""


class Irradiation(str, Enum):
    """Incident flux relative to Earth."""

    LOW = "L"
    """Below 0.3 Earth flux."""

    MODERATE = "M"
    """0.3 to 10 Earth flux."""

    HIGH = "H"
    """10 to 1000 Earth flux."""

    EXTREME = "X"
    """Above 1000 Earth flux."""

    UNKNOWN = "?"


class Confidence(str, Enum):
    """How much of the vector rests on measurements."""

    HIGH = "3"
    """Structure, thermal and orbital regime all from measured values."""

    MEDIUM = "2"
    LOW = "1"
    MINIMAL = "0"
    """Almost nothing is measured; the code is close to meaningless."""


@dataclass(frozen=True)
class PhysicalVector:
    """A planet's position in the draft classification space."""

    structure: Structure = Structure.UNKNOWN
    thermal: Thermal = Thermal.UNKNOWN
    orbital: OrbitalRegime = OrbitalRegime.UNKNOWN
    atmosphere: AtmosphericEvidence = AtmosphericEvidence.UNKNOWN
    irradiation: Irradiation = Irradiation.UNKNOWN
    confidence: Confidence = Confidence.MINIMAL
    scheme_version: str = SCHEME_VERSION

    @property
    def code(self) -> str:
        """Compact code, e.g. ``G-H-D``, plus regime and confidence."""
        return "-".join(
            (
                self.structure.value,
                self.thermal.value,
                self.orbital.value,
                self.atmosphere.value,
                self.irradiation.value,
                self.confidence.value,
            )
        )

    @property
    def short_code(self) -> str:
        """The three-axis form the roadmap uses as an example (``G-H-D``)."""
        return "-".join((self.structure.value, self.thermal.value, self.atmosphere.value))

    @property
    def caveat(self) -> str:
        return _PROVISIONAL

    def describe(self) -> list[str]:
        return [
            "Classification:    {0} (scheme {1})".format(self.code, self.scheme_version),
            "  Structure:       {0}".format(self.structure.name.replace("_", " ").title()),
            "  Thermal:         {0}".format(self.thermal.name.replace("_", " ").title()),
            "  Orbital regime:  {0}".format(self.orbital.name.replace("_", " ").title()),
            "  Atmosphere:      {0}".format(self.atmosphere.name.replace("_", " ").title()),
            "  Irradiation:     {0}".format(self.irradiation.name.title()),
            "  Confidence:      {0}".format(self.confidence.name.title()),
            "  " + _PROVISIONAL,
        ]


def _value(parameter, unit):
    if isinstance(parameter, Parameter):
        return parameter.value_in(unit), parameter.status
    return parameter, Status.MEASURED if parameter is not None else Status.UNKNOWN


def classify_vector(
    radius_earth=None,
    mass_earth=None,
    equilibrium_temperature=None,
    period_days=None,
    eccentricity=None,
    insolation_earth=None,
    atmosphere: AtmosphericEvidence = AtmosphericEvidence.UNKNOWN,
) -> PhysicalVector:
    """Place a planet in the draft classification space.

    Each dimension degrades to UNKNOWN independently, and the confidence
    axis counts how many of the three primary dimensions rested on real
    values rather than on nothing.
    """
    radius, _ = _value(radius_earth, u.R_earth)
    _mass, _ = _value(mass_earth, u.M_earth)
    teq, _ = _value(equilibrium_temperature, u.K)
    period, _ = _value(period_days, u.day)
    ecc, _ = _value(eccentricity, u.dimensionless_unscaled)
    flux, _ = _value(insolation_earth, u.dimensionless_unscaled)

    # Structure: radius-driven, with the radius valley at 1.5-2 R_earth.
    if radius is None:
        structure = Structure.UNKNOWN
    elif radius < 1.5:
        structure = Structure.ROCKY
    elif radius < 2.0:
        structure = Structure.SUPER_EARTH
    elif radius < 6.0:
        structure = Structure.NEPTUNIAN
    else:
        structure = Structure.GIANT

    if teq is None:
        thermal = Thermal.UNKNOWN
    elif teq < 180.0:
        thermal = Thermal.COLD
    elif teq < 1000.0:
        thermal = Thermal.TEMPERATE
    elif teq < 2000.0:
        thermal = Thermal.HOT
    else:
        thermal = Thermal.ULTRA_HOT

    if ecc is not None and ecc > 0.3:
        orbital = OrbitalRegime.ECCENTRIC
    elif period is None:
        orbital = OrbitalRegime.UNKNOWN
    elif period < 1.0:
        orbital = OrbitalRegime.ULTRA_SHORT
    elif period < 10.0:
        orbital = OrbitalRegime.CLOSE
    elif period < 100.0:
        orbital = OrbitalRegime.INTERMEDIATE
    else:
        orbital = OrbitalRegime.WIDE

    if flux is None:
        irradiation = Irradiation.UNKNOWN
    elif flux < 0.3:
        irradiation = Irradiation.LOW
    elif flux < 10.0:
        irradiation = Irradiation.MODERATE
    elif flux < 1000.0:
        irradiation = Irradiation.HIGH
    else:
        irradiation = Irradiation.EXTREME

    known = sum(
        1
        for dimension in (structure, thermal, orbital)
        if dimension.value != "?"
    )
    confidence = {3: Confidence.HIGH, 2: Confidence.MEDIUM, 1: Confidence.LOW}.get(
        known, Confidence.MINIMAL
    )

    return PhysicalVector(
        structure=structure,
        thermal=thermal,
        orbital=orbital,
        atmosphere=atmosphere,
        irradiation=irradiation,
        confidence=confidence,
    )
