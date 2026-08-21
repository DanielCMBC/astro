"""Planet classification: the conventional scheme and the draft vector."""

from .physical_vector import (
    SCHEME_VERSION,
    AtmosphericEvidence,
    Confidence,
    Irradiation,
    OrbitalRegime,
    PhysicalVector,
    Structure,
    Thermal,
    classify_vector,
)
from .traditional import SizeClass, ThermalClass, TraditionalClass, classify

__all__ = [
    "SCHEME_VERSION",
    "AtmosphericEvidence",
    "Confidence",
    "Irradiation",
    "OrbitalRegime",
    "PhysicalVector",
    "SizeClass",
    "Structure",
    "Thermal",
    "ThermalClass",
    "TraditionalClass",
    "classify",
    "classify_vector",
]
