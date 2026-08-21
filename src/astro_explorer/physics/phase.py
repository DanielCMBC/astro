"""Where a planet is, and how much of that is observed.

Review sections 7, 8 and 9. The previous milestone established *that* a
phase can be assumed; this module names *how* it was anchored, because the
interesting case is neither fully observed nor fully invented.

Kepler-11 is the worked example. Every planet has a published mid-transit
time - a directly observed instant - but no published argument of
periastron. So:

* the **temporal anchor is observed**: the planet really was in front of its
  star at that moment;
* the **orbital orientation is not**: without ``omega`` the orbit's rotation
  within its own plane is unknown, and the display normalises it to zero.

Calling that "positioned from a published epoch" overstates it; calling it
"assumed" understates it. It is
:attr:`PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED`.

Geometry note
-------------
Inferior conjunction is *defined* by the argument of latitude

.. math:: u = \\omega + \\nu = \\pi/2

so ``nu_transit = pi/2 - omega`` is exact for conjunction, not an
approximation. What is approximate is equating conjunction with the instant
of minimum sky-projected separation: for an eccentric, non-edge-on orbit the
two differ by a term of order ``e cos(omega) cos^2(i)``, which vanishes as
``i -> 90 deg``. Kepler-11's planets sit within a degree of edge-on, so the
offset is negligible - but it is a real effect and is named rather than
buried.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Status

__all__ = [
    "PhaseAnchor",
    "AnomalyMapping",
    "PhaseStatus",
    "PhaseProvenance",
    "PhaseSolution",
    "conjunction_offset_scale",
]


class PhaseAnchor(str, Enum):
    """Whether the *instant* the phase is tied to was observed."""

    OBSERVED = "OBSERVED"
    """A published epoch: a periastron passage or a transit."""

    ASSUMED = "ASSUMED"
    """No epoch; the zero point is arbitrary."""

    NONE = "NONE"
    """Not even a period, so there is no phase to anchor."""


class AnomalyMapping(str, Enum):
    """How the anchoring epoch was turned into a mean anomaly."""

    DIRECT = "DIRECT"
    """A periastron epoch gives ``M = 0`` outright; nothing is assumed."""

    CONJUNCTION_NORMALIZED = "CONJUNCTION_NORMALIZED"
    """A transit epoch mapped through ``nu = pi/2 - omega``.

    Exact for conjunction. When ``omega`` itself was normalised to zero the
    *time* stays observed while the in-plane orientation does not.
    """

    ARBITRARY_ZERO = "ARBITRARY_ZERO"
    """No epoch: the planet is advanced at the right rate from nothing."""

    NONE = "NONE"


class PhaseStatus(str, Enum):
    """How much the current position may be claimed to mean."""

    CONSTRAINED = "CONSTRAINED"
    """Epoch and orientation are both published: this is where it is."""

    PARTIALLY_CONSTRAINED = "PARTIALLY_CONSTRAINED"
    """The instant is observed; some orientation element is normalised."""

    ASSUMED = "ASSUMED"
    """The motion is physical, the position is illustrative."""

    UNKNOWN = "UNKNOWN"
    """No position can be drawn at all."""

    @property
    def is_observationally_anchored(self) -> bool:
        """True when a real observation fixes the timing."""
        return self in (PhaseStatus.CONSTRAINED, PhaseStatus.PARTIALLY_CONSTRAINED)

    @property
    def label(self) -> str:
        return {
            PhaseStatus.CONSTRAINED: "constrained by a published epoch and orientation",
            PhaseStatus.PARTIALLY_CONSTRAINED: (
                "timing observed, orbital orientation normalised for display"
            ),
            PhaseStatus.ASSUMED: "assumed: correct rate, arbitrary starting point",
            PhaseStatus.UNKNOWN: "no phase available",
        }[self]


class PhaseProvenance(str, Enum):
    """The vocabulary review section 9 asks for."""

    PERIASTRON_EPOCH = "PERIASTRON_EPOCH"
    TRANSIT_EPOCH = "TRANSIT_EPOCH"
    """A transit epoch with a published ``omega``."""

    TRANSIT_CONJUNCTION_NORMALIZED = "TRANSIT_CONJUNCTION_NORMALIZED"
    """A transit epoch with ``omega`` normalised to zero for display."""

    MEAN_ANOMALY_AT_EPOCH = "MEAN_ANOMALY_AT_EPOCH"
    ASSUMED_ZERO_PHASE = "ASSUMED_ZERO_PHASE"
    UNKNOWN = "UNKNOWN"

    @property
    def anchor(self) -> PhaseAnchor:
        if self is PhaseProvenance.UNKNOWN:
            return PhaseAnchor.NONE
        if self is PhaseProvenance.ASSUMED_ZERO_PHASE:
            return PhaseAnchor.ASSUMED
        return PhaseAnchor.OBSERVED

    @property
    def mapping(self) -> AnomalyMapping:
        return {
            PhaseProvenance.PERIASTRON_EPOCH: AnomalyMapping.DIRECT,
            PhaseProvenance.MEAN_ANOMALY_AT_EPOCH: AnomalyMapping.DIRECT,
            PhaseProvenance.TRANSIT_EPOCH: AnomalyMapping.CONJUNCTION_NORMALIZED,
            PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED: (
                AnomalyMapping.CONJUNCTION_NORMALIZED
            ),
            PhaseProvenance.ASSUMED_ZERO_PHASE: AnomalyMapping.ARBITRARY_ZERO,
            PhaseProvenance.UNKNOWN: AnomalyMapping.NONE,
        }[self]

    @property
    def label(self) -> str:
        return {
            PhaseProvenance.PERIASTRON_EPOCH: "published time of periastron",
            PhaseProvenance.TRANSIT_EPOCH: "published transit time and argument of periastron",
            PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED: (
                "published transit time, argument of periastron normalised to 0 deg"
            ),
            PhaseProvenance.MEAN_ANOMALY_AT_EPOCH: "published mean anomaly at epoch",
            PhaseProvenance.ASSUMED_ZERO_PHASE: "no epoch published; phase assumed",
            PhaseProvenance.UNKNOWN: "no phase available",
        }[self]


def conjunction_offset_scale(eccentricity: float, inclination_rad: float) -> float:
    """Scale of the conjunction / minimum-separation discrepancy, in radians.

    Order ``e |cos omega| cos^2 i``; the worst case over ``omega`` is taken,
    so this is an upper bound rather than the actual offset. It vanishes for
    a circular orbit and for an exactly edge-on one, which is why it is
    negligible for transiting planets and worth stating anyway.
    """
    if not np.isfinite(eccentricity) or not np.isfinite(inclination_rad):
        return float("nan")
    return float(abs(eccentricity) * np.cos(inclination_rad) ** 2)


@dataclass(frozen=True)
class PhaseSolution:
    """A mean anomaly together with everything that qualifies it."""

    mean_anomaly: float | None
    provenance: PhaseProvenance = PhaseProvenance.UNKNOWN
    #: Status of the argument of periastron actually used in the mapping.
    omega_status: Status = Status.UNKNOWN
    #: Upper bound on the conjunction offset, radians of true anomaly.
    conjunction_offset: float = 0.0
    note: str = ""

    @property
    def anchor(self) -> PhaseAnchor:
        return self.provenance.anchor

    @property
    def mapping(self) -> AnomalyMapping:
        return self.provenance.mapping

    @property
    def status(self) -> PhaseStatus:
        """The single summary field the UI shows."""
        if self.mean_anomaly is None or self.provenance is PhaseProvenance.UNKNOWN:
            return PhaseStatus.UNKNOWN
        if self.provenance is PhaseProvenance.ASSUMED_ZERO_PHASE:
            return PhaseStatus.ASSUMED
        # The instant is observed. Whether the position is fully constrained
        # depends on whether the orientation used to interpret it was too.
        if self.omega_status in (Status.MEASURED, Status.DERIVED):
            return PhaseStatus.CONSTRAINED
        if self.mapping is AnomalyMapping.DIRECT:
            # A periastron epoch needs no omega at all, so a missing one
            # does not weaken it.
            return PhaseStatus.CONSTRAINED
        return PhaseStatus.PARTIALLY_CONSTRAINED

    @property
    def is_assumed(self) -> bool:
        """Backwards-compatible flag: True unless fully constrained."""
        return self.status is not PhaseStatus.CONSTRAINED

    @property
    def is_placeable(self) -> bool:
        return self.mean_anomaly is not None

    def describe(self) -> list[str]:
        lines = [
            "Phase provenance:  {0}".format(self.provenance.value),
            "  source:          {0}".format(self.provenance.label),
            "  anchor:          {0}".format(self.anchor.value),
            "  anomaly mapping: {0}".format(self.mapping.value),
            "  phase status:    {0}".format(self.status.value),
            "                   {0}".format(self.status.label),
        ]
        if self.mapping is AnomalyMapping.CONJUNCTION_NORMALIZED:
            lines.append(
                "  omega used:      {0}".format(
                    "published" if self.omega_status is Status.MEASURED else "normalised to 0 deg"
                )
            )
            if self.conjunction_offset > 0.0:
                lines.append(
                    "  conjunction vs minimum separation differs by up to "
                    "{0:.2e} rad of true anomaly".format(self.conjunction_offset)
                )
        if self.note:
            lines.append("  note:            {0}".format(self.note))
        return lines

    def as_dict(self) -> dict:
        return {
            "mean_anomaly": self.mean_anomaly,
            "provenance": self.provenance.value,
            "anchor": self.anchor.value,
            "anomaly_mapping": self.mapping.value,
            "phase_status": self.status.value,
            "omega_status": self.omega_status.value,
            "conjunction_offset": self.conjunction_offset,
        }
