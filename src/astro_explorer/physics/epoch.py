"""Orbital epochs and the time system they are quoted in.

Review section 11, task list item "preserve time-system/epoch metadata".

An epoch is not just a number of days. ``pl_orbtper = 2458882.344`` is
meaningless without knowing which Julian-date flavour it is:

* **BJD_TDB** - barycentric, barycentric dynamical time. The standard for
  precise exoplanet ephemerides.
* **HJD_UTC** - heliocentric, coordinated universal time. Common in older
  literature.
* **BKJD** - Kepler's offset barycentric date, ``BJD - 2454833.0``.
* **BTJD** - TESS's offset barycentric date, ``BJD - 2457000.0``.

The differences matter at different magnitudes. Barycentric versus
heliocentric can reach about 8 minutes, because the Sun moves up to ~1.6
million km from the solar-system barycentre. TDB versus UTC is the
accumulated leap seconds, currently 69.184 s. Neither is visible on a
111-day orbit, but both are fatal to transit-timing work, and the mission
offsets are catastrophic if ignored: 2454833 days is thirteen years.

The archive quotes ``pl_orbtper`` and ``pl_tranmid`` as Julian days without
a machine-readable scale column, so the honest default is
:attr:`TimeScale.JD_UNSPECIFIED`: the offset is known to be zero, the
sub-minute scale is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status, unknown

__all__ = ["TimeScale", "EpochKind", "Epoch", "TDB_MINUS_UTC_SECONDS"]

#: Leap seconds accumulated between TAI and UTC, as TDB - UTC (2017 onwards).
#: Recorded so the size of the ambiguity can be stated rather than waved at.
TDB_MINUS_UTC_SECONDS = 69.184

#: Largest possible barycentric-minus-heliocentric light-time difference.
BJD_MINUS_HJD_MAX_SECONDS = 480.0


class TimeScale(str, Enum):
    """Which Julian-date convention an epoch is quoted in."""

    BJD_TDB = "BJD_TDB"
    HJD_UTC = "HJD_UTC"
    JD_UTC = "JD_UTC"
    BKJD = "BKJD"
    """Kepler: ``BJD_TDB - 2454833.0``."""

    BTJD = "BTJD"
    """TESS: ``BJD_TDB - 2457000.0``."""

    JD_UNSPECIFIED = "JD_UNSPECIFIED"
    """A Julian date with no stated scale - the NASA archive's usual state."""

    UNKNOWN = "UNKNOWN"

    @property
    def offset_to_bjd(self) -> float:
        """Days to add to reach a full Julian date."""
        return {
            TimeScale.BKJD: 2454833.0,
            TimeScale.BTJD: 2457000.0,
        }.get(self, 0.0)

    @property
    def is_barycentric(self) -> bool:
        return self in (TimeScale.BJD_TDB, TimeScale.BKJD, TimeScale.BTJD)

    @property
    def is_determinate(self) -> bool:
        """True when the scale is actually stated."""
        return self not in (TimeScale.JD_UNSPECIFIED, TimeScale.UNKNOWN)

    @property
    def label(self) -> str:
        return {
            TimeScale.BJD_TDB: "barycentric Julian date, TDB",
            TimeScale.HJD_UTC: "heliocentric Julian date, UTC",
            TimeScale.JD_UTC: "Julian date, UTC",
            TimeScale.BKJD: "Kepler barycentric date (BJD - 2454833)",
            TimeScale.BTJD: "TESS barycentric date (BJD - 2457000)",
            TimeScale.JD_UNSPECIFIED: "Julian date, scale not stated",
            TimeScale.UNKNOWN: "unknown time system",
        }[self]

    @property
    def uncertainty_seconds(self) -> float:
        """How wrong converting to BJD_TDB could be, in seconds.

        Zero when the scale is stated; otherwise the worst case of the
        conventions it might really be.
        """
        if self is TimeScale.BJD_TDB:
            return 0.0
        if self is TimeScale.HJD_UTC:
            return BJD_MINUS_HJD_MAX_SECONDS + TDB_MINUS_UTC_SECONDS
        if self is TimeScale.JD_UTC:
            return BJD_MINUS_HJD_MAX_SECONDS + TDB_MINUS_UTC_SECONDS
        if self in (TimeScale.BKJD, TimeScale.BTJD):
            return 0.0
        if self is TimeScale.JD_UNSPECIFIED:
            return BJD_MINUS_HJD_MAX_SECONDS + TDB_MINUS_UTC_SECONDS
        return float("nan")

    def to_bjd(self, value: float) -> float:
        """Convert a date in this scale to a full Julian date.

        Only the *offset* is applied. The sub-minute scale difference is
        deliberately not corrected, because doing so needs the target's sky
        position and the observation's light-time; the residual is reported
        by :attr:`uncertainty_seconds` instead of being silently absorbed.
        """
        return float(value) + self.offset_to_bjd

    def from_bjd(self, bjd: float) -> float:
        return float(bjd) - self.offset_to_bjd


class EpochKind(str, Enum):
    """Which orbital event an epoch marks."""

    PERIASTRON = "PERIASTRON"
    """Time of periastron passage; ``pl_orbtper``. Directly gives M = 0."""

    TRANSIT = "TRANSIT"
    """Mid-transit; ``pl_tranmid``. Gives M via nu = pi/2 - omega."""

    MEAN_ANOMALY_AT_EPOCH = "MEAN_ANOMALY_AT_EPOCH"
    CONJUNCTION = "CONJUNCTION"
    UNKNOWN = "UNKNOWN"

    @property
    def needs_argument_of_periapsis(self) -> bool:
        """True when converting this epoch to a mean anomaly needs omega.

        Mid-transit does: the true anomaly at transit is ``pi/2 - omega``,
        so a transit epoch inherits the periastron-convention ambiguity.
        """
        return self is EpochKind.TRANSIT

    @property
    def label(self) -> str:
        return {
            EpochKind.PERIASTRON: "time of periastron passage",
            EpochKind.TRANSIT: "mid-transit time",
            EpochKind.MEAN_ANOMALY_AT_EPOCH: "mean anomaly at a reference epoch",
            EpochKind.CONJUNCTION: "time of conjunction",
            EpochKind.UNKNOWN: "unspecified epoch",
        }[self]


@dataclass(frozen=True)
class Epoch:
    """A dated orbital event, with its time system and provenance."""

    value: Parameter
    kind: EpochKind = EpochKind.UNKNOWN
    scale: TimeScale = TimeScale.JD_UNSPECIFIED
    reference: str | None = None

    @classmethod
    def missing(cls, kind: EpochKind = EpochKind.UNKNOWN) -> "Epoch":
        return cls(value=unknown(u.day, provenance="epoch"), kind=kind, scale=TimeScale.UNKNOWN)

    @property
    def is_known(self) -> bool:
        return self.value.is_known

    @property
    def status(self) -> Status:
        return self.value.status

    def as_bjd(self) -> float | None:
        """The epoch as a full Julian date, or None when unavailable."""
        raw = self.value.value_in(u.day)
        if raw is None:
            return None
        return self.scale.to_bjd(raw)

    @property
    def scale_uncertainty_days(self) -> float:
        return self.scale.uncertainty_seconds / 86400.0

    def describe(self) -> str:
        if not self.is_known:
            return "{0}: not published".format(self.kind.label)
        text = "{0}: {1:.5f} ({2})".format(self.kind.label, self.as_bjd(), self.scale.label)
        if not self.scale.is_determinate:
            text += "; time system unstated, worst case +/- {0:.0f} s".format(
                self.scale.uncertainty_seconds
            )
        return text

    def phase_uncertainty_fraction(self, period_days: float | None) -> float | None:
        """How much of an orbit the time-system ambiguity could shift the phase.

        For HD 80606 b (P = 111 d) an unstated scale is worth about
        6e-8 of a revolution - utterly negligible. For an ultra-short-period
        planet it is still small but no longer absurd to state. Reporting it
        is cheaper than arguing about it.
        """
        if period_days is None or period_days <= 0.0:
            return None
        return self.scale_uncertainty_days / period_days
