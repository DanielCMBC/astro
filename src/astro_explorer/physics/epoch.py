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

The differences matter at three quite different magnitudes, and they must
not be conflated:

* **Heliocentric versus barycentric** is the Sun's own motion about the
  solar-system barycentre - up to ~1.6 million km, which is only about
  8 *seconds* of light travel.
* **Geocentric versus barycentric** is the Earth's orbital displacement,
  up to one AU, which is about 499 s - the familiar "8 minutes". A plain
  JD is geocentric, so this is the term an unlabelled JD carries; an HJD
  has already had almost all of it removed.
* **TDB versus UTC** is the accumulated leap seconds, currently 69.184 s.

None of these is visible on a 111-day orbit, but all are fatal to
transit-timing work, and the mission offsets are catastrophic if ignored:
2454833 days is thirteen years.

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

__all__ = [
    "TimeScale",
    "EpochKind",
    "Epoch",
    "MeanAnomalyAnchor",
    "tdb_minus_utc_seconds",
    "astropy_time",
    "orbital_time_jd",
    "INSTANT_MATCH_TOLERANCE_DAYS",
    "ASTROPY_SCALE_FOR_UNSTATED",
    "TDB_MINUS_UTC_FALLBACK_SECONDS",
    "BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS",
    "BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS",
]

#: TDB - UTC for the current era, used only when no date is available or
#: when astropy cannot convert the one it was given.
#:
#: This is deliberately *not* named as a constant of nature. ``TDB - UTC``
#: is ``(TT - UTC) + (TDB - TT)``: the first term is the accumulated leap
#: seconds, which a future IERS announcement can change, and the second is a
#: periodic relativistic term of order a millisecond that is never exactly
#: zero. Freezing 69.184 into the production path would encode a 2026-era
#: relationship as physics. Use :func:`tdb_minus_utc_seconds` instead
#: wherever the date being handled is known.
TDB_MINUS_UTC_FALLBACK_SECONDS = 69.184


def tdb_minus_utc_seconds(jd: float | None = None) -> float:
    """``TDB - UTC`` in seconds at ``jd``, a full Julian date in UTC.

    Delegates to astropy, which owns the leap-second table and the
    relativistic ``TDB - TT`` series, rather than re-deriving either. The
    difference is taken between the two scales' Julian-day *numbering* -
    using the ``jd1``/``jd2`` pair so the sub-millisecond term survives -
    because the two are the same instant and subtracting them as times
    would correctly give zero.

    Falls back to :data:`TDB_MINUS_UTC_FALLBACK_SECONDS` when the date is
    unknown or out of the range astropy will convert, so a bad date degrades
    the *precision* of a stated uncertainty rather than raising inside a
    render loop.
    """
    if jd is None or not np.isfinite(jd):
        return TDB_MINUS_UTC_FALLBACK_SECONDS
    try:
        from astropy.time import Time

        utc = Time(float(jd), format="jd", scale="utc")
        tdb = utc.tdb
        days = (tdb.jd1 - utc.jd1) + (tdb.jd2 - utc.jd2)
        seconds = float(days * 86400.0)
    except Exception:  # pragma: no cover - astropy refusing a wild date
        return TDB_MINUS_UTC_FALLBACK_SECONDS
    if not np.isfinite(seconds):
        return TDB_MINUS_UTC_FALLBACK_SECONDS
    return seconds

#: Largest barycentric-minus-heliocentric reference-frame difference. This
#: is the Sun's displacement from the solar-system barycentre, up to ~1.6
#: million km, expressed as light travel time - a few seconds, *not* the
#: 8-minute figure that belongs to the geocentric term below.
BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS = 8.0

#: Largest barycentric-minus-geocentric light-time difference: one AU of
#: light travel, ~499 s. This is what an unlabelled, geocentric JD can be
#: wrong by, and it is roughly sixty times the heliocentric term.
BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS = 499.0


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
    def frame_uncertainty_seconds(self) -> float:
        """The *reference-frame* term this scale still owes BJD_TDB.

        The two frame terms differ by nearly two orders of magnitude and
        must not share one constant: an HJD has already removed the Earth's
        orbital light time and owes only the Sun's barycentric wobble,
        whereas a plain JD is geocentric and owes the whole AU.
        """
        if self is TimeScale.BJD_TDB:
            return 0.0
        if self in (TimeScale.BKJD, TimeScale.BTJD):
            # Documented as BJD_TDB once the mission offset is applied.
            return 0.0
        if self is TimeScale.HJD_UTC:
            return BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS
        if self is TimeScale.JD_UTC:
            return BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS
        if self is TimeScale.JD_UNSPECIFIED:
            # Plain geocentric JD_UTC remains a live reading of an unlabelled
            # archive date, so the bound must cover that worst case rather
            # than the milder heliocentric one.
            return BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS
        return float("nan")

    @property
    def has_utc_offset(self) -> bool:
        """True when the scale is quoted in UTC and still owes TDB - UTC."""
        return self in (
            TimeScale.HJD_UTC,
            TimeScale.JD_UTC,
            TimeScale.JD_UNSPECIFIED,
        )

    def uncertainty_seconds_at(self, jd: float | None) -> float:
        """How wrong converting to BJD_TDB could be at ``jd``, in seconds.

        Zero when the scale is stated barycentric; otherwise the frame term
        plus ``TDB - UTC`` *evaluated at the date in question* rather than
        taken from a frozen present-day literal. Pass ``None`` when there is
        no date and the current-era fallback is acceptable.
        """
        frame = self.frame_uncertainty_seconds
        if not np.isfinite(frame):
            return float("nan")
        if not self.has_utc_offset:
            return frame
        return frame + tdb_minus_utc_seconds(jd)

    @property
    def uncertainty_seconds(self) -> float:
        """:meth:`uncertainty_seconds_at` with no date, for the current era.

        Kept as a property because plenty of callers only have a scale in
        hand. Anything that *does* know the instant should ask
        :meth:`uncertainty_seconds_at`, which is what :class:`Epoch` does.
        """
        return self.uncertainty_seconds_at(None)

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


#: Astropy time scale for each stated catalogue convention. The mission
#: offsets are *not* here: they are days, they belong to
#: :meth:`TimeScale.to_bjd`, and a scale table that quietly also shifted the
#: date would be two corrections wearing one name.
_ASTROPY_SCALE = {
    TimeScale.BJD_TDB: "tdb",
    TimeScale.BKJD: "tdb",
    TimeScale.BTJD: "tdb",
    TimeScale.HJD_UTC: "utc",
    TimeScale.JD_UTC: "utc",
}

#: What an unstated Julian date is read as when it must become an
#: :class:`~astropy.time.Time`. See :func:`astropy_time` for why this is a
#: defensible assumption *here* and nowhere near a transit ephemeris.
ASTROPY_SCALE_FOR_UNSTATED = "tdb"

#: Two instants closer than this are the same instant, project-wide. About
#: 0.09 s - far below any astrometric reference epoch's own precision, far
#: below the sub-minute ambiguity an unstated time scale already carries,
#: and far above the float64 noise in a Julian date near 2.46e6.
#:
#: It lives here because "the host, the target star and the planet are at
#: one time" has to mean the same thing on the stellar side and the orbital
#: side. Two tolerances would eventually disagree, and the disagreement
#: would show up as a separation that is accepted from one direction and
#: refused from the other.
INSTANT_MATCH_TOLERANCE_DAYS = 1e-6


def astropy_time(jd: float, scale: TimeScale = TimeScale.JD_UNSPECIFIED):
    """The explorer's clock as an :class:`astropy.time.Time`.

    **The single conversion point** from the explorer's physical clock - a
    full Julian date on the canonical axis, plus the
    :class:`TimeScale` it was published in - to the Astropy time that
    stellar space motion is evaluated at. Explorer C3.6 needs one because
    the alternative is an astrometric API taking a bare ``time_jd: float``,
    and a float has no scale: the host, the target star and the planet would
    each be free to be read in a different one.

    ``jd`` is a *full* Julian date, so a caller holding a mission-offset
    date must have gone through :attr:`Epoch.canonical_jd` first. Passing a
    raw BKJD here would be thirteen years wrong and there is nothing in a
    float that could catch it.

    An unstated scale is read as :data:`ASTROPY_SCALE_FOR_UNSTATED`. That is
    an assumption, and it is stated rather than hidden: the residual is at
    most about 569 s (:attr:`TimeScale.uncertainty_seconds_at`), which for
    space motion is 1.8e-5 yr - on the fastest known proper motion, about
    2e-4 mas, some three orders of magnitude below Gaia's own position
    uncertainty. The same 569 s is fatal to a transit ephemeris, which is
    exactly why it keeps travelling on the :class:`TimeScale` instead of
    being declared absorbed here.
    """
    from astropy.time import Time

    value = float(jd)
    if not np.isfinite(value):
        raise ValueError("an astrometric time must be a finite Julian date")
    return Time(value, format="jd", scale=_ASTROPY_SCALE.get(scale, ASTROPY_SCALE_FOR_UNSTATED))


def orbital_time_jd(time, scale: TimeScale = TimeScale.JD_UNSPECIFIED) -> float:
    """A physical instant as a Julian date on the orbital clock's own axis.

    **The single conversion point in the other direction**, and the one
    C3.6 needs so the stellar side and the orbital side can be at the same
    instant rather than at the same number.

    :func:`astropy_time` turns a catalogue Julian date into an
    :class:`~astropy.time.Time`; this turns a ``Time`` back into the Julian
    date :meth:`OrbitalElements.phase_at` consumes. The two are exact
    inverses, because both go through the same scale table: the instant is
    expressed *in the scale the orbital epoch was published in*, so the
    difference being taken inside ``phase_at`` - ``t - t0`` - is a
    difference between two dates on one axis rather than between a TDB date
    and a UTC one.

    The alternative is what this exists to prevent: call sites
    independently reaching for ``time.jd``, ``time.tdb.jd``, ``time.tcb.jd``
    or ``time.utc.jd``. Those differ by up to about 69 seconds today and by
    ~20 s more for TCB, all of which is invisible in a rendered orbit and
    none of which is invisible in a transit ephemeris.

    An unstated ``scale`` reads the instant as
    :data:`ASTROPY_SCALE_FOR_UNSTATED`, matching :func:`astropy_time`. That
    does **not** make the scale known: the published epoch is still
    undetermined to within :attr:`TimeScale.uncertainty_seconds_at`, and
    that uncertainty keeps travelling on the :class:`Epoch`. A precise
    target time cannot resolve an imprecise reference one.
    """
    target = getattr(time, _ASTROPY_SCALE.get(scale, ASTROPY_SCALE_FOR_UNSTATED))
    return float(target.jd)


class EpochKind(str, Enum):
    """Which orbital event an epoch marks."""

    PERIASTRON = "PERIASTRON"
    """Time of periastron passage; ``pl_orbtper``. Directly gives M = 0."""

    TRANSIT = "TRANSIT"
    """Mid-transit; ``pl_tranmid``. Gives M via nu = pi/2 - omega."""

    MEAN_ANOMALY_AT_EPOCH = "MEAN_ANOMALY_AT_EPOCH"
    """The reference instant ``t0`` a mean anomaly was quoted at.

    The *date*, not the angle. ``M0`` itself lives in
    :class:`MeanAnomalyAnchor`, because an angle in an epoch slot cannot be
    subtracted from a Julian date and must not look as though it can.
    """

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

    @property
    def is_dated(self) -> bool:
        """True when this epoch's value is an instant rather than an angle.

        The check is on the *unit*, so it holds however the epoch was
        built. Asking an angle for a Julian date is a category error, and it
        is refused rather than coerced - see :class:`MeanAnomalyAnchor`,
        which keeps ``M0`` and its reference date in separate fields for
        exactly this reason.
        """
        return self.value.unit.physical_type == "time"

    @property
    def canonical_jd(self) -> float | None:
        """The epoch as a full Julian date, or None when unavailable.

        This is the *only* supported route from a catalogue time into the
        explorer's clock and into :meth:`OrbitalElements.phase_at`. The
        mission offset for :attr:`TimeScale.BKJD` and
        :attr:`TimeScale.BTJD` is applied here; the sub-minute scale
        difference deliberately is not, and travels alongside as
        :attr:`scale_uncertainty_days`.

        The result is a full Julian date, not a value certified to be
        ``BJD_TDB``. Only when :attr:`scale` is already
        :attr:`TimeScale.BJD_TDB` is it exactly that.
        """
        if not self.is_dated:
            return None
        raw = self.value.value_in(u.day)
        if raw is None:
            return None
        return self.scale.to_bjd(raw)

    def as_bjd(self) -> float | None:
        """Legacy alias for :attr:`canonical_jd`.

        Kept so existing callers keep working, but the name overstates what
        the value is: unless :attr:`scale` is already
        :attr:`TimeScale.BJD_TDB`, the result is a full Julian date and not
        a barycentric dynamical one. New code should say ``canonical_jd``.
        """
        return self.canonical_jd

    @property
    def scale_uncertainty_seconds(self) -> float:
        """The residual to BJD_TDB, evaluated at this epoch's own date.

        The production path: an epoch knows when it is, so the leap-second
        count it owes is the one that applied *then*, not the one that
        happens to apply while the program is running.
        """
        return self.scale.uncertainty_seconds_at(self.canonical_jd)

    @property
    def scale_uncertainty_days(self) -> float:
        return self.scale_uncertainty_seconds / 86400.0

    def describe(self) -> str:
        if not self.is_known:
            return "{0}: not published".format(self.kind.label)
        if not self.is_dated:
            return "{0}: {1}".format(self.kind.label, self.value.format())
        text = "{0}: {1:.5f} ({2})".format(self.kind.label, self.canonical_jd, self.scale.label)
        if not self.scale.is_determinate:
            text += "; time system unstated, worst case +/- {0:.0f} s".format(
                self.scale_uncertainty_seconds
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


@dataclass(frozen=True)
class MeanAnomalyAnchor:
    """A published mean anomaly together with the instant it applies at.

    Final review section 3. ``M0`` alone does not place a planet: the
    propagation law is

    .. math:: M(t) = M_0 + n (t - t_0)

    so without ``t0`` the published angle constrains the orbit at *some*
    unstated moment and nothing at any other. The catalogue routinely
    publishes one without the other, so the two travel together in one
    object that can be asked whether it is actually usable, rather than as
    a loose parameter that reads as a phase.
    """

    anomaly: Parameter
    epoch: Epoch

    @classmethod
    def missing(cls) -> "MeanAnomalyAnchor":
        return cls(
            anomaly=unknown(u.rad, provenance="mean anomaly at epoch"),
            epoch=Epoch.missing(EpochKind.MEAN_ANOMALY_AT_EPOCH),
        )

    @property
    def is_known(self) -> bool:
        """True when the *angle* was published, dated or not."""
        return self.anomaly.is_known

    @property
    def is_dated(self) -> bool:
        """True when both ``M0`` and ``t0`` are available.

        The only state from which a current position may be claimed.
        """
        return self.is_known and self.epoch.is_known and self.epoch.is_dated

    @property
    def anomaly_rad(self) -> float | None:
        """``M0`` in radians, wrapped to ``[0, 2pi)``."""
        value = self.anomaly.value_in(u.rad)
        if value is None:
            return None
        return float(np.mod(value, 2.0 * np.pi))

    @property
    def reference_jd(self) -> float | None:
        """``t0`` as a full Julian date, mission offset applied."""
        return self.epoch.canonical_jd

    def mean_anomaly_at(self, time_jd: float, mean_motion_rad_per_day: float | None):
        """``M(t)``, or None when the anchor cannot support the question.

        Returns None for an undated anchor rather than falling back to
        ``M0``: reporting the reference angle as though it were the current
        one is wrong at every instant except the one nobody published.
        """
        if not self.is_dated or mean_motion_rad_per_day is None:
            return None
        m0 = self.anomaly_rad
        t0 = self.reference_jd
        if m0 is None or t0 is None:
            return None
        return float(np.mod(m0 + mean_motion_rad_per_day * (float(time_jd) - t0), 2.0 * np.pi))

    def describe(self) -> str:
        if not self.is_known:
            return "mean anomaly at epoch: not published"
        angle = self.anomaly.to(u.deg).format()
        if self.is_dated:
            return "mean anomaly {0} at {1:.5f} ({2})".format(
                angle, self.reference_jd, self.epoch.scale.label
            )
        return "mean anomaly {0}, but its reference epoch is not published".format(angle)

