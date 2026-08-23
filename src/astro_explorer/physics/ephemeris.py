"""Kepler's third law and physical time propagation.

Covers roadmap sections 3.3 (no fake 1 AU), 3.5 (physical rather than
normalised time) and 8.7 (Kepler consistency residual).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, derived, unknown
from .constants import G

__all__ = [
    "TimeMode",
    "TimeController",
    "semimajor_axis_from_period",
    "period_from_semimajor_axis",
    "kepler_third_law_residual",
]


def _total_mass_quantity(
    stellar_mass_solar: Parameter | float | None,
    planet_mass_solar: Parameter | float | None = None,
) -> u.Quantity | None:
    """Total system mass, or None when the stellar mass is unavailable."""
    if isinstance(stellar_mass_solar, Parameter):
        star = stellar_mass_solar.value_in(u.M_sun)
    else:
        star = stellar_mass_solar
    if star is None or not np.isfinite(star) or star <= 0.0:
        return None

    if isinstance(planet_mass_solar, Parameter):
        planet = planet_mass_solar.value_in(u.M_sun, 0.0)
    else:
        planet = planet_mass_solar or 0.0
    if planet is None or not np.isfinite(planet) or planet < 0.0:
        planet = 0.0

    return (star + planet) * u.M_sun


def semimajor_axis_from_period(
    period: Parameter | float | None,
    stellar_mass_solar: Parameter | float | None,
    planet_mass_solar: Parameter | float | None = None,
) -> Parameter:
    """Derive ``a`` from ``P`` and the total mass (roadmap section 3.3).

    .. math:: a = \\left( \\frac{G (M_* + M_p) P^2}{4 \\pi^2} \\right)^{1/3}

    Returns a DERIVED :class:`~astro_explorer.provenance.Parameter` in AU, or
    an UNKNOWN one when the inputs are insufficient.  It never returns 1 AU
    as a stand-in.

    The fractional uncertainty is propagated from ``P`` and ``M`` through
    ``da/a = (2/3) dP/P + (1/3) dM/M``.
    """
    period_param = period if isinstance(period, Parameter) else None
    period_days = period_param.value_in(u.day) if period_param else period
    if period_days is None or not np.isfinite(period_days) or period_days <= 0.0:
        return unknown(u.au, provenance="kepler3", note="orbital period unavailable")

    total_mass = _total_mass_quantity(stellar_mass_solar, planet_mass_solar)
    if total_mass is None:
        return unknown(u.au, provenance="kepler3", note="stellar mass unavailable")

    period_q = period_days * u.day
    axis = np.cbrt(G * total_mass * period_q**2 / (4.0 * np.pi**2)).to(u.au)

    # Propagate the dominant fractional errors when we have them.
    rel_period = 0.0
    if period_param is not None and period_param.error_plus is not None and period_days:
        rel_period = period_param.error_plus / abs(period_days)
    rel_mass = 0.0
    if isinstance(stellar_mass_solar, Parameter) and stellar_mass_solar.error_plus is not None:
        star_value = stellar_mass_solar.value_in(u.M_sun)
        if star_value:
            rel_mass = stellar_mass_solar.error_plus / abs(star_value)
    rel_axis = (2.0 / 3.0) * rel_period + (1.0 / 3.0) * rel_mass
    error = float(axis.value) * rel_axis if rel_axis > 0.0 else None

    return derived(
        float(axis.value),
        u.au,
        error_plus=error,
        error_minus=error,
        provenance="kepler3(period, stellar_mass)",
        note="derived from Kepler's third law, not a published semimajor axis",
    )


def period_from_semimajor_axis(
    semimajor_axis: Parameter | float | None,
    stellar_mass_solar: Parameter | float | None,
    planet_mass_solar: Parameter | float | None = None,
) -> Parameter:
    """Derive ``P`` from ``a`` and the total mass.

    .. math:: P = 2 \\pi \\sqrt{ \\frac{a^3}{G (M_* + M_p)} }
    """
    axis_param = semimajor_axis if isinstance(semimajor_axis, Parameter) else None
    axis_au = axis_param.value_in(u.au) if axis_param else semimajor_axis
    if axis_au is None or not np.isfinite(axis_au) or axis_au <= 0.0:
        return unknown(u.day, provenance="kepler3", note="semimajor axis unavailable")

    total_mass = _total_mass_quantity(stellar_mass_solar, planet_mass_solar)
    if total_mass is None:
        return unknown(u.day, provenance="kepler3", note="stellar mass unavailable")

    axis_q = axis_au * u.au
    period = (2.0 * np.pi * np.sqrt(axis_q**3 / (G * total_mass))).to(u.day)
    return derived(
        float(period.value),
        u.day,
        provenance="kepler3(semimajor_axis, stellar_mass)",
        note="derived from Kepler's third law, not a published period",
    )


def kepler_third_law_residual(
    period: Parameter | float | None,
    semimajor_axis: Parameter | float | None,
    stellar_mass_solar: Parameter | float | None,
    planet_mass_solar: Parameter | float | None = None,
) -> float | None:
    """Fractional disagreement between published ``a`` and the derived one.

    Roadmap section 8.7 suggests surfacing this as an educational
    consistency check.  Returns ``(a_published - a_derived) / a_derived`` or
    None when either value is unavailable.
    """
    published = semimajor_axis.value_in(u.au) if isinstance(semimajor_axis, Parameter) else semimajor_axis
    if published is None or not np.isfinite(published) or published <= 0.0:
        return None

    predicted = semimajor_axis_from_period(period, stellar_mass_solar, planet_mass_solar)
    if not predicted.is_known:
        return None
    return float((published - predicted.value) / predicted.value)


class TimeMode(str, Enum):
    """The three time models the UI must never confuse (roadmap 3.5)."""

    REAL = "REAL"
    """Wall-clock time; the planet is where the ephemeris says it is."""

    SCALED = "SCALED"
    """Simulated time running at a user-chosen multiple of real time."""

    NORMALIZED = "NORMALIZED"
    """Educational mode: every orbit completes in the same wall-clock time."""

    @property
    def label(self) -> str:
        return {
            TimeMode.REAL: "Real / physical time",
            TimeMode.SCALED: "Simulation time scale",
            TimeMode.NORMALIZED: "Normalised educational orbit",
        }[self]


#: Julian date of the Unix epoch, used to place wall-clock time on the
#: Julian-day axis that exoplanet ephemerides use.
JD_UNIX_EPOCH = 2440587.5


@dataclass
class TimeController:
    """Maps frames or wall-clock seconds onto a mean anomaly.

    This replaces the prototype's ``M = 2*pi*frame/160``, which gave every
    planet the same apparent orbital period regardless of physics.

    Attributes
    ----------
    mode:
        Which of the three time models is active.
    scale_days_per_second:
        In :attr:`TimeMode.SCALED`, how many simulated days pass per real
        second.
    normalized_period_seconds:
        In :attr:`TimeMode.NORMALIZED`, the wall-clock duration of one full
        orbit for every object.
    epoch_jd:
        Simulation start on the full Julian-day axis, for
        :attr:`TimeMode.SCALED`. Named ``_jd`` rather than ``_bjd``: the
        origin is whatever full Julian day the caller handed over, and
        nothing here promises it is exact BJD_TDB. Only the epoch's own
        :class:`~astro_explorer.physics.epoch.TimeScale` can say that, and
        it travels with the epoch rather than with this clock.
    """

    mode: TimeMode = TimeMode.SCALED
    scale_days_per_second: float = 1.0
    normalized_period_seconds: float = 8.0
    epoch_jd: float = JD_UNIX_EPOCH

    def simulated_jd(self, elapsed_seconds: float, *, now_jd: float | None = None) -> float | None:
        """Full Julian date represented by ``elapsed_seconds``.

        Not necessarily BJD_TDB - see :attr:`epoch_jd`. Returns None in
        NORMALIZED mode, where there is no physical date at all.
        """
        if self.mode is TimeMode.REAL:
            if now_jd is None:
                raise ValueError("REAL time mode needs the current Julian date")
            return now_jd
        if self.mode is TimeMode.SCALED:
            return self.epoch_jd + elapsed_seconds * self.scale_days_per_second
        return None

    # -- deprecated aliases ----------------------------------------------
    # Follow-up review section 4: the old names claimed a barycentric
    # dynamical scale this clock never guaranteed. Kept briefly so an
    # out-of-tree caller gets a warning rather than an AttributeError.
    @property
    def epoch_bjd(self) -> float:
        """Deprecated alias for :attr:`epoch_jd`."""
        warnings.warn(
            "TimeController.epoch_bjd is deprecated; use epoch_jd",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.epoch_jd

    @epoch_bjd.setter
    def epoch_bjd(self, value: float) -> None:
        warnings.warn(
            "TimeController.epoch_bjd is deprecated; use epoch_jd",
            DeprecationWarning,
            stacklevel=2,
        )
        self.epoch_jd = float(value)

    def simulated_bjd(self, elapsed_seconds: float, *, now_bjd: float | None = None) -> float | None:
        """Deprecated alias for :meth:`simulated_jd`."""
        warnings.warn(
            "TimeController.simulated_bjd() is deprecated; use simulated_jd()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.simulated_jd(elapsed_seconds, now_jd=now_bjd)

    def mean_anomaly(
        self,
        elements,
        elapsed_seconds: float,
        *,
        now_jd: float | None = None,
    ) -> tuple[float | None, bool]:
        """Mean anomaly to display, and whether the phase is assumed.

        Returns ``(mean_anomaly_rad, phase_is_assumed)``.  A ``None`` mean
        anomaly means the orbit may be drawn but the planet's position along
        it is not defined; the caller must not silently place it at
        periapsis.
        """
        if self.mode is TimeMode.NORMALIZED:
            fraction = (elapsed_seconds % self.normalized_period_seconds) / self.normalized_period_seconds
            return 2.0 * np.pi * fraction, True

        time_jd = self.simulated_jd(elapsed_seconds, now_jd=now_jd)
        if time_jd is None:
            return None, True

        physical = elements.mean_anomaly_at(time_jd)
        if physical is not None:
            return physical, not elements.can_compute_current_position

        # Period known but no epoch: the orbital *rate* is physical even
        # though the absolute phase is not.  Advance from an arbitrary zero
        # and say so.
        n = elements.mean_motion_rad_per_day
        if n is None:
            return None, True
        return float(np.mod(n * (time_jd - self.epoch_jd), 2.0 * np.pi)), True

    def describe(self) -> str:
        if self.mode is TimeMode.REAL:
            return "Real / physical time"
        if self.mode is TimeMode.SCALED:
            return "Simulation time: {0:g} day(s) per second".format(self.scale_days_per_second)
        return "Normalised educational orbit: {0:g} s per revolution (not physical)".format(
            self.normalized_period_seconds
        )
