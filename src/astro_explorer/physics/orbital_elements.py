"""Orbital elements and the perifocal-to-reference rotation.

Covers roadmap sections 4.3, 4.4, 4.5, 8.1, 8.5, 8.6 and 8.7.

Two rules are enforced here rather than left to callers:

* a missing element is never replaced by an Earth-like default; the element
  keeps its :class:`~astro_explorer.provenance.Status`, and the caller can
  see which parts of the geometry are real;
* the orbit's *shape*, its *orientation* and its *phase* are tracked as three
  separate knowledge states, because an exoplanet routinely has a well
  measured shape, a partly measured orientation and no usable phase at all.

Review sections 9-11 add a third rule: an element's *meaning* is recorded
alongside its value. The raw ``argument_of_periastron`` is stored exactly as
the catalogue gave it, and :attr:`OrbitalElements.periastron_convention`
says whose orbit it describes. Nothing infers the convention from the
number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status, assumed, derived, unknown
from .epoch import Epoch, EpochKind, TimeScale
from .kepler import solve_kepler, true_anomaly_from_eccentric
from .orbital_semantics import (
    OrbitValidity,
    PeriastronConvention,
    resolve_argument_of_periapsis,
)
from .orientation import (
    perifocal_position,
    position_from_eccentric_anomaly,
    rotation_perifocal_to_inertial,
)

__all__ = [
    "PhaseKnowledge",
    "OrbitValidity",
    "PeriastronConvention",
    "OrbitalElements",
    "rotation_perifocal_to_reference",
    "perifocal_position",
    "orbital_radius",
]


class PhaseKnowledge(str, Enum):
    """How much of the orbit we actually know (roadmap section 4.5)."""

    ORBIT_SHAPE_KNOWN = "ORBIT_SHAPE_KNOWN"
    """a and e are available; the ellipse can be drawn."""

    ORBIT_PHASE_CONSTRAINED = "ORBIT_PHASE_CONSTRAINED"
    """An epoch (periastron or transit) plus a period exist."""

    CURRENT_POSITION_COMPUTABLE = "CURRENT_POSITION_COMPUTABLE"
    """Shape, period and epoch are all present: 'where is it now' is real."""

    DISPLAY_PHASE_ASSUMED = "DISPLAY_PHASE_ASSUMED"
    """The planet is drawn somewhere on the ellipse for illustration only."""


def _param(value, unit) -> Parameter:
    """Coerce ``value`` to a Parameter without inventing provenance."""
    if isinstance(value, Parameter):
        return value.to(unit) if value.unit != unit else value
    if value is None:
        return unknown(unit)
    return Parameter(value, unit, status=Status.MEASURED, provenance="caller")


@dataclass(frozen=True)
class OrbitalElements:
    """A provenance-aware Keplerian element set.

    All angles are stored in radians, ``semimajor_axis`` in AU and
    ``period`` in days.  Elements that were never published stay UNKNOWN;
    :meth:`for_display` is the only place that substitutes anything, and
    everything it substitutes is tagged ASSUMED_FOR_VISUALIZATION.
    """

    name: str = ""
    semimajor_axis: Parameter = unknown(u.au)
    eccentricity: Parameter = unknown()
    period: Parameter = unknown(u.day)
    inclination: Parameter = unknown(u.rad)
    argument_of_periastron: Parameter = unknown(u.rad)
    longitude_of_ascending_node: Parameter = unknown(u.rad)
    epoch_periastron: Parameter = unknown(u.day)  # BJD
    epoch_transit: Parameter = unknown(u.day)  # BJD
    mean_anomaly_at_epoch: Parameter = unknown(u.rad)

    #: Whose orbit ``argument_of_periastron`` describes.  Never inferred from
    #: the value; defaults to AS_REPORTED because that is what a catalogue
    #: without a convention column actually tells us (review section 10).
    periastron_convention: PeriastronConvention = PeriastronConvention.AS_REPORTED

    #: Time system the epochs are quoted in (review section 11).
    epoch_scale: TimeScale = TimeScale.JD_UNSPECIFIED

    #: Publication the elements came from, kept with them rather than only
    #: on the record that owns them.
    reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "semimajor_axis", _param(self.semimajor_axis, u.au))
        object.__setattr__(self, "eccentricity", _param(self.eccentricity, u.dimensionless_unscaled))
        object.__setattr__(self, "period", _param(self.period, u.day))
        object.__setattr__(self, "inclination", _param(self.inclination, u.rad))
        object.__setattr__(self, "argument_of_periastron", _param(self.argument_of_periastron, u.rad))
        object.__setattr__(
            self, "longitude_of_ascending_node", _param(self.longitude_of_ascending_node, u.rad)
        )
        object.__setattr__(self, "epoch_periastron", _param(self.epoch_periastron, u.day))
        object.__setattr__(self, "epoch_transit", _param(self.epoch_transit, u.day))
        object.__setattr__(self, "mean_anomaly_at_epoch", _param(self.mean_anomaly_at_epoch, u.rad))

        # An absent angle has no convention to speak of.
        if not self.argument_of_periastron.is_known:
            object.__setattr__(self, "periastron_convention", PeriastronConvention.UNKNOWN)

    # -- knowledge state -------------------------------------------------
    @property
    def shape_known(self) -> bool:
        """True when the ellipse itself can be drawn from published values."""
        return self.semimajor_axis.is_scientific and self.eccentricity.is_scientific

    @property
    def orientation_known(self) -> bool:
        """True only when i, omega and Omega are all measured.

        For the great majority of exoplanets this is False, because the
        longitude of the ascending node is not observable from transits or
        radial velocity alone.
        """
        return (
            self.inclination.is_scientific
            and self.argument_of_periastron.is_scientific
            and self.longitude_of_ascending_node.is_scientific
        )

    @property
    def phase_knowledge(self) -> PhaseKnowledge:
        """Strongest phase statement the published elements support."""
        has_epoch = (
            self.epoch_periastron.is_known
            or self.epoch_transit.is_known
            or self.mean_anomaly_at_epoch.is_known
        )
        if not self.shape_known:
            return PhaseKnowledge.DISPLAY_PHASE_ASSUMED
        if has_epoch and self.period.is_scientific:
            return PhaseKnowledge.CURRENT_POSITION_COMPUTABLE
        if has_epoch:
            return PhaseKnowledge.ORBIT_PHASE_CONSTRAINED
        return PhaseKnowledge.ORBIT_SHAPE_KNOWN

    @property
    def can_compute_current_position(self) -> bool:
        return self.phase_knowledge is PhaseKnowledge.CURRENT_POSITION_COMPUTABLE

    @property
    def argument_of_periapsis_planet(self) -> Parameter:
        """The planet-frame argument of periapsis implied by the raw value.

        The raw element is never modified. This is the value the 3D
        transform should use, and its status records how much was known:
        MEASURED under a stated planet convention, DERIVED after a
        stellar-reflex conversion, ASSUMED_FOR_VISUALIZATION when the
        catalogue never said (review section 10).
        """
        return resolve_argument_of_periapsis(
            self.argument_of_periastron, self.periastron_convention
        )

    @property
    def periastron_convention_is_assumed(self) -> bool:
        """True when periapsis orientation rests on an unstated convention."""
        return (
            self.argument_of_periastron.is_known
            and not self.periastron_convention.is_determinate
        )

    @property
    def validity(self) -> OrbitValidity:
        """What the published elements support (review section 11)."""
        flags = OrbitValidity.NONE

        if self.shape_known:
            flags |= OrbitValidity.GEOMETRY_VALID
        if self.can_compute_current_position:
            flags |= OrbitValidity.PHASE_VALID

        known_angles = sum(
            1
            for parameter in (
                self.inclination,
                self.argument_of_periastron,
                self.longitude_of_ascending_node,
            )
            if parameter.is_scientific
        )
        # ORIENTATION_FULL additionally requires that omega's convention be
        # stated: three known angles under an unknown convention still leave
        # periapsis ambiguous by 180 degrees.
        if known_angles == 3 and self.periastron_convention.is_determinate:
            flags |= OrbitValidity.ORIENTATION_FULL
        elif known_angles > 0:
            flags |= OrbitValidity.ORIENTATION_PARTIAL

        return flags

    @property
    def epochs(self) -> list[Epoch]:
        """Published epochs, each with its kind and time system."""
        found: list[Epoch] = []
        if self.epoch_periastron.is_known:
            found.append(
                Epoch(self.epoch_periastron, EpochKind.PERIASTRON, self.epoch_scale, self.reference)
            )
        if self.epoch_transit.is_known:
            found.append(
                Epoch(self.epoch_transit, EpochKind.TRANSIT, self.epoch_scale, self.reference)
            )
        if self.mean_anomaly_at_epoch.is_known:
            found.append(
                Epoch(
                    self.mean_anomaly_at_epoch,
                    EpochKind.MEAN_ANOMALY_AT_EPOCH,
                    self.epoch_scale,
                    self.reference,
                )
            )
        return found

    # -- display normalisation -------------------------------------------
    def for_display(self) -> "OrbitalElements":
        """Fill only what a renderer strictly needs, tagged as assumptions.

        Roadmap sections 3.4 and 4.3: unknown eccentricity may be drawn as a
        circle and an unknown node normalised to zero, but the substitution
        must be visible in the parameter's status.  A missing semimajor axis
        is *not* filled: without it there is no orbit to draw at all, and
        inventing 1 AU is exactly the bug this replaces.
        """
        updates: dict[str, Parameter] = {}

        if not self.eccentricity.is_known:
            updates["eccentricity"] = assumed(
                0.0,
                provenance="display-normalisation",
                note="no published eccentricity; drawn as a circular orbit",
            )
        if not self.inclination.is_known:
            updates["inclination"] = assumed(
                0.0,
                u.rad,
                provenance="display-normalisation",
                note="inclination unknown; orbit drawn face-on",
            )
        resolved = self.argument_of_periapsis_planet
        if not resolved.is_known:
            updates["argument_of_periastron"] = assumed(
                0.0,
                u.rad,
                provenance="display-normalisation",
                note="argument of periastron unknown; normalised to 0 deg",
            )
        elif resolved is not self.argument_of_periastron:
            # Either converted from the stellar reflex orbit, or used under
            # an assumed convention.  Either way the drawn value is not the
            # raw catalogue number, so it travels with its own status.
            updates["argument_of_periastron"] = resolved
            updates["periastron_convention"] = PeriastronConvention.PLANET
        if not self.longitude_of_ascending_node.is_known:
            updates["longitude_of_ascending_node"] = assumed(
                0.0,
                u.rad,
                provenance="display-normalisation",
                note="ascending node not observable; normalised to 0 deg",
            )

        return replace(self, **updates) if updates else self

    # -- geometry --------------------------------------------------------
    @property
    def periapsis(self) -> Parameter:
        """Closest approach distance ``a(1-e)``."""
        if not self.semimajor_axis.is_known or not self.eccentricity.is_known:
            return unknown(u.au, provenance="a(1-e)")
        a = self.semimajor_axis.value_in(u.au)
        e = self.eccentricity.value
        return derived(a * (1.0 - e), u.au, provenance="a(1-e)")

    @property
    def apoapsis(self) -> Parameter:
        """Farthest distance ``a(1+e)``."""
        if not self.semimajor_axis.is_known or not self.eccentricity.is_known:
            return unknown(u.au, provenance="a(1+e)")
        a = self.semimajor_axis.value_in(u.au)
        e = self.eccentricity.value
        return derived(a * (1.0 + e), u.au, provenance="a(1+e)")

    @property
    def mean_motion_rad_per_day(self) -> float | None:
        """``n = 2 pi / P`` in rad/day, or None when the period is unknown."""
        period = self.period.value_in(u.day)
        if period is None or period <= 0.0:
            return None
        return 2.0 * np.pi / period

    def mean_anomaly_at(self, time_bjd: float) -> float | None:
        """Mean anomaly at barycentric Julian date ``time_bjd``.

        Returns None when the phase is not constrained; the caller must then
        either show the orbit without a planet marker or explicitly opt into
        an assumed display phase.
        """
        n = self.mean_motion_rad_per_day
        if n is None:
            return None

        if self.epoch_periastron.is_known:
            t0 = self.epoch_periastron.value_in(u.day)
            return float(np.mod(n * (time_bjd - t0), 2.0 * np.pi))

        if self.epoch_transit.is_known and self.eccentricity.is_known:
            # At mid-transit the true anomaly is nu = pi/2 - omega.
            omega = self.argument_of_periastron.value_in(u.rad)
            if omega is None:
                return None
            from .kepler import eccentric_from_true_anomaly, mean_anomaly_from_eccentric

            e = self.eccentricity.value
            nu_transit = 0.5 * np.pi - omega
            ecc_anom = eccentric_from_true_anomaly(nu_transit, e)
            m_transit = mean_anomaly_from_eccentric(ecc_anom, e)
            t0 = self.epoch_transit.value_in(u.day)
            return float(np.mod(m_transit + n * (time_bjd - t0), 2.0 * np.pi))

        if self.mean_anomaly_at_epoch.is_known:
            return float(np.mod(self.mean_anomaly_at_epoch.value_in(u.rad), 2.0 * np.pi))

        return None

    def describe_orientation(self) -> list[str]:
        """UI lines that state exactly what is measured (roadmap 4.3)."""

        def line(label: str, param: Parameter) -> str:
            if param.status is Status.MEASURED:
                return "{0}: {1}".format(label, param.to(u.deg).format())
            if param.is_assumed:
                return "{0}: unknown (display normalisation {1})".format(
                    label, param.to(u.deg).format(with_status=False)
                )
            if param.status is Status.DERIVED:
                return "{0}: {1}".format(label, param.to(u.deg).format())
            return "{0}: unknown".format(label)

        return [
            line("Inclination", self.inclination),
            line("Argument of periastron", self.argument_of_periastron),
            line("Ascending node", self.longitude_of_ascending_node),
            "Phase knowledge: {0}".format(self.phase_knowledge.value),
        ]


#: The transform itself lives in :mod:`astro_explorer.physics.orientation`,
#: which knows nothing about units or provenance and is tested purely
#: numerically.  This module is the provenance-aware wrapper around it, and
#: re-exports the primitive under its historical name.
rotation_perifocal_to_reference = rotation_perifocal_to_inertial


def orbital_radius(semimajor_axis: float, eccentricity: float, true_anomaly):
    """Kepler's first law: ``r = a(1-e^2) / (1 + e cos nu)`` (section 8.1)."""
    nu = np.asarray(true_anomaly, dtype=np.float64)
    return semimajor_axis * (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(nu))


def _display_angles(elements: "OrbitalElements") -> dict[str, float]:
    """Angles for the transform, defaulting unknown ones to zero.

    Callers are expected to have gone through
    :meth:`OrbitalElements.for_display` so that any substitution is already
    recorded as ASSUMED_FOR_VISUALIZATION.  The zero defaults here are the
    last line of that policy, not a shortcut around it.
    """
    return {
        "inclination": elements.inclination.value_in(u.rad, 0.0),
        "argument_of_periapsis": elements.argument_of_periastron.value_in(u.rad, 0.0),
        "longitude_of_ascending_node": elements.longitude_of_ascending_node.value_in(u.rad, 0.0),
    }


def position_at_eccentric_anomaly(
    elements: "OrbitalElements", eccentric_anomaly
) -> np.ndarray:
    """Reference-frame position in AU for a given eccentric anomaly.

    Requires a known semimajor axis; call :meth:`OrbitalElements.for_display`
    first if you want unknown angles normalised to zero.
    """
    return position_from_eccentric_anomaly(
        elements.semimajor_axis.require(u.au),
        elements.eccentricity.value_in(u.dimensionless_unscaled, 0.0),
        eccentric_anomaly,
        **_display_angles(elements),
    )


def position_at_mean_anomaly(elements: "OrbitalElements", mean_anomaly) -> np.ndarray:
    """Reference-frame position in AU for a given mean anomaly."""
    e = elements.eccentricity.value_in(u.dimensionless_unscaled, 0.0)
    return position_at_eccentric_anomaly(elements, solve_kepler(mean_anomaly, e))


def state_at_mean_anomaly(elements: "OrbitalElements", mean_anomaly, *, mu=None):
    """Position and, when ``mu`` is given, velocity for a mean anomaly.

    ``mu`` comes from
    :func:`astro_explorer.physics.state_vectors.gravitational_parameter`,
    which returns ``None`` when the stellar mass is unpublished - in which
    case the returned state simply has no velocity.
    """
    from .state_vectors import state_at_mean_anomaly as _state

    return _state(
        elements.semimajor_axis.require(u.au),
        elements.eccentricity.value_in(u.dimensionless_unscaled, 0.0),
        mean_anomaly,
        mu=mu,
        **_display_angles(elements),
    )


def true_anomaly_at_mean_anomaly(elements: OrbitalElements, mean_anomaly):
    """True anomaly for a mean anomaly, using the real Kepler solver."""
    e = elements.eccentricity.value_in(u.dimensionless_unscaled, 0.0)
    return true_anomaly_from_eccentric(solve_kepler(mean_anomaly, e), e)


__all__ += [
    "position_at_eccentric_anomaly",
    "position_at_mean_anomaly",
    "state_at_mean_anomaly",
    "true_anomaly_at_mean_anomaly",
]
