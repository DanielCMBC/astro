"""What a catalogued orbital element actually *means*.

Review sections 9, 10 and 11. The numerics are settled; what remains is
semantics, and the sharpest case is the argument of periastron.

The NASA Exoplanet Archive's ``pl_orblper`` generally preserves the
convention of the source publication. Radial-velocity papers habitually
report the argument of periastron of the **star's reflex orbit**, while
transit and astrometry papers report the **planet's**. The two differ by
exactly 180 degrees:

.. math:: \\omega_{\\rm planet} = (\\omega_{\\rm star} + \\pi) \\bmod 2\\pi

So a transform that is numerically perfect can still put periapsis on the
wrong side of the star. The fix is not arithmetic: it is refusing to claim
a convention the catalogue never stated.

This module therefore keeps the raw value untouched and records, separately,
which convention it is believed to follow. Converting between them is
explicit and tested; assuming one is tagged
``ASSUMED_FOR_VISUALIZATION`` like any other assumption.
"""

from __future__ import annotations

from enum import Enum, Flag, auto

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status, assumed, derived, unknown

__all__ = [
    "PeriastronConvention",
    "OrbitValidity",
    "stellar_reflex_to_planet",
    "planet_to_stellar_reflex",
    "angular_difference",
    "resolve_argument_of_periapsis",
]


class PeriastronConvention(str, Enum):
    """Whose orbit an argument of periastron describes (review section 10)."""

    PLANET = "PLANET"
    """Explicitly the planet's orbit. Usable directly."""

    STELLAR_REFLEX = "STELLAR_REFLEX"
    """The host star's reflex orbit. Needs +180 degrees to become the planet's."""

    AS_REPORTED = "AS_REPORTED"
    """Taken from the catalogue with the convention unstated.

    This is the honest default for the NASA archive: the value is real, but
    which body it refers to depends on the source publication and the
    archive does not carry that distinction in a machine-readable column.
    """

    UNKNOWN = "UNKNOWN"
    """No argument of periastron at all."""

    @property
    def is_determinate(self) -> bool:
        """True when the convention is actually known."""
        return self in (PeriastronConvention.PLANET, PeriastronConvention.STELLAR_REFLEX)

    @property
    def label(self) -> str:
        return {
            PeriastronConvention.PLANET: "planet's orbit",
            PeriastronConvention.STELLAR_REFLEX: "host star's reflex orbit",
            PeriastronConvention.AS_REPORTED: "as reported; convention not stated",
            PeriastronConvention.UNKNOWN: "not available",
        }[self]

    @property
    def caveat(self) -> str:
        if self is PeriastronConvention.AS_REPORTED:
            return (
                "The source convention is unstated. If the publication reported the "
                "stellar reflex orbit, periapsis is oriented 180 degrees away from "
                "the truth."
            )
        return ""


class OrbitValidity(Flag):
    """What the published elements actually support (review section 11).

    A flag set rather than a single state, because the four questions are
    independent: an orbit can have a fully determined shape, no usable
    phase, and a partly known orientation all at once - which is the normal
    condition for an exoplanet.
    """

    NONE = 0

    GEOMETRY_VALID = auto()
    """``a`` and ``e`` are available: the shape of the orbit can be drawn."""

    PHASE_VALID = auto()
    """An epoch and a period exist: the planet can be placed at time t."""

    ORIENTATION_PARTIAL = auto()
    """Some 3D orientation elements are known; at least one is not."""

    ORIENTATION_FULL = auto()
    """i, omega and Omega are all known under a stated convention."""

    def describe(self) -> list[str]:
        lines = []
        for flag, text in (
            (OrbitValidity.GEOMETRY_VALID, "orbit shape can be drawn"),
            (OrbitValidity.PHASE_VALID, "planet can be placed at a given time"),
            (OrbitValidity.ORIENTATION_PARTIAL, "3D orientation is partly constrained"),
            (OrbitValidity.ORIENTATION_FULL, "3D orientation is fully constrained"),
        ):
            if flag in self:
                lines.append("{0}: {1}".format(flag.name, text))
        return lines or ["NONE: nothing about this orbit is usable"]

    @property
    def names(self) -> list[str]:
        return [flag.name for flag in OrbitValidity if flag is not OrbitValidity.NONE and flag in self]


def _wrap_two_pi(angle):
    """Wrap into [0, 2 pi)."""
    return np.mod(np.asarray(angle, dtype=np.float64), 2.0 * np.pi)


def angular_difference(first, second):
    """Smallest absolute angle between two directions, in [0, pi].

    Used by the regression test of review section 13, and by anything that
    needs to compare two angles without tripping over the wrap point.
    """
    delta = np.mod(
        np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64) + np.pi,
        2.0 * np.pi,
    ) - np.pi
    return np.abs(delta)


def stellar_reflex_to_planet(omega_star):
    """``omega_planet = (omega_star + pi) mod 2 pi`` (review section 10).

    Accepts and returns radians, or a :class:`~astro_explorer.provenance.Parameter`
    in any angular unit, in which case a DERIVED parameter comes back.
    """
    if isinstance(omega_star, Parameter):
        value = omega_star.value_in(u.rad)
        if value is None:
            return unknown(u.rad, provenance="stellar_reflex_to_planet")
        return derived(
            float(_wrap_two_pi(value + np.pi)),
            u.rad,
            error_plus=omega_star.error_plus,
            error_minus=omega_star.error_minus,
            provenance="stellar_reflex_to_planet({0})".format(
                omega_star.provenance or "omega_star"
            ),
            note="converted from the host star's reflex orbit by +180 degrees",
        )
    return float(_wrap_two_pi(np.asarray(omega_star, dtype=np.float64) + np.pi))


def planet_to_stellar_reflex(omega_planet):
    """The exact inverse of :func:`stellar_reflex_to_planet`."""
    if isinstance(omega_planet, Parameter):
        value = omega_planet.value_in(u.rad)
        if value is None:
            return unknown(u.rad, provenance="planet_to_stellar_reflex")
        return derived(
            float(_wrap_two_pi(value + np.pi)),
            u.rad,
            error_plus=omega_planet.error_plus,
            error_minus=omega_planet.error_minus,
            provenance="planet_to_stellar_reflex({0})".format(
                omega_planet.provenance or "omega_planet"
            ),
            note="converted to the host star's reflex orbit by +180 degrees",
        )
    return float(_wrap_two_pi(np.asarray(omega_planet, dtype=np.float64) + np.pi))


def resolve_argument_of_periapsis(
    raw: Parameter,
    convention: PeriastronConvention,
) -> Parameter:
    """The planet-frame argument of periapsis implied by ``raw``.

    The raw catalogue value is never modified. What comes back is a new
    parameter whose status records how much was actually known:

    ==========================  ==========================================
    convention                  result
    ==========================  ==========================================
    ``PLANET``                  the raw value, still MEASURED
    ``STELLAR_REFLEX``          raw + 180 degrees, DERIVED
    ``AS_REPORTED``             the raw value, ASSUMED_FOR_VISUALIZATION
    ``UNKNOWN``                 UNKNOWN
    ==========================  ==========================================

    The ``AS_REPORTED`` row is the important one. Using the value as if it
    were the planet's convention is a *choice*, and it is tagged as one, so
    an orbit drawn from it cannot be mistaken for a measured orientation.
    """
    if not raw.is_known or convention is PeriastronConvention.UNKNOWN:
        return unknown(
            u.rad,
            provenance=raw.provenance or "pl_orblper",
            note="no argument of periastron published",
        )

    if convention is PeriastronConvention.PLANET:
        return raw.to(u.rad)

    if convention is PeriastronConvention.STELLAR_REFLEX:
        return stellar_reflex_to_planet(raw)

    # AS_REPORTED: the number is real, the convention is a guess.
    return assumed(
        float(_wrap_two_pi(raw.value_in(u.rad))),
        u.rad,
        provenance=raw.provenance or "pl_orblper",
        note=(
            "catalogue value used as the planet's argument of periapsis; the "
            "source convention is unstated and may be the stellar reflex orbit, "
            "which would rotate periapsis by 180 degrees"
        ),
    )
