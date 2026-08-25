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
from .node_semantics import node_publication_blockers

__all__ = [
    "PeriastronConvention",
    "OrbitValidity",
    "stellar_reflex_to_planet",
    "planet_to_stellar_reflex",
    "angular_difference",
    "resolve_argument_of_periapsis",
    "absolute_orientation_blockers",
    "INCLINATION_UNRESOLVED",
    "PERIAPSIS_DIRECTION_UNRESOLVED",
    "PERIASTRON_CONVENTION_UNSTATED",
    "CIRCULAR_ORBIT_PERIAPSIS_NOTE",
]


# ---------------------------------------------------------------------------
# The absolute-orientation publication gate (Explorer C3.6)
# ---------------------------------------------------------------------------

INCLINATION_UNRESOLVED = (
    "the inclination is not a published measurement, so the orbital plane is "
    "normalised for display and its tilt in space is not known"
)
PERIAPSIS_DIRECTION_UNRESOLVED = (
    "the argument of periapsis is not published, so the orbit's orientation "
    "within its own plane is normalised for display"
)
PERIASTRON_CONVENTION_UNSTATED = (
    "the argument of periastron was published under an unstated convention, "
    "so periapsis may be the star's reflex direction and 180 degrees from "
    "the planet's"
)

#: Why a scientifically circular orbit is still blocked on periapsis.
#:
#: For ``e = 0`` there is no periapsis, so ``omega`` is not a physical
#: degree of freedom and demanding it looks like an over-refusal. It is a
#: deliberate one, and the reason is that the exemption is not a property of
#: the *orientation* alone:
#:
#: * the position depends on the argument of latitude ``u = omega + nu``;
#: * for a transit or conjunction anchor, ``nu_t = pi/2 - omega`` at the
#:   anchor, so ``u(t) = pi/2 + n(t - t_t)`` and ``omega`` cancels exactly.
#:   Such an orbit genuinely does not need it;
#: * for a periastron anchor, or a mean anomaly quoted at an epoch, the
#:   anchor is measured *from periapsis* - which does not exist at ``e = 0``,
#:   so the epoch itself has no meaning and nothing is recovered.
#:
#: So the correct rule needs the phase anchor kind, and phase and orientation
#: are kept as separate epistemic dimensions in this codebase on purpose.
#: Rather than fold one into the other for a case that no catalogue row in
#: the snapshot can currently reach - every real exoplanet is already blocked
#: on the node - this stays conservative and the analysis stays written down.
CIRCULAR_ORBIT_PERIAPSIS_NOTE = (
    "a circular orbit has no periapsis direction to publish; lifting this "
    "blocker needs the phase anchor as well, because omega cancels for a "
    "transit anchor and the epoch is undefined for a periastron one"
)


def absolute_orientation_blockers(elements) -> tuple[str, ...]:
    """Every reason an orbit's orientation cannot fix an absolute position.

    Explorer C3.6. The node gates alone are necessary and **not
    sufficient**, which is the hole this closes: a planet whose inclination
    and argument of periapsis were normalised for display could reach a
    published ICRS coordinate as long as its node happened to be tagged, and
    the number looked entirely ordinary.

    A unique physical orientation needs all three Euler angles to be
    observations:

    ``inclination``
        the tilt of the orbital plane. A display normalisation draws the
        orbit face-on, which is a picture and not a plane in space.

    ``argument of periapsis``
        where periapsis points *within* that plane, and under a **stated
        convention**: radial-velocity papers habitually report the star's
        reflex orbit and transit papers the planet's, so an unstated one
        leaves periapsis ambiguous by 180 degrees - which for an eccentric
        orbit puts the planet on the wrong side of its star.

    ``longitude of the ascending node``
        the rotation about the line of sight, with its convention stated and
        its sense resolved. Delegated to
        :func:`~astro_explorer.physics.node_semantics.node_publication_blockers`.

    Every reason is returned rather than the first, because a row blocked
    for four reasons should say four.

    This is deliberately **not** merged into
    :class:`~astro_explorer.physics.phase.PhaseStatus`. Phase and
    orientation are separate epistemic dimensions here: a transit epoch is a
    real observation of *when*, and it says nothing about *which way*. The
    two gates are checked side by side and reported side by side.
    """
    reasons: list[str] = []

    if not elements.inclination.is_scientific:
        reasons.append(INCLINATION_UNRESOLVED)

    # The resolved planet-frame value already encodes the convention rule -
    # MEASURED under PLANET, DERIVED after a reflex conversion, ASSUMED under
    # AS_REPORTED - so this asks it rather than restating the table.
    omega = elements.argument_of_periastron
    resolved = resolve_argument_of_periapsis(omega, elements.periastron_convention)
    if not resolved.is_scientific:
        if omega.is_scientific and elements.periastron_convention is (
            PeriastronConvention.AS_REPORTED
        ):
            # A real published number under a convention nobody stated.
            # Naming the convention is the useful message here; adding "the
            # direction is unresolved" underneath would say the same thing
            # twice about one root cause.
            reasons.append(PERIASTRON_CONVENTION_UNSTATED)
        else:
            # Never published, or filled in by ``for_display`` so a scene
            # could be drawn. Both are "nobody measured this", and a display
            # normalisation must not be reported as a convention problem -
            # it would suggest the number exists and only its meaning is
            # missing.
            reasons.append(PERIAPSIS_DIRECTION_UNRESOLVED)

    reasons.extend(
        node_publication_blockers(elements.longitude_of_ascending_node)
    )
    return tuple(reasons)


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
