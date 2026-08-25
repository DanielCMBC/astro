"""Explorer C3.6: an orbital state that knows *when* it is.

C3.6 made the host's astrometry carry its instant, as
:class:`~astro_explorer.coordinates.astrometry.PropagatedAstrometry`. That
closed one half of the common-time rule and left the other half open, in a
way that is easy to miss because the arithmetic never complains:

.. code-block:: python

    row = absolute_planet_position(host, position_au, astrometry=propagated)

``position_au`` is a bare ``(3,)`` array of astronomical units. It has no
epoch. Nothing in it can disagree with ``propagated.obstime``, so a host
propagated to 2035 and a planet propagated to 2025 add together perfectly
and produce a coordinate that is wrong by ten years of orbital phase - a
number which, for a short-period planet, is simply "somewhere else on the
ellipse".

The offset being added is the *small* term. Getting its epoch wrong does not
make the answer slightly worse; it makes it an answer about a different
night.

So the offset stops being a bare array. :class:`TimedOrbitalState` is the
propagator's float64 state with the instant it was evaluated at and the
phase provenance that says how much that instant is worth, and the
publication path takes one of these rather than an array. A caller can still
hand in a raw vector for a display realisation - drawing does not need a
common epoch - but it cannot open a scientific gate with one, because the
thing the gate checks is not present on an array.

Which clock
-----------

``obstime`` is an :class:`~astropy.time.Time`: a physical instant, not a
Julian day number. The orbital propagator runs on the project's canonical
full-JD axis instead, so exactly one function converts between them -
:func:`~astro_explorer.physics.epoch.orbital_time_jd` - and the scale the
element set's epoch was published in is what it converts through. Two states
built by that route are comparable as instants even when one came from a
``BJD_TDB`` ephemeris and the other from an unlabelled archive date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ..provenance import Status
from .epoch import INSTANT_MATCH_TOLERANCE_DAYS, TimeScale, orbital_time_jd
from .orbital_semantics import absolute_orientation_blockers
from .phase import PhaseSolution
from .state_vectors import StateVector

if TYPE_CHECKING:  # pragma: no cover - a type-only import
    from .orbital_elements import OrbitalElements

__all__ = ["TimedOrbitalState", "same_instant"]


def same_instant(a, b, tolerance_days: float = INSTANT_MATCH_TOLERANCE_DAYS) -> bool:
    """True when two dated objects were evaluated at the same instant.

    Takes anything carrying an :class:`~astropy.time.Time` ``obstime`` - a
    :class:`TimedOrbitalState`, a
    :class:`~astro_explorer.coordinates.astrometry.PropagatedAstrometry`, or
    a bare ``Time`` - so the host, the target star and the planet can all be
    compared against each other by one rule.

    The comparison is between **instants**, never between Julian day
    numbers or formatted dates. Two objects reached through different time
    scales represent the same instant at Julian dates differing by up to
    about 89 seconds, and a comparison on the numbers would reject that pair
    while accepting a genuinely mismatched one that happened to share a
    number.
    """
    first = getattr(a, "obstime", a)
    second = getattr(b, "obstime", b)
    if not isinstance(first, Time) or not isinstance(second, Time):
        return False
    return abs(float((first - second).to_value("day"))) <= tolerance_days


@dataclass(frozen=True)
class TimedOrbitalState:
    """A propagated orbital state, the instant it holds at, and its phase.

    The three travel together because none of them means much alone:

    ``state``
        the float64 position (AU) and, when the stellar mass is published,
        velocity (AU/day) in the system frame. This is the scientific
        output; the renderer's copy of it is not.

    ``obstime``
        the physical instant. Without it the vector is a picture of the
        orbit rather than a claim about a date, and it can be silently
        combined with a host position from another decade.

    ``phase``
        how the instant was reached - a published periastron epoch, a
        transit epoch read through an assumed argument of periastron, or an
        arbitrary zero advanced at the right rate. A vector whose phase is
        ``ASSUMED_ZERO_PHASE`` is at a defensible *place* on the ellipse and
        at no particular *time*, so it must not be published as an absolute
        position however carefully its epoch was tracked.

    ``elements``
        the orbit this is a state *of*. Required, not optional: a state
        vector alone cannot be asked how well its orientation is known, and
        that question decides whether an absolute celestial position may be
        published at all. Phase and orientation stay separate epistemic
        dimensions - a transit epoch is a real observation of *when* and
        says nothing about *which way* - so both live here and are gated
        side by side rather than folded into one status.
    """

    state: StateVector
    obstime: Time
    phase: PhaseSolution
    elements: "OrbitalElements"
    time_scale: TimeScale = TimeScale.JD_UNSPECIFIED

    @property
    def position_au(self) -> np.ndarray:
        """The star-centred float64 position in AU."""
        return np.asarray(self.state.position, dtype=np.float64)

    @property
    def obstime_jd(self) -> float:
        """The instant on the orbital clock's own axis.

        Goes through :func:`~astro_explorer.physics.epoch.orbital_time_jd`
        rather than ``obstime.jd``, so the number is on the same axis the
        published epoch is quoted on.
        """
        return orbital_time_jd(self.obstime, self.time_scale)

    @property
    def is_time_constrained(self) -> bool:
        """True when the instant is an observation rather than a placeholder.

        An assumed phase advances at the correct rate from an arbitrary
        zero. Tracking its epoch perfectly does not make it a position on a
        date, so this is a separate question from whether the times match.

        The test is
        :attr:`~astro_explorer.physics.phase.PhaseStatus.is_observationally_anchored`
        and deliberately not ``not is_assumed``. A transit-epoch phase read
        through a normalised argument of periastron is
        ``PARTIALLY_CONSTRAINED``: its *timing* is a real observation and
        only its in-plane orientation was normalised. That orientation is
        already gated, twice, by the node convention and node sense checks -
        so rejecting it here as well would refuse a published epoch for a
        reason that has nothing to do with what this property is about.
        """
        return (
            self.phase.is_placeable
            and self.phase.status.is_observationally_anchored
        )

    @property
    def orientation_blockers(self) -> tuple[str, ...]:
        """Every reason this orbit's orientation cannot fix a 3D position.

        Empty only when inclination, the planet-frame argument of periapsis
        and the node are all observations under stated conventions. See
        :func:`~astro_explorer.physics.orbital_semantics.absolute_orientation_blockers`.
        """
        return absolute_orientation_blockers(self.elements)

    @property
    def is_orientation_resolved(self) -> bool:
        return not self.orientation_blockers

    def at_same_time_as(self, other) -> bool:
        """True when ``other`` holds at this state's instant."""
        return same_instant(self, other)

    def describe(self) -> list[str]:
        lines = [
            "Orbital state at:   JD {0:.6f} ({1})".format(
                self.obstime_jd, self.time_scale.label
            ),
            "Phase provenance:   {0}".format(self.phase.status.label),
        ]
        for reason in self.orientation_blockers:
            lines.append("Orientation:        {0}".format(reason))
        return lines

    @property
    def status(self) -> Status:
        """The status an absolute position built from this may claim."""
        if not self.phase.is_placeable:
            return Status.UNKNOWN
        return (
            Status.DERIVED
            if self.phase.status.is_observationally_anchored
            else Status.ASSUMED_FOR_VISUALIZATION
        )
