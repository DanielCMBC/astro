"""Explorer C3.5: the SystemFrame -> ICRS tangent basis.

C3 refused to publish a planet's absolute celestial position because the
host's vector is ICRS, the planet's local vector is in the system frame,
and adding them adds components measured along different axes. This module
supplies the missing rotation.

It is almost entirely a module about **conventions**. The arithmetic is
four lines. Everything else here exists because a basis that is
mathematically self-consistent can still encode a *different* convention
from the catalogue, and the result looks entirely plausible in a render.

The canonical frame
-------------------

At a host with ICRS coordinates ``(alpha, delta)``:

.. math::
    \\hat e_r = (\\cos\\delta\\cos\\alpha,\\ \\cos\\delta\\sin\\alpha,\\ \\sin\\delta)

    \\hat e_{east} = (-\\sin\\alpha,\\ \\cos\\alpha,\\ 0)

    \\hat e_{north} = (-\\sin\\delta\\cos\\alpha,\\ -\\sin\\delta\\sin\\alpha,\\ \\cos\\delta)

and the canonical sky frame is

.. math::
    +X \\rightarrow \\hat e_{east}, \\quad
    +Y \\rightarrow \\hat e_{north}, \\quad
    +Z \\rightarrow \\hat e_{east}\\times\\hat e_{north} = \\hat e_r

so ``+Z`` points **away from the observer**, and a body moving toward
``+Z`` is receding.

Why the line of sight points away, not toward
---------------------------------------------

This is the correction that matters, and it is not a matter of taste.

The standard astronomical definition of the *ascending* node is the
crossing where the body moves **away** from the observer. Under the
production transform ``R_z R_x(i) R_z(omega)``, a body just past the node
(argument of latitude ``u`` slightly positive) has ``z = sin(u) sin(i) > 0``
for any prograde or retrograde inclination - it moves toward ``+z``. For
that crossing to be the *ascending* one, ``+z`` must point away from the
observer.

An earlier draft of this module assigned ``+x = North, +y = East``, which is
self-consistent and makes ``R_z`` carry North toward East - but it forces
``+z = North x East = -e_r``, toward the observer. That would have made this
codebase's "ascending" node the *approaching* one: a silent 180-degree
disagreement with every catalogue it reads.

Position angle is not a mathematical azimuth
--------------------------------------------

Because ``+X`` is East and ``+Y`` is North, a rotation ``R_z(theta)`` carries
East toward North. The catalogued longitude of the ascending node is a
*position angle*, measured from North increasing toward East - the opposite
sense, from the other axis. The two are related by

.. math::
    \\theta = \\pi/2 - \\Omega_{PA}

and :func:`position_angle_to_azimuth` is the only place that conversion is
allowed to happen. Keeping the source convention and the internal Cartesian
convention separate - rather than feeding a catalogue angle straight into a
rotation matrix - is the whole point.

What still is not enough to publish a position
----------------------------------------------

A valid rotation is necessary and nowhere near sufficient. Three further
things gate publication, and in the current snapshot at least one always
blocks:

* the **node convention** must be stated. An angle whose convention is
  unrecorded is a number, not a direction;
* the **node sense** must be resolved. Relative astrometry commonly
  determines the node only modulo 180 degrees - ``(omega, Omega)`` and
  ``(omega + pi, Omega - pi)`` project identically, and only radial-velocity
  or equivalent line-of-sight information tells them apart;
* the **coordinate epoch** must be handled. C3.5 could only defer this,
  because :class:`SkyPosition` carries no obstime, proper motion or radial
  velocity: a host position was at its catalogue epoch while the planet
  offset was at the requested time, and for a nearby high-proper-motion
  star that mismatch is far larger than the AU-scale offset it would be
  added to. C3.6 supplies the missing state as
  :class:`~astro_explorer.coordinates.astrometry.PropagatedAstrometry`, so
  this gate is now satisfied by a propagated position at a stated instant
  rather than deferred by a boolean.

So this module ships the transform and asks for the astrometry; what it
still refuses to do is publish a position for an orbit whose node is a
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import astropy.units as u
import numpy as np

from ..physics.node_semantics import (
    NODE_CONVENTION_UNSTATED,
    NODE_SENSE_UNRESOLVED,
    NodeConvention,
    NodeSense,
    NodeSenseEvidence,
    azimuth_to_position_angle,
    node_convention_of,
    node_is_constrained,
    node_publication_blockers,
    node_sense_evidence_of,
    node_sense_of,
    position_angle_to_azimuth,
    resolve_node_azimuth,
)
from ..physics.orbital_semantics import absolute_orientation_blockers
from ..provenance import Parameter
from .astrometry import PropagatedAstrometry
from .frames import SkyPosition

__all__ = [
    "LINE_OF_SIGHT",
    "SKY_BASIS_CONVENTION",
    "POLE_TOLERANCE_DEG",
    "NodeConvention",
    "NodeSense",
    "NodeSenseEvidence",
    "TangentBasis",
    "tangent_basis",
    "position_angle_to_azimuth",
    "azimuth_to_position_angle",
    "system_to_icrs_rotation",
    "system_offset_to_icrs_pc",
    "is_pole_degenerate",
    "node_convention_of",
    "node_sense_of",
    "node_sense_evidence_of",
    "node_is_constrained",
    "resolve_node_azimuth",
    "absolute_position_blockers",
    "absolute_orientation_blockers",
    "node_publication_blockers",
    "ASTROMETRY_NOT_PROPAGATED",
    "NODE_CONVENTION_UNSTATED",
    "NODE_SENSE_UNRESOLVED",
    "POLE_DEGENERATE",
]

#: Which way ``+Z`` points, in words, because a sign error here mirrors
#: every orbit through the plane of the sky and looks entirely plausible.
LINE_OF_SIGHT = "observer -> star (a body moving toward +Z is receding)"

#: The full convention in one place, so a reader never has to reconstruct it
#: from the arithmetic.
SKY_BASIS_CONVENTION = (
    "Canonical sky frame: +X = East, +Y = North, +Z = +X cross +Y = away "
    "from the observer. Right-handed, determinant +1. The ascending node is "
    "the receding crossing, which is what fixes the sign of +Z. A catalogued "
    "longitude of ascending node is a position angle measured from North "
    "increasing toward East, and is converted to an internal azimuth by "
    "theta = pi/2 - Omega_PA."
)

#: Within this many degrees of a celestial pole, right ascension is not a
#: unique physical direction, so the East/North azimuth is gauge-dependent.
POLE_TOLERANCE_DEG = 1e-6


#: Reasons an absolute position cannot be published. Each is a separate
#: sentence so several can be reported together - a row blocked for three
#: reasons should say three, not pick one.
ASTROMETRY_NOT_PROPAGATED = (
    "no astrometric state was propagated to the requested time, so the "
    "host's position is at an unstated epoch and cannot be combined with a "
    "planet offset at that time"
)
#: ``NODE_CONVENTION_UNSTATED`` and ``NODE_SENSE_UNRESOLVED`` are re-exported
#: from :mod:`astro_explorer.physics.node_semantics`, which owns them: they
#: describe what a catalogue recorded about an angle, not anything about the
#: tangent basis. Existing importers of this module keep working.
POLE_DEGENERATE = (
    "the host is at a celestial pole, where right ascension is not a unique "
    "physical direction and the sky-plane azimuth is gauge-dependent"
)


@dataclass(frozen=True)
class TangentBasis:
    """The orthonormal triad at one host star, in ICRS components.

    ``radial`` points from the Sun outward to the star - which, in the
    canonical frame, is also ``+Z``: away from the observer.
    """

    east: np.ndarray
    north: np.ndarray
    radial: np.ndarray

    def matrix(self) -> np.ndarray:
        """The canonical-sky-frame -> ICRS rotation, ``(3, 3)``.

        Columns are the ICRS images of ``+X`` (East), ``+Y`` (North) and
        ``+Z`` (away from the observer), in that order.
        """
        return np.column_stack([self.east, self.north, self.radial])

    def is_orthonormal(self, tolerance: float = 1e-12) -> bool:
        matrix = self.matrix()
        return bool(
            np.allclose(matrix @ matrix.T, np.eye(3), atol=tolerance)
            and abs(np.linalg.det(matrix) - 1.0) < tolerance
        )


def tangent_basis(ra_deg: float, dec_deg: float) -> TangentBasis:
    """The local tangent triad at ``(ra, dec)``, in ICRS components."""
    alpha = np.deg2rad(float(ra_deg))
    delta = np.deg2rad(float(dec_deg))
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_d, sin_d = np.cos(delta), np.sin(delta)

    return TangentBasis(
        east=np.array([-sin_a, cos_a, 0.0], dtype=np.float64),
        north=np.array([-sin_d * cos_a, -sin_d * sin_a, cos_d], dtype=np.float64),
        radial=np.array([cos_d * cos_a, cos_d * sin_a, sin_d], dtype=np.float64),
    )


def is_pole_degenerate(dec_deg: float | None) -> bool:
    """True at a celestial pole, where sky-plane azimuth has no unique meaning.

    The closed-form triad stays algebraically orthonormal there, which is
    exactly the danger: it will happily return a basis whose azimuth is set
    by whatever right ascension happened to be recorded for an object that
    does not have a meaningful one.
    """
    if dec_deg is None:
        return False
    return abs(abs(float(dec_deg)) - 90.0) <= POLE_TOLERANCE_DEG


def system_to_icrs_rotation(host: SkyPosition | None) -> np.ndarray | None:
    """The canonical-frame -> ICRS rotation, or None without a direction.

    Only RA and Dec are needed. A star with an unusable parallax still has a
    perfectly good tangent basis - it is the *origin* that is unknown, not
    the orientation.
    """
    if host is None:
        return None
    ra = host.ra.value_in(u.deg)
    dec = host.dec.value_in(u.deg)
    if ra is None or dec is None:
        return None
    return tangent_basis(ra, dec).matrix()


def system_offset_to_icrs_pc(host: SkyPosition | None, offset_au) -> np.ndarray | None:
    """Rotate a canonical-frame offset in AU into ICRS components in parsecs.

    Both steps are explicit and both happen in float64. The unit conversion
    spans five orders of magnitude and the rotation is the piece C3 lacked;
    doing either implicitly is how an AU-scale offset becomes either zero or
    a number expressed in the wrong axes.
    """
    rotation = system_to_icrs_rotation(host)
    if rotation is None or offset_au is None:
        return None
    offset = np.asarray(offset_au, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        return None
    return rotation @ (offset * float((1.0 * u.au).to_value(u.pc)))


def absolute_position_blockers(
    host: SkyPosition | None,
    node: Parameter | None = None,
    *,
    astrometry: "PropagatedAstrometry | None" = None,
    elements=None,
) -> list[str]:
    """Every reason an absolute planet position may not be published.

    Returns them all rather than the first, because a row blocked for three
    reasons should say three. An empty list means every scientific gate is
    satisfied.

    ``astrometry`` is what C3.6 puts where C3.5 had ``epoch_resolved:
    bool``. The difference is not cosmetic. A boolean was an *assertion*
    that the epoch had been handled, and the only thing standing between a
    correct program and a wrong one was that nobody wrote ``True``. A
    :class:`~astro_explorer.coordinates.astrometry.PropagatedAstrometry` is
    the handling itself: it carries the instant, the propagated position and
    the list of things it could not do. There is no value of this argument
    that opens the gate without a real propagation behind it, and the
    reasons a propagation fell short travel through into this list rather
    than being collapsed into one word.

    ``elements`` is the C3.6 second-audit correction. The node gates are
    necessary and **not sufficient**: a unique physical orientation also
    needs the inclination and the planet-frame argument of periapsis, and
    an orbit whose plane was drawn face-on for display could otherwise reach
    a published ICRS coordinate on the strength of a tagged node alone. When
    an element set is supplied its node is authoritative and the full
    orientation gate runs; passing only ``node`` keeps the narrower check,
    which is enough for the callers that are asking about the node itself
    and can never publish, because publication additionally requires a dated
    orbital state that carries its elements.

    The pole check uses the **propagated** declination when there is one.
    A star can cross the pole tolerance between its reference epoch and the
    requested date, and the position being published is the propagated one.
    """
    reasons: list[str] = []

    if elements is not None:
        reasons.extend(absolute_orientation_blockers(elements))
    else:
        reasons.extend(node_publication_blockers(node))

    if astrometry is None:
        reasons.append(ASTROMETRY_NOT_PROPAGATED)
    else:
        reasons.extend(r for r in astrometry.blockers if r not in reasons)
        if not astrometry.is_publishable and not astrometry.blockers:
            reasons.append(ASTROMETRY_NOT_PROPAGATED)
        host = astrometry.position

    if host is not None and is_pole_degenerate(host.dec.value_in(u.deg)):
        reasons.append(POLE_DEGENERATE)

    return reasons
