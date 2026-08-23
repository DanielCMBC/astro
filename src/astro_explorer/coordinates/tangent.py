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
* the **coordinate epoch** must be handled. :class:`SkyPosition` carries no
  obstime, proper motion or radial velocity, so a host position is at its
  catalogue epoch while the planet offset is at the requested time. For a
  nearby high-proper-motion star that mismatch is far larger than the
  AU-scale offset it would be added to.

So this slice ships the transform and keeps the absolute position withheld.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status
from .frames import SkyPosition

__all__ = [
    "LINE_OF_SIGHT",
    "SKY_BASIS_CONVENTION",
    "POLE_TOLERANCE_DEG",
    "NodeConvention",
    "NodeSense",
    "TangentBasis",
    "tangent_basis",
    "position_angle_to_azimuth",
    "azimuth_to_position_angle",
    "system_to_icrs_rotation",
    "system_offset_to_icrs_pc",
    "is_pole_degenerate",
    "node_convention_of",
    "node_sense_of",
    "absolute_position_blockers",
    "EPOCH_NOT_MODELLED",
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


class NodeConvention(str, Enum):
    """How a catalogued longitude of ascending node was defined.

    ``UNSPECIFIED`` is the default and is never treated as a guess at the
    standard: an angle whose convention was not recorded is a number, not a
    direction, and must not unlock an absolute position.
    """

    PA_EAST_OF_NORTH_RECEDING = "PA_EAST_OF_NORTH_RECEDING"
    """Position angle from North toward East; ascending = receding."""

    UNSPECIFIED = "UNSPECIFIED"
    """Not recorded. Blocks publication rather than defaulting."""

    @property
    def is_stated(self) -> bool:
        return self is not NodeConvention.UNSPECIFIED


class NodeSense(str, Enum):
    """Whether the ascending/descending sense of the node is resolved.

    A measured *number* is not a resolved node. Relative astrometry
    routinely determines the node only modulo 180 degrees, because
    ``(omega, Omega)`` and ``(omega + pi, Omega - pi)`` produce the same
    projected orbit on the sky. Distinguishing them needs radial-velocity or
    equivalent line-of-sight information.

    ``RESOLVED`` means evidence identified *this orbit's* receding node - a
    radial-velocity orbit, an eclipse timing that fixes which node recedes,
    or an equivalent observation. A generic **systemic** stellar radial
    velocity is **not** enough by itself: it describes the whole system's
    motion relative to the Sun and says nothing about which of the two nodes
    of a planet's orbit recedes. Because that distinction is invisible in
    the number, :func:`node_sense_of` requires the evidence to be named.
    """

    RESOLVED = "RESOLVED"
    """The receding node is identified; the 3D orientation is unique.

    Only for evidence that identifies *this orbit's* ascending node -
    orbit-specific line-of-sight information such as a radial-velocity
    orbit, a transit/eclipse timing that fixes which node recedes, or an
    equivalent observation.

    A generic systemic stellar radial velocity is **not** enough by itself.
    It describes the motion of the whole system relative to the Sun and says
    nothing about which of the two nodes of a planet's orbit is the receding
    one. Because that distinction is invisible in the number itself, the
    evidence has to be named: see :func:`node_sense_of`.
    """

    MODULO_180 = "MODULO_180"
    """The projected orbit is known; which node is ascending is not."""

    UNKNOWN = "UNKNOWN"
    """No information about the sense at all."""

    @property
    def is_resolved(self) -> bool:
        return self is NodeSense.RESOLVED


#: Reasons an absolute position cannot be published. Each is a separate
#: sentence so several can be reported together - a row blocked for three
#: reasons should say three, not pick one.
EPOCH_NOT_MODELLED = (
    "the coordinate epoch is not modelled - the host has no obstime, proper "
    "motion or radial velocity, so its catalogue-epoch position cannot be "
    "combined with a planet offset at the requested time"
)
NODE_CONVENTION_UNSTATED = (
    "the node convention was not recorded, so the catalogued angle does not "
    "identify a direction on the sky"
)
NODE_SENSE_UNRESOLVED = (
    "the ascending/descending sense of the node is not resolved, so the "
    "orientation is known only modulo 180 degrees"
)
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


def position_angle_to_azimuth(position_angle_rad: float) -> float:
    """Catalogue position angle -> internal azimuth: ``theta = pi/2 - PA``.

    The single place this conversion is permitted. A position angle runs
    from North toward East; the internal azimuth runs from ``+X`` (East)
    toward ``+Y`` (North). They increase in opposite senses from different
    axes, which is precisely why feeding a catalogue angle straight into a
    rotation matrix is wrong.
    """
    return float(np.pi / 2.0 - float(position_angle_rad))


def azimuth_to_position_angle(azimuth_rad: float) -> float:
    """Inverse of :func:`position_angle_to_azimuth`; it is its own inverse."""
    return float(np.pi / 2.0 - float(azimuth_rad))


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


def node_convention_of(node: Parameter | None) -> NodeConvention:
    """The recorded convention for a node parameter, or UNSPECIFIED.

    Read from ``Parameter.extra`` so an ingestion path can tag it without
    every consumer having to know. Absence is never treated as the standard:
    silence must not unlock anything.
    """
    if node is None:
        return NodeConvention.UNSPECIFIED
    recorded = node.extra.get("node_convention")
    try:
        return NodeConvention(recorded)
    except ValueError:
        return NodeConvention.UNSPECIFIED


def node_sense_of(node: Parameter | None) -> NodeSense:
    """The recorded ascending/descending sense, or UNKNOWN.

    A node with a value but no recorded sense is ``MODULO_180``, not
    ``RESOLVED``: having a number is the normal situation in which the
    ambiguity exists.

    ``RESOLVED`` additionally requires ``node_sense_evidence`` to name what
    actually broke the tie. That is not bureaucracy: the claim being made is
    that some observation identified *this orbit's* receding node, and a
    generic systemic radial velocity - the thing most likely to be reached
    for - cannot do that. An unevidenced ``RESOLVED`` tag therefore falls
    back to ``MODULO_180`` rather than unlocking publication.
    """
    if node is None or not node.is_known:
        return NodeSense.UNKNOWN
    try:
        recorded = NodeSense(node.extra.get("node_sense"))
    except ValueError:
        return NodeSense.MODULO_180
    if recorded is NodeSense.RESOLVED:
        evidence = node.extra.get("node_sense_evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            return NodeSense.MODULO_180
    return recorded


def absolute_position_blockers(
    host: SkyPosition | None,
    node: Parameter | None,
    *,
    epoch_resolved: bool = False,
) -> list[str]:
    """Every reason an absolute planet position may not be published.

    Returns them all rather than the first, because a row blocked for three
    reasons should say three. An empty list means every scientific gate is
    satisfied - which, with :class:`SkyPosition` carrying no epoch or space
    motion, does not currently happen for any real object.
    """
    reasons: list[str] = []

    if node is None or not node.is_known or node.status not in (
        Status.MEASURED,
        Status.DERIVED,
    ):
        reasons.append(NODE_SENSE_UNRESOLVED)
    else:
        if not node_convention_of(node).is_stated:
            reasons.append(NODE_CONVENTION_UNSTATED)
        if not node_sense_of(node).is_resolved:
            reasons.append(NODE_SENSE_UNRESOLVED)

    if host is not None and is_pole_degenerate(host.dec.value_in(u.deg)):
        reasons.append(POLE_DEGENERATE)

    if not epoch_resolved:
        reasons.append(EPOCH_NOT_MODELLED)

    return reasons
