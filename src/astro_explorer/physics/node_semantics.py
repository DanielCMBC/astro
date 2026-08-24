"""Explorer C3.5.1: what a catalogued ascending node actually means.

C3.5 established the canonical sky frame - ``+X`` East, ``+Y`` North,
``+Z`` away from the observer - and supplied
:func:`position_angle_to_azimuth` to convert a catalogue angle into an
internal one. What it did *not* do was put that conversion on the
production path.

That gap is the reason this module exists at the physics level rather than
in ``coordinates``. Three separate places feed a node angle into
``R_z(Omega) R_x(i) R_z(omega)``:

* :func:`~astro_explorer.physics.orbital_elements.position_at_eccentric_anomaly`
  and its siblings, through ``_display_angles``;
* the vertical slice's propagated state;
* the C2 orientation guides.

If the conversion lives anywhere those three cannot reach, one of them will
eventually pass a raw position angle straight into a rotation matrix, and
the resulting orbit will be ninety degrees out and running backwards - while
looking entirely reasonable.

Position angle is not a mathematical azimuth
--------------------------------------------

A catalogued longitude of the ascending node is a **position angle**: it
starts at North and increases toward East. The internal azimuth starts at
``+X`` - which is East - and increases toward ``+Y``, which is North. The
two run from different axes in opposite senses, so:

.. math::
    \\theta = \\pi/2 - \\Omega_{PA}

This matters even for the normalisation. Display code that reports
``Omega = 0 deg`` is saying *North*, and North is :math:`\\theta = \\pi/2`
internally, not zero. Passing the zero through unconverted would draw the
line of nodes East while the text beside it said North.

Evidence, not adjectives
------------------------

:class:`NodeSense` decides whether an orbit's 3D orientation is unique, and
``RESOLVED`` is a claim about an observation. An earlier version accepted
any non-empty evidence string, which meant the literal text
``"systemic radial velocity"`` - the one thing that specifically cannot
resolve a node - would have unlocked it.

:class:`NodeSenseEvidence` makes the kinds a closed set, with
``SYSTEMIC_RADIAL_VELOCITY`` present and explicitly non-resolving, so the
insufficient case is named and rejected rather than merely undocumented.
"""

from __future__ import annotations

from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status

__all__ = [
    "NodeConvention",
    "NodeSense",
    "NodeSenseEvidence",
    "position_angle_to_azimuth",
    "azimuth_to_position_angle",
    "node_convention_of",
    "node_sense_of",
    "node_sense_evidence_of",
    "resolve_node_azimuth",
    "NODE_KEY_CONVENTION",
    "NODE_KEY_SENSE",
    "NODE_KEY_EVIDENCE",
    "NODE_KEY_EVIDENCE_NOTE",
]

#: Keys an ingestion path writes into ``Parameter.extra``.
NODE_KEY_CONVENTION = "node_convention"
NODE_KEY_SENSE = "node_sense"
NODE_KEY_EVIDENCE = "node_sense_evidence"
NODE_KEY_EVIDENCE_NOTE = "node_sense_evidence_note"


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


class NodeSenseEvidence(str, Enum):
    """What kind of observation is claimed to have resolved the node.

    A closed set on purpose. The distinction being claimed - which of an
    orbit's two nodes is the receding one - is invisible in the angle
    itself, so the evidence has to be named, and naming it as free text
    means a plausible-sounding string can unlock the gate.

    ``SYSTEMIC_RADIAL_VELOCITY`` is listed precisely so it can be rejected:
    it is the obvious thing to reach for and it describes the whole system's
    motion relative to the Sun, saying nothing about which node of a
    planet's orbit recedes.
    """

    ORBITAL_RV_SOLUTION = "ORBITAL_RV_SOLUTION"
    """A radial-velocity orbit for this companion."""

    DIRECT_COMPANION_LOS_VELOCITY = "DIRECT_COMPANION_LOS_VELOCITY"
    """A measured line-of-sight velocity of the companion itself."""

    ASTROMETRY_PLUS_ORBITAL_RV = "ASTROMETRY_PLUS_ORBITAL_RV"
    """Relative astrometry with the RV orbit that breaks its degeneracy."""

    OTHER_ORBIT_SPECIFIC_LOS = "OTHER_ORBIT_SPECIFIC_LOS"
    """Some other observation specific to this orbit's line of sight."""

    SYSTEMIC_RADIAL_VELOCITY = "SYSTEMIC_RADIAL_VELOCITY"
    """Explicitly insufficient: says nothing about which node recedes."""

    NONE = "NONE"
    """No evidence recorded."""

    @property
    def resolves_node(self) -> bool:
        """True only for evidence specific to this orbit's line of sight."""
        return self in (
            NodeSenseEvidence.ORBITAL_RV_SOLUTION,
            NodeSenseEvidence.DIRECT_COMPANION_LOS_VELOCITY,
            NodeSenseEvidence.ASTROMETRY_PLUS_ORBITAL_RV,
            NodeSenseEvidence.OTHER_ORBIT_SPECIFIC_LOS,
        )


class NodeSense(str, Enum):
    """Whether the ascending/descending sense of the node is resolved.

    A measured *number* is not a resolved node. Relative astrometry
    routinely determines the node only modulo 180 degrees, because
    ``(omega, Omega)`` and ``(omega + pi, Omega - pi)`` produce the same
    projected orbit on the sky. Distinguishing them needs radial-velocity or
    equivalent line-of-sight information.

    ``RESOLVED`` means evidence identified *this orbit's* receding node, and
    the evidence kind must be one :class:`NodeSenseEvidence` recognises as
    orbit-specific. A generic **systemic** stellar radial velocity is
    **not** enough by itself: it describes the whole system's motion
    relative to the Sun and says nothing about which of the two nodes of a
    planet's orbit recedes.
    """

    RESOLVED = "RESOLVED"
    """The receding node is identified; the 3D orientation is unique."""

    MODULO_180 = "MODULO_180"
    """The projected orbit is known; which node is ascending is not."""

    UNKNOWN = "UNKNOWN"
    """No information about the sense at all."""

    @property
    def is_resolved(self) -> bool:
        return self is NodeSense.RESOLVED


def position_angle_to_azimuth(position_angle_rad: float) -> float:
    """Catalogue position angle -> internal azimuth: ``theta = pi/2 - PA``.

    The single place this conversion is permitted. A position angle runs
    from North toward East; the internal azimuth runs from ``+X`` (East)
    toward ``+Y`` (North). They start at different axes and increase in
    opposite senses, which is exactly why feeding a catalogue angle straight
    into a rotation matrix is wrong rather than merely imprecise.
    """
    return float(np.pi / 2.0 - float(position_angle_rad))


def azimuth_to_position_angle(azimuth_rad: float) -> float:
    """Inverse of :func:`position_angle_to_azimuth`; it is its own inverse."""
    return float(np.pi / 2.0 - float(azimuth_rad))


def node_convention_of(node: Parameter | None) -> NodeConvention:
    """The recorded convention for a node parameter, or UNSPECIFIED.

    Read from ``Parameter.extra`` so an ingestion path can record it without
    every consumer having to know. Absence is never treated as the standard:
    silence must not unlock anything, and an unrecognised string fails
    closed rather than falling through as valid.
    """
    if node is None:
        return NodeConvention.UNSPECIFIED
    try:
        return NodeConvention(node.extra.get(NODE_KEY_CONVENTION))
    except ValueError:
        return NodeConvention.UNSPECIFIED


def node_sense_evidence_of(node: Parameter | None) -> NodeSenseEvidence:
    """The recorded evidence kind, or NONE.

    Typed rather than free text. A future ingestion that writes
    ``"systemic radial velocity"`` as prose would otherwise satisfy a
    non-empty-string check and unlock a gate the documentation says it must
    not.
    """
    if node is None:
        return NodeSenseEvidence.NONE
    try:
        return NodeSenseEvidence(node.extra.get(NODE_KEY_EVIDENCE))
    except ValueError:
        return NodeSenseEvidence.NONE


def node_sense_of(node: Parameter | None) -> NodeSense:
    """The recorded ascending/descending sense, or UNKNOWN.

    A node with a value but no recorded sense is ``MODULO_180``, not
    ``RESOLVED``: having a number is the normal situation in which the
    ambiguity exists.

    ``RESOLVED`` additionally requires a :class:`NodeSenseEvidence` kind
    that actually resolves a node. An unevidenced tag, an unrecognised one,
    or ``SYSTEMIC_RADIAL_VELOCITY`` all fall back to ``MODULO_180``.
    """
    if node is None or not node.is_known:
        return NodeSense.UNKNOWN
    try:
        recorded = NodeSense(node.extra.get(NODE_KEY_SENSE))
    except ValueError:
        return NodeSense.MODULO_180
    if recorded is NodeSense.RESOLVED and not node_sense_evidence_of(node).resolves_node:
        return NodeSense.MODULO_180
    return recorded


def resolve_node_azimuth(node: Parameter | None, default: float = 0.0) -> float:
    """The internal azimuth for a node parameter, in radians.

    **This is the only supported way for a node angle to reach**
    ``rotation_perifocal_to_inertial``. It applies
    :func:`position_angle_to_azimuth`, including to a display normalisation:
    a normalised ``Omega_PA = 0`` means North, and North is ``pi/2``
    internally. Passing the zero through unconverted would draw the line of
    nodes East while the annotation beside it said North.

    ``default`` is the position angle to assume when the node is unknown,
    and is itself converted - so the caller states a *sky* convention and
    never has to think in internal axes.
    """
    position_angle = default if node is None else node.value_in(u.rad, default)
    if position_angle is None:  # pragma: no cover - value_in honours default
        position_angle = default
    return position_angle_to_azimuth(position_angle)


def node_is_constrained(node: Parameter | None) -> bool:
    """True when the node is an observation rather than a normalisation."""
    return (
        node is not None
        and node.is_known
        and node.status in (Status.MEASURED, Status.DERIVED)
    )


__all__ += ["node_is_constrained"]
