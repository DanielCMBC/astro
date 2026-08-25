"""Explorer C3.5.1: what a catalogued ascending node actually means.

C3.5 established the canonical sky frame - ``+X`` East, ``+Y`` North,
``+Z`` away from the observer - and supplied
:func:`position_angle_to_azimuth` to convert a catalogue angle into an
internal one. What it did *not* do was put that conversion on the
production path.

That gap is the reason this module exists at the physics level rather than
in ``coordinates``. Two production routes feed a node angle into
``R_z(Omega) R_x(i) R_z(omega)``:

* :func:`~astro_explorer.physics.orbital_elements.position_at_eccentric_anomaly`
  and its siblings, through ``_display_angles`` - which is also how the
  vertical slice's propagated state reaches the transform, since C3.6
  routed ``SystemSlice.state`` through the provenance-aware wrapper rather
  than letting it unpack the angles itself;
* the C2 orientation guides.

If the conversion lives anywhere those cannot reach, one of them will
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

from dataclasses import dataclass
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
    "resolve_node_azimuth_detailed",
    "NodeAzimuth",
    "NODE_CONVENTION_NOT_RENDERABLE",
    "NODE_CONVENTION_UNSTATED",
    "NODE_SENSE_UNRESOLVED",
    "NODE_KEY_CONVENTION",
    "NODE_KEY_SENSE",
    "NODE_KEY_EVIDENCE",
    "NODE_KEY_EVIDENCE_NOTE",
]

#: Why a node may not contribute to a published absolute position. They
#: live here rather than in :mod:`astro_explorer.coordinates.tangent`
#: because they are statements about the *parameter* - what a catalogue did
#: and did not record about an angle - and not about the tangent basis. That
#: also lets one orientation gate report every reason together.
NODE_CONVENTION_UNSTATED = (
    "the node convention was not recorded, so the catalogued angle does not "
    "identify a direction on the sky"
)
NODE_SENSE_UNRESOLVED = (
    "the ascending/descending sense of the node is not resolved, so the "
    "orientation is known only modulo 180 degrees"
)

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


#: Why a scientific node value was not used as a direction.
NODE_CONVENTION_NOT_RENDERABLE = (
    "the node was measured but its convention was not recorded, so the "
    "number is not a position angle and is not drawn as one; the display "
    "normalisation is shown instead"
)


@dataclass(frozen=True)
class NodeAzimuth:
    """The azimuth a node parameter is entitled to, and why.

    :func:`resolve_node_azimuth` returns only the number, because that is
    all a rotation matrix wants. This is the same answer with the reasoning
    attached, for the display and provenance layers that have to say what
    the reader is looking at.
    """

    azimuth_rad: float
    """The internal azimuth, measured from ``+X`` (East) toward ``+Y``."""

    position_angle_rad: float
    """The sky position angle the azimuth came from."""

    convention: NodeConvention
    """The convention recorded on the parameter, if any."""

    used_catalogue_value: bool
    """False when the drawn angle is a normalisation, not the published one."""

    note: str = ""
    """Why the published value was not used, when it was not."""

    @property
    def is_display_normalisation(self) -> bool:
        return not self.used_catalogue_value


def resolve_node_azimuth_detailed(
    node: Parameter | None, default: float = 0.0
) -> NodeAzimuth:
    """:func:`resolve_node_azimuth` with its reasoning attached.

    Four cases, and the fourth is the one C3.6 adds:

    * **no parameter, or no value** - the caller's ``default`` position
      angle is used. That is the ordinary display normalisation, and the
      published node genuinely does not exist for most exoplanets;
    * **an ASSUMED_FOR_VISUALIZATION value** - used as a position angle.
      Display normalisation is *allowed* to assume the standard convention
      because it invented the number itself: ``Omega_PA = 0`` written by
      :meth:`OrbitalElements.for_display` means North, and saying so is the
      whole point of converting it;
    * **a scientific value under a stated convention** - used, converted;
    * **a scientific value under** :attr:`NodeConvention.UNSPECIFIED` -
      **not used**. An angle whose convention nobody recorded is a number,
      not a direction on the sky. Feeding it to the rotation would silently
      assert the standard convention, which is the failure mode that is
      right most of the time and therefore the worst one to have. The
      documented normalisation is drawn instead and
      :data:`NODE_CONVENTION_NOT_RENDERABLE` says so.

    The fourth case cannot arise from the NASA archive, which publishes no
    node at all. It is written now because the moment a provider *does*
    ingest real node values - which is exactly what C3.6 starts doing for
    astrometry - the silent path would already exist.
    """
    convention = node_convention_of(node)

    if node is None or not node.is_known:
        return NodeAzimuth(
            azimuth_rad=position_angle_to_azimuth(default),
            position_angle_rad=float(default),
            convention=convention,
            used_catalogue_value=False,
        )

    published = node.value_in(u.rad, default)
    if published is None:  # pragma: no cover - value_in honours default
        published = default

    if node.status.is_scientific and not convention.is_stated:
        return NodeAzimuth(
            azimuth_rad=position_angle_to_azimuth(default),
            position_angle_rad=float(default),
            convention=convention,
            used_catalogue_value=False,
            note=NODE_CONVENTION_NOT_RENDERABLE,
        )

    return NodeAzimuth(
        azimuth_rad=position_angle_to_azimuth(published),
        position_angle_rad=float(published),
        convention=convention,
        used_catalogue_value=True,
    )


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
    never has to think in internal axes. It is also what a *measured* node
    with an unrecorded convention falls back to; see
    :func:`resolve_node_azimuth_detailed` for why that is a refusal rather
    than a rounding of the truth.
    """
    return resolve_node_azimuth_detailed(node, default).azimuth_rad


def node_publication_blockers(node: Parameter | None) -> list[str]:
    """Why this node may not fix an absolute orientation. Empty when it may.

    The three states are not interchangeable and each gets its own sentence:

    * no node at all - or a node that is a display normalisation rather than
      an observation - leaves the orbit's rotation about the line of sight
      entirely unconstrained;
    * a node that is an observation but whose convention nobody recorded is
      a number, not a direction on the sky;
    * a node whose convention is stated but whose sense is not is known only
      modulo 180 degrees, because ``(omega, Omega)`` and
      ``(omega + pi, Omega - pi)`` project identically.
    """
    if not node_is_constrained(node):
        return [NODE_SENSE_UNRESOLVED]

    reasons: list[str] = []
    if not node_convention_of(node).is_stated:
        reasons.append(NODE_CONVENTION_UNSTATED)
    if not node_sense_of(node).is_resolved:
        reasons.append(NODE_SENSE_UNRESOLVED)
    return reasons


def node_is_constrained(node: Parameter | None) -> bool:
    """True when the node is an observation rather than a normalisation."""
    return (
        node is not None
        and node.is_known
        and node.status in (Status.MEASURED, Status.DERIVED)
    )


__all__ += ["node_is_constrained", "node_publication_blockers"]
