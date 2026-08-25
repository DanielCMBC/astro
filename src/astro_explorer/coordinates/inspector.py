"""Explorer C3: the read-only coordinate and distance model.

C1 and C2 made the explorer show derived *regions* and *orientations*
honestly. C3 makes it numerically useful: what are this star's coordinates,
how far away is it, and how far is this planet from its host right now.

The rule the whole module exists to enforce is a single sentence:

    every scientific distance and coordinate comes from float64 scientific
    state, and never from display geometry.

That is not pedantry. The renderer deliberately works in float32, exaggerates
radii so a planet is visible next to its star, swaps in coarser meshes at
distance, and moves everything relative to the camera. Every one of those is
correct for drawing and fatal for measuring. A distance read back out of the
scene would therefore be a number that changes when the viewer zooms - which
is precisely the sort of quantity that looks authoritative and is not.

So nothing here imports the rendering layer, and the functions take
scientific inputs only: a :class:`~astro_explorer.coordinates.frames.SkyPosition`,
orbital elements, and the propagator's float64 position in AU. The
architecture tests hold that line from the outside.

Three shapes of row come out, and the differences matter:

* :class:`InspectorRow` for a scalar - a distance, a light travel time, an
  angle - carrying a full :class:`~astro_explorer.provenance.Parameter`;
* :class:`CoordinateRow` for an x/y/z triplet, which *cannot be constructed
  without naming its frame*. A bare triplet is meaningless: the same planet
  is at three entirely different coordinates in ICRS, in Galactic and in its
  own system frame, and a reader who is not told which one is looking at a
  number that means nothing;
* :class:`NoteRow` for a qualifier that is words rather than a number - the
  phase provenance, which decides whether the instantaneous distances above
  it describe tonight or merely describe the motion.

One coordinate is produced only under conditions. The planet's absolute
ICRS position needs the C3.5 SystemFrame -> ICRS rotation, a node whose
convention and sense the catalogue actually recorded, and - since C3.6 - a
host astrometric state propagated to the same instant the orbit was. Every
gate that is not satisfied is reported by name on the row, so a withheld
coordinate says which piece of science is missing rather than only that one
is: see :func:`absolute_planet_position`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..physics.timed_state import TimedOrbitalState, same_instant
from ..provenance import Parameter, Status, combined_status, derived, unknown
from .astrometry import PropagatedAstrometry
from .frames import Frame, SkyPosition
from .tangent import absolute_position_blockers, system_offset_to_icrs_pc

__all__ = [
    "InspectorFrame",
    "InspectorRow",
    "NoteRow",
    "CoordinateRow",
    "galactic_coordinates",
    "star_coordinate_rows",
    "host_planet_distance",
    "periapsis_distance",
    "apoapsis_distance",
    "system_frame_position",
    "ABSOLUTE_POSITION_NO_HOST",
    "ABSOLUTE_POSITION_NO_ORBIT",
    "PLANET_STATE_NOT_TIMED",
    "PLANET_TIME_MISMATCH",
    "PLANET_PHASE_NOT_CONSTRAINED",
    "absolute_planet_position",
    "planet_distance_rows",
    "planet_to_star_distance",
    "COMMON_TIME_VIOLATED",
    "TARGET_ASTROMETRY_NOT_PROPAGATED",
]


class InspectorFrame(str, Enum):
    """The frame a displayed coordinate belongs to.

    ``NONE`` is for genuinely frame-independent scalars - a separation, a
    light travel time - and is the one value a :class:`CoordinateRow`
    refuses to accept.
    """

    ICRS = "ICRS"
    GALACTIC = "Galactic"
    GALACTOCENTRIC = "Galactocentric"
    SYSTEM = "SystemFrame"
    NONE = ""

    @property
    def sky_frame(self) -> Frame | None:
        """The :class:`~astro_explorer.coordinates.frames.Frame`, if any.

        ``SYSTEM`` has none: a host-centred AU frame is not a celestial
        frame Astropy can transform to, and pretending otherwise is how a
        local offset ends up quietly interpreted as a position on the sky.
        """
        return {
            InspectorFrame.ICRS: Frame.ICRS,
            InspectorFrame.GALACTIC: Frame.GALACTIC,
            InspectorFrame.GALACTOCENTRIC: Frame.GALACTOCENTRIC,
        }.get(self)


@dataclass(frozen=True)
class InspectorRow:
    """One scalar quantity, with everything needed to read it correctly."""

    label: str
    parameter: Parameter
    frame: InspectorFrame = InspectorFrame.NONE
    epoch: str = ""

    @property
    def is_known(self) -> bool:
        return self.parameter.is_known

    def format(self, digits: int = 6) -> str:
        text = "{0}: {1}".format(self.label, self.parameter.format(digits))
        if self.frame is not InspectorFrame.NONE:
            text += " [{0}]".format(self.frame.value)
        if self.epoch:
            text += " (epoch {0})".format(self.epoch)
        return text


@dataclass(frozen=True)
class NoteRow:
    """A qualifier that is words rather than a number.

    Exists for one thing in particular: the phase provenance. "Distance from
    host: 0.2057 AU" means something quite different depending on whether the
    planet's position at this instant is fixed by a published epoch or is
    being advanced from an arbitrary zero. In the second case the number is a
    picture of the motion, not a claim about tonight.

    That qualifier travels in the row list rather than only in prose, so a
    panel cannot render the distances and silently drop the sentence that
    says how to read them.
    """

    label: str
    text: str
    status: Status = Status.UNKNOWN

    @property
    def is_known(self) -> bool:
        return self.status is not Status.UNKNOWN

    def format(self, digits: int = 6) -> str:
        del digits  # a note has no precision to control
        return "{0}: {1}".format(self.label, self.text)


@dataclass(frozen=True)
class CoordinateRow:
    """An x/y/z triplet that is required to say what it is a triplet *of*.

    The frame is not optional and there is no default. A triplet whose frame
    the caller could forget to supply is exactly the failure this type
    exists to make impossible, so omission is a construction error rather
    than a display quirk noticed later.
    """

    label: str
    values: np.ndarray | None
    unit: u.UnitBase
    frame: InspectorFrame
    status: Status = Status.UNKNOWN
    provenance: str = ""
    epoch: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.frame is InspectorFrame.NONE:
            raise ValueError(
                "a coordinate triplet must name its frame: the same object "
                "has different x/y/z in ICRS, Galactic and its system frame"
            )
        if self.values is None:
            return
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError("a coordinate triplet must have exactly 3 components")
        if not np.all(np.isfinite(values)):
            # An unknown position is UNKNOWN, not NaN dressed as a number.
            raise ValueError(
                "coordinate components must be finite; an unknown position "
                "must be passed as None so it reads as UNKNOWN"
            )
        object.__setattr__(self, "values", values)

    @property
    def is_known(self) -> bool:
        return self.values is not None

    def component(self, index: int) -> Parameter:
        """One axis as a full :class:`Parameter`, unit and status intact."""
        if self.values is None:
            return unknown(self.unit, provenance=self.provenance)
        return Parameter(
            float(self.values[index]),
            self.unit,
            status=self.status,
            provenance=self.provenance,
            note=self.note,
        )

    def to(self, unit: u.UnitBase) -> "CoordinateRow":
        """The same triplet in another unit, frame and status preserved."""
        if self.values is None:
            return CoordinateRow(
                self.label, None, u.Unit(unit), self.frame,
                status=self.status, provenance=self.provenance,
                epoch=self.epoch, note=self.note,
            )
        factor = (1.0 * self.unit).to_value(unit)
        return CoordinateRow(
            self.label,
            self.values * factor,
            u.Unit(unit),
            self.frame,
            status=self.status,
            provenance=self.provenance,
            epoch=self.epoch,
            note=self.note,
        )

    def norm(self, label: str = "") -> Parameter:
        """The length of the triplet, keeping the triplet's own status."""
        if self.values is None:
            return unknown(self.unit, provenance=self.provenance)
        return Parameter(
            float(np.linalg.norm(self.values)),
            self.unit,
            status=self.status,
            provenance=label or "|{0}|".format(self.provenance or self.label),
        )

    def format(self, digits: int = 6) -> str:
        if self.values is None:
            # The note is the useful half of an unknown row: "unknown" alone
            # reads as missing data, when the reason may be that the value
            # is not well defined at all.
            text = "{0}: unknown [{1}]".format(self.label, self.frame.value)
            return "{0} - {1}".format(text, self.note) if self.note else text
        body = ", ".join("{0:.{1}g}".format(v, digits) for v in self.values)
        text = "{0}: ({1}) {2} [{3}]".format(
            self.label, body, self.unit.to_string(), self.frame.value
        )
        if self.status is not Status.MEASURED:
            text += " ({0})".format(Parameter(0.0, status=self.status).status_label)
        if self.epoch:
            text += " (epoch {0})".format(self.epoch)
        return text


#: The pessimistic status rule now lives in :mod:`astro_explorer.provenance`,
#: beside :class:`~astro_explorer.provenance.Status` itself, so the coordinate
#: inspector and the C4a HR placement cannot drift apart about what an
#: assumed input entitles a computed value to claim. Kept under the old name
#: because every call site in this module reads better for it.
_combined_status = combined_status


def _with_status(value: float, unit, status: Status, provenance: str, note: str = "") -> Parameter:
    if status is Status.UNKNOWN:
        return unknown(unit, provenance=provenance, note=note)
    return Parameter(float(value), unit, status=status, provenance=provenance, note=note)


# -- the host star ----------------------------------------------------------


def galactic_coordinates(position: SkyPosition) -> tuple[Parameter, Parameter]:
    """Galactic ``(l, b)`` for a sky position, via Astropy.

    Distance is not required: ``l`` and ``b`` are a direction, and a star
    with an unusable parallax still has a perfectly well determined one.
    That is why this does not go through :meth:`SkyPosition.cartesian_pc`,
    which needs a distance and correctly refuses without one.
    """
    coord = position.skycoord
    if coord is None:
        provenance = "galactic(ra, dec)"
        return unknown(u.deg, provenance=provenance), unknown(u.deg, provenance=provenance)

    galactic = coord.galactic
    status = _combined_status(position.ra, position.dec)
    return (
        _with_status(
            galactic.l.to_value(u.deg), u.deg, status, "galactic_l(ra, dec)"
        ),
        _with_status(
            galactic.b.to_value(u.deg), u.deg, status, "galactic_b(ra, dec)"
        ),
    )


def _cartesian_row(
    position: SkyPosition, frame: InspectorFrame, label: str
) -> CoordinateRow:
    sky_frame = frame.sky_frame
    values = position.cartesian_pc(sky_frame) if sky_frame is not None else None
    status = (
        _combined_status(position.ra, position.dec, position.distance)
        if values is not None
        else Status.UNKNOWN
    )
    return CoordinateRow(
        label,
        None if status is Status.UNKNOWN else values,
        u.pc,
        frame,
        status=status,
        provenance="cartesian({0})".format(frame.value.lower()),
        note="" if values is not None else "no usable distance for this host",
    )


def star_coordinate_rows(position: SkyPosition | None) -> list:
    """Every coordinate and distance row for a host star.

    A host with no usable distance is not a host at the origin. It keeps its
    RA, Dec and Galactic direction - those are measured - and reports UNKNOWN
    for everything that needs a radial scale. This is the same detached-frame
    rule Explorer B established for the system frame, applied to the sky.
    """
    if position is None:
        return []

    l, b = galactic_coordinates(position)
    rows = [
        InspectorRow("Right ascension", position.ra.to(u.deg), InspectorFrame.ICRS),
        InspectorRow("Declination", position.dec.to(u.deg), InspectorFrame.ICRS),
        InspectorRow("Galactic longitude l", l, InspectorFrame.GALACTIC),
        InspectorRow("Galactic latitude b", b, InspectorFrame.GALACTIC),
        # One distance row, in parsecs. Light years and kilometres are the
        # same row rendered differently, and a caller that wants them has
        # ``row.parameter.to(...)``; two rows with the same label reading
        # different numbers is a table that invites a misreading.
        InspectorRow("Distance from Sun", position.distance.to(u.pc)),
        InspectorRow("Light travel time", position.light_travel_time()),
        _cartesian_row(position, InspectorFrame.ICRS, "Cartesian position"),
        _cartesian_row(position, InspectorFrame.GALACTIC, "Cartesian position"),
    ]
    return rows


# -- the selected planet ----------------------------------------------------


def host_planet_distance(position_au) -> Parameter:
    """Instantaneous host-to-planet distance from the propagated state.

    ``position_au`` is the float64 orbital position the physics layer
    produced, in AU, star-centred - the value *before* anything turned it
    into render coordinates. Taking its norm is the whole computation.

    It is deliberately not recomputed from ``a(1 - e cos E)``. That identity
    holds, and the tests check that it holds, but recomputing it here would
    create a second opinion about the orbit that could drift from the one
    actually drawn. One propagation, one answer.
    """
    if position_au is None:
        return unknown(u.au, provenance="|r_planet|")
    values = np.asarray(position_au, dtype=np.float64)
    if values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        return unknown(u.au, provenance="|r_planet|")
    return derived(
        float(np.linalg.norm(values)), u.au, provenance="|r_planet| (propagated state)"
    )


def periapsis_distance(elements) -> Parameter:
    """``a(1-e)``, carrying the provenance of ``a`` and ``e``."""
    axis, ecc = elements.semimajor_axis, elements.eccentricity
    status = _combined_status(axis, ecc)
    if status is Status.UNKNOWN:
        return unknown(u.au, provenance="a(1-e)")
    return _with_status(
        axis.value_in(u.au) * (1.0 - ecc.value), u.au, status, "a(1-e)"
    )


def apoapsis_distance(elements) -> Parameter:
    """``a(1+e)``, carrying the provenance of ``a`` and ``e``."""
    axis, ecc = elements.semimajor_axis, elements.eccentricity
    status = _combined_status(axis, ecc)
    if status is Status.UNKNOWN:
        return unknown(u.au, provenance="a(1+e)")
    return _with_status(
        axis.value_in(u.au) * (1.0 + ecc.value), u.au, status, "a(1+e)"
    )


def system_frame_position(position_au, *, label: str = "System-frame position") -> CoordinateRow:
    """The planet's star-centred position, in AU, labelled as such.

    No arithmetic: the system frame is already AU and already star-centred,
    which is the point of it. The work this does is naming the frame.
    """
    if position_au is None:
        values = None
    else:
        values = np.asarray(position_au, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            values = None
    return CoordinateRow(
        label,
        values,
        u.au,
        InspectorFrame.SYSTEM,
        status=Status.DERIVED if values is not None else Status.UNKNOWN,
        provenance="propagated orbital state",
        note="" if values is not None else "the orbit could not be propagated",
    )


#: The reason that applies when the host itself has no address.
ABSOLUTE_POSITION_NO_HOST = "the host has no usable absolute position"

#: The reason that applies when the orbit could not be propagated.
ABSOLUTE_POSITION_NO_ORBIT = "the orbit could not be propagated"

#: The reason that applies when the planet vector carries no instant.
#:
#: A bare ``(3,)`` array of AU cannot disagree with the host's obstime,
#: which is exactly why it must not be able to satisfy the epoch gate: the
#: host and the planet would combine perfectly while holding at different
#: dates. Publication therefore requires a
#: :class:`~astro_explorer.physics.timed_state.TimedOrbitalState`.
PLANET_STATE_NOT_TIMED = (
    "the planet's orbital state carries no instant, so it cannot be shown "
    "to hold at the same time as the host's propagated position"
)

#: The reason that applies when the planet and the host hold at different
#: instants. Both can be individually correct and the sum still wrong.
PLANET_TIME_MISMATCH = (
    "the planet's orbital state and the host's astrometry were evaluated at "
    "different instants, so the sum mixes epochs"
)

#: The reason that applies when the planet's *place* on the ellipse is real
#: but its *date* is not.
PLANET_PHASE_NOT_CONSTRAINED = (
    "the planet's phase is advanced from an arbitrary zero rather than from "
    "a published epoch, so the position describes the motion and not a date"
)


def _planet_time_blockers(
    planet, astrometry: PropagatedAstrometry | None
) -> list[str]:
    """Why this planet vector may not be added to this host position.

    Separate from the node and astrometry gates because it is a *relational*
    check on top of two individually valid things. The host can be
    propagated impeccably and the orbit propagated impeccably, and the sum
    still be an answer about no particular night.
    """
    if astrometry is None:
        # The epoch gate is already closed by ASTROMETRY_NOT_PROPAGATED;
        # adding a second sentence about the planet's clock would report the
        # same missing half twice.
        return []
    if not isinstance(planet, TimedOrbitalState):
        return [PLANET_STATE_NOT_TIMED]
    reasons = []
    if not planet.at_same_time_as(astrometry):
        reasons.append(PLANET_TIME_MISMATCH)
    if not planet.is_time_constrained:
        reasons.append(PLANET_PHASE_NOT_CONSTRAINED)
    return reasons


def absolute_planet_position(
    host: SkyPosition | None,
    planet,
    *,
    node: Parameter | None = None,
    astrometry: PropagatedAstrometry | None = None,
    normalised: bool = False,
) -> CoordinateRow:
    r"""The planet's absolute ICRS position - still withheld, now precisely.

    .. math::
        \mathbf r_{planet} = \mathbf r_{host} + R_{sky
ightarrow ICRS}\,\mathbf r_{local}

    C3.5 supplies :math:`R`, which C3 lacked. That removes the *basis* error
    but not the remaining scientific gates, and every one of them is checked
    here through :func:`absolute_position_blockers`:

    * the host must have a usable distance - otherwise there is no origin;
    * the orbit must propagate - otherwise there is no offset;
    * the orbit's **physical orientation** must be an observation, not a
      display normalisation. All three Euler angles: the inclination that
      tilts the plane, the planet-frame argument of periapsis that orients
      the ellipse within it under a *stated* convention, and the node's
      convention and sense. The node gates alone are necessary and not
      sufficient - an orbit drawn face-on because nobody measured its
      inclination has no plane in space, however well its node is tagged.
      The host must also not sit at a pole where sky-plane azimuth is
      gauge-dependent;
    * the **coordinate epoch** must be handled, and C3.6 handles it. Pass a
      :class:`~astro_explorer.coordinates.astrometry.PropagatedAstrometry`
      for the host, evaluated at the *same* instant the orbit was
      propagated to. Omitting it blocks the row, exactly as the old
      ``epoch_resolved=False`` did - but there is now no argument that
      opens the gate without a real propagated state behind it.

    When astrometry is supplied and publishable, **the propagated position
    is the one used** - as the origin and as the source of the tangent
    basis. That is the common-time rule made structural rather than
    documented: there is no path through this function that builds the
    basis from the catalogue-epoch RA/Dec and then adds the offset to the
    propagated origin, because the two come from the same object.

    ``planet`` is the other half of that rule. It may be a bare ``(3,)``
    array of AU, which is enough to *draw* something, or a
    :class:`~astro_explorer.physics.timed_state.TimedOrbitalState`, which is
    what publication requires. An array carries neither an instant nor the
    elements it came from, so it can answer neither "does this hold at the
    host's time" nor "is this orbit's orientation an observation" - the
    check is structural rather than a convention the caller has to observe.
    A timed state additionally has to agree with the host's instant, to have
    reached it from a published epoch rather than an arbitrary zero, and to
    carry a fully resolved physical orientation.

    ``node`` is used only when no timed state is supplied. When one is, its
    element set is the single source of orientation truth, node included, so
    there is no way for two disagreeing nodes to reach the same row.

    Every failing gate is reported, not just the first: a row blocked for
    three reasons should say three.

    ``normalised=True`` returns the display realisation anyway, stamped
    ASSUMED_FOR_VISUALIZATION and carrying every unresolved reason. It
    exists so a scene can be drawn. It is never a catalogue coordinate.
    """
    position_au = (
        planet.position_au if isinstance(planet, TimedOrbitalState) else planet
    )
    local = system_frame_position(position_au)
    label = "Absolute position"
    frame = InspectorFrame.ICRS
    provenance = "r_host + R_sky->icrs * r_planet/local"

    # The common-time rule: when a propagated state is available and usable,
    # it *replaces* the catalogue-epoch host everywhere below - origin and
    # basis together. A propagation that fell short leaves the catalogue
    # position in place, and its blockers make the row unpublishable anyway.
    if astrometry is not None and astrometry.is_publishable:
        host = astrometry.position

    reasons: list[str] = []
    if host is None or not host.has_distance:
        reasons.append(ABSOLUTE_POSITION_NO_HOST)
    if not local.is_known:
        reasons.append(ABSOLUTE_POSITION_NO_ORBIT)
    # A dated orbital state carries the elements it is a state of, so the
    # full orientation gate runs. A bare vector cannot answer the question -
    # and is already refused below by PLANET_STATE_NOT_TIMED, so the narrower
    # node-only check it falls back to can never publish.
    elements = planet.elements if isinstance(planet, TimedOrbitalState) else None
    reasons.extend(
        r
        for r in absolute_position_blockers(
            host, node, astrometry=astrometry, elements=elements
        )
        if r not in reasons
    )
    reasons.extend(
        r for r in _planet_time_blockers(planet, astrometry) if r not in reasons
    )

    # Nothing can be computed at all without an origin and an offset.
    unbuildable = (
        ABSOLUTE_POSITION_NO_HOST in reasons or ABSOLUTE_POSITION_NO_ORBIT in reasons
    )
    if reasons and (unbuildable or not normalised):
        return CoordinateRow(
            label, None, u.pc, frame,
            status=Status.UNKNOWN,
            provenance=provenance,
            note="; also, ".join(reasons),
        )

    host_pc = host.cartesian_pc(Frame.ICRS)
    offset_pc = system_offset_to_icrs_pc(host, local.values)
    if host_pc is None or offset_pc is None:  # pragma: no cover - guarded above
        return CoordinateRow(
            label, None, u.pc, frame, status=Status.UNKNOWN,
            provenance=provenance, note=ABSOLUTE_POSITION_NO_HOST,
        )

    return CoordinateRow(
        label,
        np.asarray(host_pc, dtype=np.float64) + offset_pc,
        u.pc,
        frame,
        status=Status.ASSUMED_FOR_VISUALIZATION
        if reasons
        else _combined_status(host.ra, host.dec, host.distance),
        provenance=provenance,
        note="; also, ".join(reasons),
    )


def planet_distance_rows(
    elements,
    planet,
    *,
    host: SkyPosition | None = None,
    astrometry: PropagatedAstrometry | None = None,
) -> list:
    """Every distance and coordinate row for one selected planet.

    ``planet`` is a
    :class:`~astro_explorer.physics.timed_state.TimedOrbitalState` for the
    absolute-position row to be publishable; a bare AU vector still produces
    every host-relative row, because a separation from the host is a
    property of the orbit and needs no coordinate epoch at all.

    ``astrometry`` must be the host's state propagated to the same instant.
    The caller owns that pairing because it owns the clock, and
    :class:`~astro_explorer.app.vertical_slice.SystemSlice` does it in one
    method so there is one place to check rather than one per row.
    """
    position_au = (
        planet.position_au if isinstance(planet, TimedOrbitalState) else planet
    )
    rows = [
        InspectorRow("Distance from host", host_planet_distance(position_au)),
        InspectorRow("Periapsis distance", periapsis_distance(elements)),
        InspectorRow("Apoapsis distance", apoapsis_distance(elements)),
        system_frame_position(position_au),
    ]
    if host is not None or astrometry is not None:
        rows.append(
            absolute_planet_position(
                host,
                planet,
                node=elements.longitude_of_ascending_node,
                astrometry=astrometry,
            )
        )
    return rows


#: The reason that applies when the two objects were not evaluated at one
#: instant. Its own sentence because it is a *relational* failure: each
#: state can be individually perfect and the pair still meaningless.
COMMON_TIME_VIOLATED = (
    "the host and the target star were propagated to different instants, so "
    "the separation between them mixes coordinate epochs"
)

#: The reason that applies when the target star has no propagated state.
TARGET_ASTROMETRY_NOT_PROPAGATED = (
    "the target star has no astrometric state propagated to the requested "
    "time, so its position is at an unstated epoch"
)


def planet_to_star_distance(
    host: SkyPosition | None,
    planet,
    other: SkyPosition | None,
    *,
    node: Parameter | None = None,
    astrometry: PropagatedAstrometry | None = None,
    other_astrometry: PropagatedAstrometry | None = None,
) -> Parameter:
    r"""Distance from a planet to another star, in a common float64 frame.

    .. math::
        D = \left|\mathbf r_{other} - \mathbf r_{planet}
ight|

    Needs a real absolute planet vector, so every gate that blocks
    :func:`absolute_planet_position` blocks this too, and the reasons are
    passed through rather than replaced.

    C3.6 adds the gate this function could not previously state separately:
    **three** clocks must agree, not two.

    .. code-block:: text

        host astrometry obstime
            == target-star astrometry obstime
            == planet orbital-state obstime

    Two of those were checked from the start; the planet's was the one that
    could not be, because a bare offset array has no instant. It does now:
    ``planet`` is a
    :class:`~astro_explorer.physics.timed_state.TimedOrbitalState` for any
    published answer, and all three are compared as **instants** rather than
    as Julian day numbers - two objects reached through different time
    scales sit at the same instant while their JDs differ by up to about
    89 seconds, and comparing the numbers would reject that pair while
    accepting a genuinely mismatched one that happened to share a number.

    Each state can be individually impeccable and the difference still
    meaningless: a host at J2016 minus a target at J2000 is a separation
    that includes sixteen years of both stars' motion. So the times are
    compared rather than assumed equal, and a mismatch is reported as
    :data:`COMMON_TIME_VIOLATED` rather than silently differenced.

    Every orientation gate is inherited rather than restated: this needs a
    real absolute planet vector, so it calls
    :func:`absolute_planet_position` and passes its refusal through. A
    second copy of the orientation rules here would be a second opinion able
    to drift from the first.

    It deliberately does **not** fall back to the host-to-star separation.
    That fallback would be plausible - the two differ by an AU at parsec
    range - and it would answer a question nobody asked, with nothing on the
    number saying it was about the star instead of the planet.
    """
    provenance = "|r_other - r_planet| (ICRS, float64, common epoch)"

    if other_astrometry is not None and other_astrometry.is_publishable:
        other = other_astrometry.position

    if other is None or not other.has_distance:
        return unknown(
            u.pc, provenance=provenance, note="the other star has no usable distance"
        )

    absolute = absolute_planet_position(
        host, planet, node=node, astrometry=astrometry
    )
    if not absolute.is_known or absolute.status is Status.ASSUMED_FOR_VISUALIZATION:
        # No fallback: the reason travels instead of a substitute number.
        return unknown(u.pc, provenance=provenance, note=absolute.note)

    # The planet vector cleared every gate, so ``astrometry`` is publishable
    # and carries the instant the whole answer is supposed to be at. The
    # target has to be brought to that same instant, not merely be good.
    if other_astrometry is None or not other_astrometry.is_publishable:
        blocked = [TARGET_ASTROMETRY_NOT_PROPAGATED]
        if other_astrometry is not None:
            blocked.extend(other_astrometry.blockers)
        return unknown(u.pc, provenance=provenance, note="; also, ".join(blocked))
    # Three clocks. The host-planet pair was checked inside
    # ``absolute_planet_position``; what is left is the target star against
    # both of them, and the planet is compared to the target directly rather
    # than trusting transitivity through a tolerance.
    if (
        astrometry is None
        or not same_instant(astrometry, other_astrometry)
        or not same_instant(planet, other_astrometry)
    ):
        return unknown(u.pc, provenance=provenance, note=COMMON_TIME_VIOLATED)

    other_pc = other.cartesian_pc(Frame.ICRS)
    if other_pc is None:  # pragma: no cover - guarded by has_distance
        return unknown(u.pc, provenance=provenance)

    separation = float(
        np.linalg.norm(np.asarray(other_pc, dtype=np.float64) - absolute.values)
    )
    status = _combined_status(other.ra, other.dec, other.distance)
    if Status.ASSUMED_FOR_VISUALIZATION in (status, absolute.status):
        status = Status.ASSUMED_FOR_VISUALIZATION
    return _with_status(separation, u.pc, status, provenance)
