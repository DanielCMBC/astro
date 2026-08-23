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

One coordinate is deliberately *not* produced: the planet's absolute
position. See :data:`ABSOLUTE_POSITION_UNRESOLVED` - the host term and the
local term live in different bases, and no rotation between them exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status, derived, unknown
from .frames import Frame, SkyPosition

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
    "ABSOLUTE_POSITION_UNRESOLVED",
    "ABSOLUTE_POSITION_NO_HOST",
    "absolute_planet_position",
    "planet_distance_rows",
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


def _combined_status(*parameters: Parameter) -> Status:
    """The status a value computed from ``parameters`` is entitled to.

    Deliberately pessimistic, in this order: anything unknown makes the
    result unknown; anything assumed for visualisation makes the result
    assumed, however solid the arithmetic; otherwise the result is derived,
    because it was computed rather than observed.

    The middle rule is the one with teeth. ``a(1-e)`` is an exact identity,
    so it is tempting to report a derived periapsis from an assumed
    semimajor axis - and that would turn a number invented so a picture
    could be drawn into a quotable orbital distance.
    """
    if any(not p.is_known for p in parameters):
        return Status.UNKNOWN
    if any(p.status is Status.ASSUMED_FOR_VISUALIZATION for p in parameters):
        return Status.ASSUMED_FOR_VISUALIZATION
    return Status.DERIVED


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


#: Why C3 publishes no absolute planet position at all.
#:
#: The host's Cartesian position lives in the ICRS basis. The planet's local
#: vector lives in the system frame, whose axes are set by the orbital
#: transform: the reference plane is the plane of the sky, and ``+x`` within
#: it is the direction the longitude of the ascending node is measured from.
#:
#: There is no defined rotation between those two bases in this codebase, so
#: adding the two vectors is not a valid vector addition - it adds components
#: expressed in different axes. That is a *basis* error, not an uncertainty,
#: and it does not become correct when the node happens to be measured: the
#: mapping from the system frame to a local tangent triad at the host
#: (radial, east, north) still has to be defined and tested before any such
#: sum means anything.
#:
#: At 66 pc an AU-scale offset is numerically tiny, which is exactly the trap.
#: A small error is not a correct coordinate, and scale must not be allowed to
#: hide a basis mistake. So the row is reported as unresolved and carries the
#: reason, rather than publishing a number that would read as an ICRS
#: position. Building the real transform is its own milestone.
ABSOLUTE_POSITION_UNRESOLVED = (
    "not resolved - the SystemFrame -> ICRS basis rotation is not defined, "
    "so the host's ICRS vector and the planet's local vector cannot be added"
)

#: The additional reason that applies when the host has no address either.
ABSOLUTE_POSITION_NO_HOST = "the host has no usable absolute position"


def absolute_planet_position(
    host: SkyPosition | None,
    position_au,
) -> CoordinateRow:
    """The planet's absolute position - deliberately unresolved in C3.

    Always returns an UNKNOWN row. See :data:`ABSOLUTE_POSITION_UNRESOLVED`:
    the host term and the local term are expressed in different bases, and
    C3 has no rotation between them. Returning the sum anyway would publish
    a triplet labelled ICRS that is not an ICRS position.

    A row is still returned rather than nothing, because "we cannot resolve
    this and here is why" is information the panel should show. What is
    withheld is the number, not the explanation.

    Everything that *is* well defined remains available:
    :func:`system_frame_position` for the local coordinates and
    :func:`host_planet_distance` for the separation. Neither needs a basis
    rotation - a distance is invariant under one, and the local triplet is
    reported in the frame it is actually expressed in.
    """
    note = ABSOLUTE_POSITION_UNRESOLVED
    if host is None or not host.has_distance:
        note = "{0}; also, {1}".format(note, ABSOLUTE_POSITION_NO_HOST)
    return CoordinateRow(
        "Absolute position",
        None,
        u.pc,
        InspectorFrame.ICRS,
        status=Status.UNKNOWN,
        provenance="unresolved(r_host + R * r_planet/local)",
        note=note,
    )


def planet_distance_rows(elements, position_au, *, host: SkyPosition | None = None) -> list:
    """Every distance and coordinate row for one selected planet."""
    rows = [
        InspectorRow("Distance from host", host_planet_distance(position_au)),
        InspectorRow("Periapsis distance", periapsis_distance(elements)),
        InspectorRow("Apoapsis distance", apoapsis_distance(elements)),
        system_frame_position(position_au),
    ]
    if host is not None:
        rows.append(absolute_planet_position(host, position_au))
    return rows
