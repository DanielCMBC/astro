"""Explicit, non-mixable reference frames.

Roadmap section 9 and reference section 20. The rule this module enforces is
that parsecs, AU and kilometres are *never* combined in one OpenGL
coordinate space. It enforces it structurally: a position carries the frame
it belongs to, and arithmetic between two frames raises instead of silently
producing a number that looks plausible.

    UniverseFrame     origin = Sun or a floating origin,  unit = pc
    SystemFrame       origin = the host star,             unit = AU
    PlanetFrame       origin = the planet,                unit = km

All CPU storage is float64. :meth:`ReferenceFrame.to_render` is the single
place float32 appears, and it refuses to narrow a coordinate large enough
that float32 would lose sub-unit precision.

For the one-star-one-planet vertical slice only :class:`SystemFrame` is
needed: the star sits at the origin and the planet's position is the orbital
vector in AU, exactly as produced by
:mod:`astro_explorer.physics.orientation`. No conversion happens at all,
which is the point - the ``0.005`` scaling bug had no opportunity to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import astropy.units as u
import numpy as np

__all__ = [
    "FrameKind",
    "FramedPosition",
    "ReferenceFrame",
    "UniverseFrame",
    "SystemFrame",
    "PlanetFrame",
    "FrameMismatchError",
    "PrecisionError",
    "FLOAT32_SAFE_MAGNITUDE",
    "FRAME_ENGAGE_MARGIN",
]

#: float32 carries about seven significant decimal digits, so a coordinate
#: beyond this many display units cannot represent a unit step.
FLOAT32_SAFE_MAGNITUDE = 1.0e6

#: Safety factor between the float32 limit and the radius at which a frame is
#: allowed to become active. Ten leaves an order of magnitude of headroom for
#: the geometry drawn around the camera, not just the camera itself.
FRAME_ENGAGE_MARGIN = 10.0


class FrameMismatchError(TypeError):
    """Raised when two positions from different frames are combined."""


class PrecisionError(ValueError):
    """Raised when narrowing to float32 would lose meaningful precision."""


class FrameKind(str, Enum):
    """The three nested frames."""

    UNIVERSE = "UNIVERSE"
    SYSTEM = "SYSTEM"
    PLANET = "PLANET"

    @property
    def unit(self) -> u.UnitBase:
        return {
            FrameKind.UNIVERSE: u.pc,
            FrameKind.SYSTEM: u.au,
            FrameKind.PLANET: u.km,
        }[self]

    @property
    def unit_label(self) -> str:
        return {FrameKind.UNIVERSE: "pc", FrameKind.SYSTEM: "AU", FrameKind.PLANET: "km"}[self]


@dataclass(frozen=True)
class FramedPosition:
    """A position that knows which frame it is expressed in.

    ``values`` is float64 and shaped ``(3,)`` or ``(N, 3)``, in the frame's
    unit, relative to the frame's origin.
    """

    values: np.ndarray
    frame: "ReferenceFrame"

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.shape[-1] != 3 or array.ndim not in (1, 2):
            raise ValueError("a framed position must be (3,) or (N, 3)")
        object.__setattr__(self, "values", array)

    @property
    def kind(self) -> FrameKind:
        return self.frame.kind

    @property
    def unit(self) -> u.UnitBase:
        return self.frame.kind.unit

    def _require_same_frame(self, other: "FramedPosition") -> None:
        if not isinstance(other, FramedPosition):
            raise FrameMismatchError("can only combine two FramedPosition values")
        if self.frame is not other.frame and self.frame != other.frame:
            raise FrameMismatchError(
                "refusing to combine a position in {0} ({1}) with one in {2} ({3}); "
                "convert explicitly first".format(
                    self.frame.describe(),
                    self.kind.unit_label,
                    other.frame.describe(),
                    other.kind.unit_label,
                )
            )

    def __add__(self, other: "FramedPosition") -> "FramedPosition":
        self._require_same_frame(other)
        return FramedPosition(self.values + other.values, self.frame)

    def __sub__(self, other: "FramedPosition") -> "FramedPosition":
        self._require_same_frame(other)
        return FramedPosition(self.values - other.values, self.frame)

    def distance_to(self, other: "FramedPosition") -> np.ndarray:
        """Distance in the frame's unit; refuses a cross-frame comparison."""
        self._require_same_frame(other)
        return np.linalg.norm(self.values - other.values, axis=-1)

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.values, axis=-1)

    def to_render(self) -> np.ndarray:
        """Narrow to float32 for the GPU, guarding precision."""
        return self.frame.to_render(self)

    def quantity(self) -> u.Quantity:
        """The position as an astropy Quantity, for reporting."""
        return self.values * self.kind.unit

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "FramedPosition({0}, {1} in {2})".format(
            np.array2string(self.values, precision=6), self.kind.unit_label, self.frame.describe()
        )


@dataclass(frozen=True, eq=False)
class ReferenceFrame:
    """Base frame: an origin in absolute parsecs plus a unit.

    The origin is always stored in parsecs and float64, whatever the frame's
    display unit, so conversions between frames go through one well-defined
    absolute space rather than a chain of ad-hoc factors.
    """

    kind: FrameKind
    origin_pc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_pc", np.asarray(self.origin_pc, dtype=np.float64).reshape(3)
        )

    # -- unit bridges ----------------------------------------------------
    @property
    def unit(self) -> u.UnitBase:
        return self.kind.unit

    @property
    def unit_label(self) -> str:
        return self.kind.unit_label

    @property
    def _pc_per_unit(self) -> float:
        return float((1.0 * self.unit).to_value(u.pc))

    @property
    def _unit_per_pc(self) -> float:
        return float((1.0 * u.pc).to_value(self.unit))

    # -- construction ----------------------------------------------------
    def at(self, values) -> FramedPosition:
        """Label raw numbers as a position in this frame."""
        return FramedPosition(values, self)

    def origin(self) -> FramedPosition:
        """The frame's own origin, which is the zero vector by definition."""
        return FramedPosition(np.zeros(3), self)

    # -- absolute space --------------------------------------------------
    def to_absolute_pc(self, position: FramedPosition) -> np.ndarray:
        """Convert a position in this frame to absolute parsecs, float64."""
        if position.frame is not self and position.frame != self:
            raise FrameMismatchError("position does not belong to this frame")
        return position.values * self._pc_per_unit + self.origin_pc

    def from_absolute_pc(self, absolute_pc) -> FramedPosition:
        """Convert absolute parsecs into a position in this frame."""
        absolute = np.asarray(absolute_pc, dtype=np.float64)
        return FramedPosition((absolute - self.origin_pc) * self._unit_per_pc, self)

    def convert(self, position: FramedPosition) -> FramedPosition:
        """Re-express a position from another frame in this one.

        This is the only route between frames, and it goes through absolute
        parsecs in float64, so no conversion factor is ever applied twice or
        applied to the wrong quantity.
        """
        absolute = position.frame.to_absolute_pc(position)
        return self.from_absolute_pc(absolute)

    # -- the GPU boundary ------------------------------------------------
    def to_render(self, position: FramedPosition) -> np.ndarray:
        """Narrow to float32. The single place precision is reduced."""
        if position.frame is not self and position.frame != self:
            raise FrameMismatchError("position does not belong to this frame")

        values = position.values
        finite = values[np.isfinite(values)]
        if finite.size and float(np.max(np.abs(finite))) > FLOAT32_SAFE_MAGNITUDE:
            raise PrecisionError(
                "coordinate magnitude {0:.3g} {1} exceeds the float32 safe range; "
                "rebase the origin or switch to a finer frame".format(
                    float(np.max(np.abs(finite))), self.unit_label
                )
            )
        if finite.size and not np.all(np.isfinite(values)):
            raise PrecisionError("refusing to render a non-finite coordinate")
        return np.ascontiguousarray(values, dtype=np.float64).astype(np.float32)

    # -- the radius within which this frame may be the active one --------
    @property
    def safe_radius(self) -> float:
        """Largest coordinate this frame can render, in its own unit."""
        return FLOAT32_SAFE_MAGNITUDE

    @property
    def engage_radius(self) -> float:
        """Radius inside which this frame may become active, in its own unit.

        Derived from the float32 limit rather than chosen: a frame becomes
        usable exactly when its coordinates start fitting in the buffer the
        GPU will receive, with an order of magnitude of headroom for the
        geometry drawn around the camera.
        """
        return self.safe_radius / FRAME_ENGAGE_MARGIN

    @property
    def engage_radius_pc(self) -> float:
        """The same radius expressed in absolute parsecs."""
        return self.engage_radius * self._pc_per_unit

    def contains(self, absolute_pc) -> bool:
        """True when a point in absolute parsecs is inside the engage radius."""
        offset = np.asarray(absolute_pc, dtype=np.float64).reshape(3) - self.origin_pc
        return bool(np.linalg.norm(offset) <= self.engage_radius_pc)

    def describe(self) -> str:
        label = self.name or self.kind.value.lower()
        return "{0} frame [{1}]".format(label, self.unit_label)

    # Frames compare by value, but the generated __eq__ would compare the
    # origin arrays with `==` and raise on the ambiguous truth value.
    def __eq__(self, other) -> bool:
        if not isinstance(other, ReferenceFrame):
            return NotImplemented
        return (
            self.kind is other.kind
            and self.name == other.name
            and np.array_equal(self.origin_pc, other.origin_pc)
        )

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash((self.kind, self.name, self.origin_pc.tobytes()))


@dataclass(frozen=True, eq=False)
class UniverseFrame(ReferenceFrame):
    """Galaxy-scale frame: unit parsec, origin the Sun or a floating origin.

    Used for the stellar-neighbourhood view. Not required by the
    one-system vertical slice, but defined here so that entering a system is
    a frame *transition* rather than a scale factor.
    """

    def __init__(self, origin_pc=None, name: str = "Sun"):
        super().__init__(
            kind=FrameKind.UNIVERSE,
            origin_pc=np.zeros(3) if origin_pc is None else origin_pc,
            name=name,
        )

    def rebase_to(self, position: FramedPosition) -> "UniverseFrame":
        """Move the floating origin onto ``position`` (roadmap 9.4)."""
        return UniverseFrame(self.to_absolute_pc(position), name=self.name)

    def enter_system(self, host_position: FramedPosition, host_name: str = "") -> "SystemFrame":
        """Descend into a host system; the star becomes the origin."""
        return SystemFrame(self.to_absolute_pc(host_position), host_name=host_name)


@dataclass(frozen=True, eq=False)
class SystemFrame(ReferenceFrame):
    """Host-system frame: unit AU, origin the host star.

    This is the frame the vertical slice renders in. The star is at
    ``(0, 0, 0)`` by construction, and a planet's position is the orbital
    vector in AU with no conversion applied to it at all.
    """

    host_name: str = ""

    def __init__(self, origin_pc=None, host_name: str = ""):
        super().__init__(
            kind=FrameKind.SYSTEM,
            origin_pc=np.zeros(3) if origin_pc is None else origin_pc,
            name=host_name,
        )
        object.__setattr__(self, "host_name", host_name)

    @classmethod
    def for_host(cls, host_name: str, host_position_pc=None) -> "SystemFrame":
        """Build the frame for a named host star.

        ``host_position_pc`` may be omitted: a system viewed on its own does
        not need to know where it is in the galaxy, and requiring a distance
        would exclude every host whose parallax is unusable.
        """
        return cls(origin_pc=host_position_pc, host_name=host_name)

    def star_position(self) -> FramedPosition:
        """The host star, which is the origin: exactly ``(0, 0, 0)`` AU."""
        return self.origin()

    def place_planet(self, orbital_vector_au) -> FramedPosition:
        """Label an orbital vector from the physics layer as a position.

        No arithmetic happens here. The orbital vector is already
        star-centred and already in AU, which is precisely why the system
        frame uses AU: the conversion that used to be wrong is now absent.
        """
        return FramedPosition(orbital_vector_au, self)

    def to_universe(self, universe: UniverseFrame, position: FramedPosition) -> FramedPosition:
        """Lift a system-frame position into the galaxy frame."""
        return universe.convert(position)

    def enter_planet(self, planet_position: FramedPosition) -> "PlanetFrame":
        """Descend further; the planet becomes the origin."""
        return PlanetFrame(self.to_absolute_pc(planet_position), name=self.host_name)


@dataclass(frozen=True, eq=False)
class PlanetFrame(ReferenceFrame):
    """Planet-local frame: unit kilometre, origin the planet.

    Defined for completeness of the hierarchy (roadmap 9.3). The vertical
    slice does not use it yet.
    """

    def __init__(self, origin_pc=None, name: str = ""):
        super().__init__(
            kind=FrameKind.PLANET,
            origin_pc=np.zeros(3) if origin_pc is None else origin_pc,
            name=name,
        )
