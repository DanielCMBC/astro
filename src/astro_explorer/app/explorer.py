"""Navigation and selection: the explorer state machine.

Review section 10. Three things that have to work together, because each one
breaks the others if done alone:

* **frame transitions** - flying from the stellar neighbourhood into a host
  system, without a coordinate ever losing precision on the way;
* **selection** - picking a body that is actually visible, and keeping that
  identity while its level of detail changes underneath;
* **labels** - naming what is on screen without ever touching the numbers
  that placed it there.

The interaction model comes from the legacy prototype, which was right about
what the program should feel like: fly through hosts, select a star, enter
its system, inspect planets. None of its implementation is reused.

Which frame is active
---------------------
Not a mode the user toggles and not a flag that can drift out of step with
the camera. It is a **pure function of where the camera is**: the finest
frame whose ``engage_radius`` contains it. That radius is itself derived
from the float32 limit rather than picked, so "close enough to enter the
system" and "close enough for AU coordinates to survive the GPU" are the
same statement.

A transition is therefore just camera movement. There is no window in which
the camera has moved but the frame has not, which is the state that would
produce a precision failure.

Where the camera is
-------------------
Explorer B review section 5. The camera is held in exactly one of two
states, never in a hybrid of them:

* ``_camera_absolute_pc`` - an absolute position in parsecs, float64, set
  whenever the camera is anywhere at all;
* ``_camera_local`` - a :class:`FramedPosition` in an unlocated frame's own
  unit, set exactly when the camera is inside a detached system.

They are mutually exclusive, and the parsec field's invariant - "this is an
absolute galactic position" - therefore holds unconditionally. An earlier
version stored the detached camera's AU offset in the parsec field, scaled
so the arithmetic worked out; it produced correct pictures while quietly
making the field mean two different things. A detached camera is really a
``FramedPosition([0, 0, 3], SystemFrame[AU])``, so that is what it is now.

Every absolute-space operation - :meth:`Explorer.move_to_pc`,
:meth:`Explorer.approach`, :meth:`Explorer.enter_system`,
:meth:`Explorer.leave_system`, :meth:`Explorer.path_to_system` and reading
:attr:`Explorer.camera_pc` - refuses while detached, through one shared
guard rather than a check repeated at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import astropy.units as u
import numpy as np

from ..coordinates.system_frame import (
    FramedPosition,
    ReferenceFrame,
    SystemFrame,
    UniverseFrame,
)
from ..rendering.camera import Camera
from ..rendering.picking import PickResult, pick
from ..rendering.renderer import RenderStar, SceneDescription
from ..rendering.scene_builder import build_frame_scene

__all__ = [
    "ViewState",
    "Selection",
    "Explorer",
    "UniverseTarget",
    "UnknownSystemPositionError",
]


class UnknownSystemPositionError(ValueError):
    """Raised when a system without a known galactic position is flown to.

    Review section 5. A host whose absolute position is unknown must not
    silently acquire one, and the most tempting wrong answer is
    ``np.zeros(3)`` - which does not mean "unknown", it means "at the
    Sun". Such a system can still be opened *detached*, where the star is
    the origin of its own frame and no absolute claim is made at all.
    """


class ViewState(str, Enum):
    """What the explorer is currently showing."""

    UNIVERSE = "UNIVERSE"
    """The stellar neighbourhood, in parsecs."""

    SYSTEM = "SYSTEM"
    """Inside a host system, in AU."""

    @property
    def label(self) -> str:
        return {
            ViewState.UNIVERSE: "stellar neighbourhood",
            ViewState.SYSTEM: "host system",
        }[self]


@dataclass(frozen=True)
class Selection:
    """What the user has selected.

    Identity is a stable ``entity_id`` - ``planet:nasa:HD_80606_b`` - not an
    array index, a render handle, or the display name. Names and aliases
    change between catalogue releases; a selection must not.

    ``generation`` increments on every selection change, so a background
    task that finishes late can compare its token against the current one
    and discard itself instead of overwriting a newer panel. That is the
    legacy prototype's async race, made impossible.
    """

    entity_id: str
    display_name: str = ""
    kind: str = "star"
    host_id: str = ""
    generation: int = 0

    @property
    def is_star(self) -> bool:
        return self.kind == "star"

    @property
    def is_planet(self) -> bool:
        return self.kind == "planet"

    @property
    def identifier(self) -> str:
        """The stable key. Kept as an alias for render-primitive matching."""
        return self.entity_id

    @property
    def label(self) -> str:
        """What a human should see - never used as identity."""
        return self.display_name or self.entity_id


@dataclass(frozen=True)
class UniverseTarget:
    """A host star as it appears in the neighbourhood view."""

    name: str
    position_pc: np.ndarray
    temperature_k: float | None = None
    radius_solar: float | None = None
    distance_pc: float | None = None
    planet_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_pc", np.asarray(self.position_pc, dtype=np.float64).reshape(3)
        )


#: Stars are points at neighbourhood scale; a fixed display radius in parsecs
#: keeps them visible without pretending to be a physical size.
UNIVERSE_STAR_DISPLAY_PC = 0.35

#: Where a detached camera sits, in the system frame's own unit. It is an
#: AU offset in a SystemFrame, not a parsec value scaled to look like one.
DETACHED_STANDOFF_AU = 3.0

#: Where the camera lands when a system is focused from a state that had no
#: absolute position of its own.
DEFAULT_STANDOFF_PC = 20.0


@dataclass
class Explorer:
    """Camera, active frame and selection, kept mutually consistent."""

    universe: UniverseFrame = field(default_factory=UniverseFrame)
    targets: list[UniverseTarget] = field(default_factory=list)
    camera: Camera = field(default_factory=lambda: Camera(distance=40.0))
    system: SystemFrame | None = None
    system_records: list = field(default_factory=list)
    system_star: object | None = None
    selection: Selection | None = None

    #: True when the focused system has no known galactic position and is
    #: being inspected locally. Mirrors ``system.located``; the frame is the
    #: authority, this is the convenient name for it.
    detached: bool = False

    #: Incremented on every selection change. A late async result carrying
    #: an older token is stale and must be discarded.
    generation: int = 0

    #: Absolute camera position in parsecs, float64. The single source of
    #: truth for where the viewer is while the camera is anywhere at all;
    #: the camera's own target/distance are expressed in whichever frame is
    #: active. ``None`` exactly when the camera is in a detached system,
    #: because then there is no absolute position to hold.
    _camera_absolute_pc: np.ndarray | None = field(
        default_factory=lambda: np.array([0.0, 0.0, 40.0]), repr=False
    )

    #: The camera inside an unlocated frame, in that frame's own unit.
    #: ``None`` exactly when :attr:`_camera_absolute_pc` is set. Review
    #: section 5: a detached camera is genuinely a
    #: ``FramedPosition([0, 0, 3], SystemFrame[AU])``, and storing it as one
    #: means the parsec field's invariant - "this is an absolute position" -
    #: is never quietly violated to carry a local offset.
    _camera_local: FramedPosition | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._camera_absolute_pc is not None:
            self._camera_absolute_pc = np.asarray(
                self._camera_absolute_pc, dtype=np.float64
            ).reshape(3)

    # -- where we are ----------------------------------------------------
    @property
    def camera_pc(self) -> np.ndarray:
        """Absolute camera position in parsecs.

        Raises :class:`UnknownSystemPositionError` in a detached system,
        where the question has no answer. Use
        :attr:`camera_absolute_pc` when "nowhere" is an acceptable reply.
        """
        if self._camera_absolute_pc is None:
            raise UnknownSystemPositionError(
                "the camera is inside {0}, which has no known galactic "
                "position, so it has no absolute position either".format(
                    self.system.host_name if self.system is not None else "a detached system"
                )
            )
        return self._camera_absolute_pc

    @property
    def camera_absolute_pc(self) -> np.ndarray | None:
        """Absolute camera position in parsecs, or None while detached."""
        return self._camera_absolute_pc

    @property
    def camera_local(self) -> FramedPosition | None:
        """The camera as a local position, or None while it is located."""
        return self._camera_local

    def _require_located(self, action: str) -> None:
        """Refuse an absolute-space operation while detached.

        Review section 5: absolute navigation is not merely unhelpful in a
        detached system, it is undefined, so every entry point into it
        rejects the state structurally rather than trusting each caller.
        """
        if self.detached:
            raise UnknownSystemPositionError(
                "{0} has no absolute position; {1}".format(
                    self.system.host_name if self.system is not None else "this system",
                    action,
                )
            )

    @property
    def active_frame(self) -> ReferenceFrame:
        """The finest frame whose engage radius contains the camera.

        A pure function of position. There is no separate mode flag that
        could disagree with where the camera actually is.
        """
        # An unlocated system is not anywhere, so proximity cannot decide:
        # it is the active frame whenever it is open at all.
        if self.system is not None and not self.system.located:
            return self.system
        if (
            self._camera_absolute_pc is not None
            and self.system is not None
            and self.system.contains(self._camera_absolute_pc)
        ):
            return self.system
        return self.universe

    @property
    def view(self) -> ViewState:
        return (
            ViewState.SYSTEM
            if isinstance(self.active_frame, SystemFrame)
            else ViewState.UNIVERSE
        )

    @property
    def camera_position(self) -> FramedPosition:
        """The camera, expressed in the active frame."""
        frame = self.active_frame
        if not frame.located:
            # Already a position in this frame's own unit. Nothing is
            # converted, so nothing has to pretend to be parsecs first.
            if self._camera_local is None:
                return frame.origin()
            return self._camera_local
        return frame.from_absolute_pc(self.camera_pc)

    def distance_to_system_pc(self) -> float | None:
        """How far the camera is from the focused host, in parsecs.

        None for a detached system: its origin is a local convention, not a
        location, so a distance measured from it would be meaningless.
        """
        if self.system is None or self.detached:
            return None
        return float(np.linalg.norm(self.camera_pc - self.system.origin_pc))

    # -- moving ----------------------------------------------------------
    def move_to_pc(self, absolute_pc) -> None:
        """Place the camera at an absolute position in parsecs.

        Refused while detached: there is no absolute space to move through,
        and accepting the call would silently re-locate a system whose
        position is unknown. Use :meth:`move_to_local` instead.
        """
        self._require_located("the camera cannot be placed in absolute parsecs")
        self._camera_absolute_pc = np.asarray(absolute_pc, dtype=np.float64).reshape(3)
        self._camera_local = None
        self._sync_camera()

    def move_to_local(self, values) -> None:
        """Place the camera inside the active frame, in that frame's unit.

        The only way to move a detached camera, and the shape the eventual
        free-flight camera wants everywhere: a position that carries its
        frame rather than a bare triple whose meaning depends on state.
        """
        frame = self.active_frame
        if frame.located:
            self.move_to_pc(frame.to_absolute_pc(frame.at(values)))
            return
        self._camera_local = frame.at(values)
        self._camera_absolute_pc = None
        self._sync_camera()

    @staticmethod
    def absolute_position_of(host_name: str, star, targets=()) -> np.ndarray | None:
        """The host's absolute position in parsecs, or None if unknown.

        None is a real answer, not a failure to look hard enough. It is
        returned rather than substituted.
        """
        if star is not None and star.position is not None and star.position.has_distance:
            return star.position.cartesian_pc()
        target = next((t for t in targets if t.name == host_name), None)
        return target.position_pc if target is not None else None

    def focus(self, host_name: str, records: list, star) -> None:
        """Make a host system the navigation target, without moving.

        The system frame exists from this moment, but it only becomes the
        *active* frame once the camera is inside its engage radius.

        Raises
        ------
        UnknownSystemPositionError
            When the host has no usable absolute position. Flying to a
            system requires knowing where it is; use
            :meth:`open_detached` to inspect it locally instead.
        """
        position = self.absolute_position_of(host_name, star, self.targets)
        if position is None:
            raise UnknownSystemPositionError(
                "{0} has no usable absolute position, so it cannot be "
                "navigated to. Its parallax or catalogue distance is "
                "unavailable. Use open_detached() to inspect the system "
                "locally without claiming a galactic position.".format(host_name)
            )

        self.system = SystemFrame.for_host(host_name, position)
        self.system_records = list(records)
        self.system_star = star
        self.detached = False

        if self._camera_absolute_pc is None:
            # Coming out of a detached system: the local offset cannot be
            # lifted into absolute space - that is the whole point of an
            # unlocated frame - so the camera is placed at a plain standoff
            # from the newly focused host rather than being "converted".
            self._camera_local = None
            self._camera_absolute_pc = self.system.origin_pc + np.array(
                [0.0, 0.0, DEFAULT_STANDOFF_PC]
            )
            self._sync_camera()

    def open_detached(self, host_name: str, records: list, star) -> None:
        """Inspect a system locally, making no claim about where it is.

        The host star is the origin of its own frame - that is true of every
        system frame - but no absolute position is asserted, so the system
        does not appear in the neighbourhood view, has no distance from
        Earth or from any other star, and cannot be flown to.
        """
        self.system = SystemFrame.for_host(host_name, None)
        self.system_records = list(records)
        self.system_star = star
        self.detached = True
        assert not self.system.located

        # The camera is placed in the system's own frame, as a position in
        # AU that says which frame it belongs to. There is no absolute
        # position to place it at, none is invented, and no parsec-shaped
        # field is borrowed to carry the offset.
        self._camera_absolute_pc = None
        self._camera_local = self.system.at([0.0, 0.0, DETACHED_STANDOFF_AU])
        self._sync_camera()

    def can_navigate_to(self, host_name: str, star=None) -> bool:
        """True when the host has a position to fly to."""
        return self.absolute_position_of(host_name, star, self.targets) is not None

    def approach(self, fraction: float) -> None:
        """Move a fraction of the way from here to the focused host.

        ``fraction = 1`` lands at the host itself. Interpolation happens in
        absolute parsecs and float64, so no intermediate position is ever
        expressed in a frame that cannot represent it.
        """
        if self.system is None:
            raise ValueError("no system is focused; call focus() first")
        self._require_located("it cannot be flown to")
        start = self.camera_pc
        end = self.system.origin_pc
        self.move_to_pc(start + (end - start) * float(np.clip(fraction, 0.0, 1.0)))

    def enter_system(self, *, standoff_au: float = 3.0) -> None:
        """Fly in until the system frame is active.

        Lands ``standoff_au`` from the host so the camera is inside the
        engage radius by a wide margin and the system fills the view.
        """
        if self.system is None:
            raise ValueError("no system is focused; call focus() first")
        self._require_located("it cannot be flown to")
        offset = self.camera_pc - self.system.origin_pc
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        standoff_pc = standoff_au * float((1.0 * u.au).to_value(u.pc))
        self.move_to_pc(self.system.origin_pc + direction * standoff_pc)

    def leave_system(self, *, distance_pc: float = 20.0) -> None:
        """Back out until the universe frame is active again."""
        if self.system is None:
            return
        self._require_located("there is nothing to back away from")
        offset = self.camera_pc - self.system.origin_pc
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        self.move_to_pc(self.system.origin_pc + direction * max(distance_pc, 1.0))

    def path_to_system(
        self, steps: int = 24, *, standoff_au: float = 3.0
    ) -> list[np.ndarray]:
        """Absolute positions along an approach, for testing and animation.

        Spaced evenly in *log distance* rather than linearly, because the
        journey spans five orders of magnitude and the interesting part is
        the last thousandth of it. Linear spacing would spend every frame
        in empty space and arrive in one step.

        The path ends ``standoff_au`` from the host rather than at it, so
        the final waypoints are inside the system with it filling the view.
        """
        if self.system is None:
            raise ValueError("no system is focused")
        self._require_located("there is no path to it")

        start = self.camera_pc.copy()
        origin = self.system.origin_pc
        offset = start - origin
        distance = float(np.linalg.norm(offset))

        standoff_pc = standoff_au * float((1.0 * u.au).to_value(u.pc))
        if distance <= standoff_pc or steps < 2:
            return [start]

        direction = offset / distance
        return [origin + direction * d for d in np.geomspace(distance, standoff_pc, steps)]

    @property
    def look_at_pc(self) -> np.ndarray:
        """What the camera is aimed at, in absolute parsecs.

        The focused system when there is one - a viewer flying towards a
        star should be looking at it, not at the Sun - otherwise the
        neighbourhood origin.
        """
        if self.system is not None and not self.detached:
            return self.system.origin_pc
        return self.universe.origin_pc

    @property
    def has_absolute_position(self) -> bool:
        """True when the focused system knows where it is in the galaxy."""
        return self.system is not None and not self.detached

    def _sync_camera(self) -> None:
        """Re-express the camera in the active frame after a move.

        The target is the focused system, which is exactly the origin once
        the system frame is active, so the same expression serves both
        views.
        """
        frame = self.active_frame
        if not frame.located:
            # Nothing to convert: the camera is already a position in this
            # frame, measured from its own origin.
            local = self._camera_local
            offset = frame.origin().values if local is None else local.values
            distance = float(np.linalg.norm(offset))
            self.camera.target = np.zeros(3)
            self.camera.distance = max(distance, self.camera.min_distance)
            if distance > 0:
                self.camera.pitch = float(
                    np.arcsin(np.clip(offset[1] / distance, -1.0, 1.0))
                )
                self.camera.yaw = float(np.arctan2(offset[0], offset[2]))
            return

        local = frame.from_absolute_pc(self.camera_pc).values
        target = frame.from_absolute_pc(self.look_at_pc).values
        offset = local - target
        distance = float(np.linalg.norm(offset))

        self.camera.target = target
        self.camera.distance = max(distance, self.camera.min_distance)
        if distance > 0:
            self.camera.pitch = float(np.arcsin(np.clip(offset[1] / distance, -1.0, 1.0)))
            self.camera.yaw = float(np.arctan2(offset[0], offset[2]))

    # -- scenes ----------------------------------------------------------
    def target(self, name: str) -> UniverseTarget | None:
        return next((t for t in self.targets if t.name == name), None)

    def scene(self, time_jd: float | None = None) -> SceneDescription:
        """The scene for the current view state."""
        if self.view is ViewState.SYSTEM:
            return self._system_scene(time_jd)
        return self._universe_scene()

    def _universe_scene(self) -> SceneDescription:
        from ..assets.procedural import star_display_color

        scene = SceneDescription(unit_label="pc")
        for target in self.targets:
            local = self.universe.from_absolute_pc(target.position_pc)
            colour = star_display_color(target.temperature_k)
            scene.stars.append(
                RenderStar(
                    identifier=target.name,
                    position_local=local.to_render(),
                    radius_display=UNIVERSE_STAR_DISPLAY_PC,
                    color=colour.stylized if colour else (1.0, 0.95, 0.85),
                    temperature_k=target.temperature_k,
                    label=target.name,
                )
            )
        scene.annotations.append(
            "Stellar neighbourhood: {0} host star(s), drawn at a fixed display "
            "size that is not a physical radius.".format(len(self.targets))
        )
        if self.system is not None and not self.detached:
            distance = self.distance_to_system_pc()
            scene.annotations.append(
                "Focused on {0}: {1:.3f} pc away; the system frame engages "
                "within {2:.3f} pc.".format(
                    self.system.host_name, distance, self.system.engage_radius_pc
                )
            )
        return scene

    def _system_scene(self, time_jd: float | None) -> SceneDescription:
        anomalies = {}
        if time_jd is not None and self.system_records:
            from .vertical_slice import SystemSlice

            slice_ = SystemSlice(
                frame=self.system, star=self.system_star, planets=self.system_records
            )
            anomalies = slice_.mean_anomalies(time_jd)

        scene = build_frame_scene(
            self.system, self.system_star, self.system_records, mean_anomalies=anomalies
        )
        scene.annotations.insert(
            0,
            "Inside {0}: system frame active, {1:.4g} AU from the host.".format(
                self.system.host_name,
                float(np.linalg.norm(self.camera_position.values)),
            ),
        )
        if self.detached:
            scene.annotations.insert(
                1,
                "This system has no published galactic position. It is shown "
                "detached: the host star is the origin of its own frame, and "
                "no location relative to the Sun is claimed.",
            )
        return scene

    # -- selection -------------------------------------------------------
    def pick_at(
        self, scene: SceneDescription, x: float, y: float, width: int, height: int
    ) -> Selection | None:
        """Select whatever is under a pixel, or clear the selection.

        Delegates the geometry to :func:`astro_explorer.rendering.picking.pick`,
        which rejects anything behind the camera and takes the nearest hit
        rather than the nearest to the ray - so an occluded body is never
        chosen over the one in front of it.
        """
        result: PickResult | None = pick(scene, self.camera, x, y, width, height)
        if result is None:
            self.clear_selection()
            return None

        label = next(
            (
                body.label
                for body in list(scene.stars) + list(scene.planets)
                if body.identifier == result.identifier
            ),
            "",
        )
        return self.select(result.identifier, result.kind, display_name=label)

    def select(
        self, entity_id: str, kind: str = "star", *, display_name: str = ""
    ) -> Selection:
        """Select by stable key, e.g. from a search box rather than a click."""
        self.generation += 1
        host = ""
        if self.system_star is not None and self.system_star.entity_id is not None:
            host = str(self.system_star.entity_id)
        self.selection = Selection(
            entity_id=entity_id,
            display_name=display_name or self.display_name_for(entity_id),
            kind=kind,
            host_id=host,
            generation=self.generation,
        )
        return self.selection

    def clear_selection(self) -> None:
        self.generation += 1
        self.selection = None

    def is_current(self, selection: Selection | None) -> bool:
        """Whether an async result is still wanted.

        The guard against the legacy prototype's race: A starts, B starts,
        B finishes, A finishes later and overwrites B's panel with stale
        data. A result whose token is not the current one is dropped.
        """
        if selection is None or self.selection is None:
            return False
        return selection.generation == self.selection.generation

    def display_name_for(self, entity_id: str) -> str:
        """Resolve a stable key back to a human-readable name."""
        record = self.record_for(entity_id)
        if record is not None:
            return record.name
        if self.system_star is not None and str(self.system_star.entity_id) == entity_id:
            return self.system_star.name
        target = next((t for t in self.targets if str(t.name) == entity_id), None)
        return target.name if target else entity_id

    def record_for(self, entity_id: str):
        """The planet record behind a stable key, or None."""
        for record in self.system_records:
            if record.entity_id is not None and str(record.entity_id) == entity_id:
                return record
        return None

    @property
    def selected_identifier(self) -> str | None:
        return self.selection.entity_id if self.selection else None

    @property
    def selected_record(self):
        """The scientific record behind the selection, if it is a planet."""
        if self.selection is None:
            return None
        return self.record_for(self.selection.entity_id)

    # -- panels ----------------------------------------------------------
    def panel(self, time_jd: float | None = None):
        """A read-only information panel for the current selection.

        Returns None when nothing is selected. Building a panel never
        mutates a record, so selecting a planet cannot perturb its orbit.
        """
        from .panel import build_planet_panel, build_star_panel, build_system_panel

        if self.selection is None:
            if self.system is not None and self.system_star is not None:
                return build_system_panel(
                    self.system.host_name,
                    self.system_star,
                    self.system_records,
                    generation=self.generation,
                )
            return None

        record = self.selected_record
        if record is not None:
            phase = None
            if time_jd is not None:
                phase = record.elements.phase_at(time_jd, allow_assumed=True)
            return build_planet_panel(record, phase, generation=self.generation)

        if self.system_star is not None and (
            str(self.system_star.entity_id) == self.selection.entity_id
        ):
            return build_star_panel(self.system_star, generation=self.generation)

        return None

    # -- labels ----------------------------------------------------------
    def labels(self, scene: SceneDescription, width: int, height: int):
        """Screen-space label placements, selection first.

        The selected body's label is passed as a priority so decluttering
        can never drop it. Nothing here writes to the scene.
        """
        priority = {self.selected_identifier} - {None}
        return scene.project_labels(self.camera, width, height, priority=priority)

    # -- reporting -------------------------------------------------------
    def describe(self) -> list[str]:
        frame = self.active_frame
        lines = [
            "View:              {0} ({1})".format(self.view.value, self.view.label),
            "Active frame:      {0}".format(frame.describe()),
            "Camera:            {0:.6g} {1} from the frame origin".format(
                float(np.linalg.norm(self.camera_position.values)), frame.unit_label
            ),
        ]
        if self.system is not None and self.detached:
            lines.append(
                "Focused system:    {0} (detached: no known galactic position, "
                "so no distance and no route)".format(self.system.host_name)
            )
        elif self.system is not None:
            distance = self.distance_to_system_pc()
            lines.append(
                "Focused system:    {0} at {1:.4g} pc (engages within {2:.4g} pc)".format(
                    self.system.host_name, distance, self.system.engage_radius_pc
                )
            )
        if self.selection is not None:
            lines.append(
                "Selected:          {0} [{1}] ({2}, generation {3})".format(
                    self.selection.label,
                    self.selection.entity_id,
                    self.selection.kind,
                    self.selection.generation,
                )
            )
        else:
            lines.append("Selected:          nothing")
        return lines
