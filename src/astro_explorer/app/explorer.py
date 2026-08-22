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

__all__ = ["ViewState", "Selection", "Explorer", "UniverseTarget"]


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

    Identity is the catalogue name, never an array index or a render handle,
    so it survives a level-of-detail change, a rebuild of the scene, or a
    frame transition.
    """

    identifier: str
    kind: str = "star"
    host: str = ""

    @property
    def is_star(self) -> bool:
        return self.kind == "star"

    @property
    def is_planet(self) -> bool:
        return self.kind == "planet"


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

    #: Absolute camera position in parsecs, float64. The single source of
    #: truth for where the viewer is; the camera's own target/distance are
    #: expressed in whichever frame is active.
    _camera_pc: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 40.0]), repr=False
    )

    def __post_init__(self) -> None:
        self._camera_pc = np.asarray(self._camera_pc, dtype=np.float64).reshape(3)

    # -- where we are ----------------------------------------------------
    @property
    def camera_pc(self) -> np.ndarray:
        """Absolute camera position in parsecs."""
        return self._camera_pc

    @property
    def active_frame(self) -> ReferenceFrame:
        """The finest frame whose engage radius contains the camera.

        A pure function of position. There is no separate mode flag that
        could disagree with where the camera actually is.
        """
        if self.system is not None and self.system.contains(self._camera_pc):
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
        return self.active_frame.from_absolute_pc(self._camera_pc)

    def distance_to_system_pc(self) -> float | None:
        """How far the camera is from the focused host, in parsecs."""
        if self.system is None:
            return None
        return float(np.linalg.norm(self._camera_pc - self.system.origin_pc))

    # -- moving ----------------------------------------------------------
    def move_to_pc(self, absolute_pc) -> None:
        """Place the camera at an absolute position in parsecs."""
        self._camera_pc = np.asarray(absolute_pc, dtype=np.float64).reshape(3)
        self._sync_camera()

    def focus(self, host_name: str, records: list, star) -> None:
        """Make a host system the transition target, without moving.

        The system frame exists from this moment, but it only becomes the
        *active* frame once the camera is inside its engage radius.
        """
        position = None
        if star is not None and star.position is not None and star.position.has_distance:
            position = star.position.cartesian_pc()
        if position is None:
            target = self.target(host_name)
            position = target.position_pc if target else np.zeros(3)

        self.system = SystemFrame.for_host(host_name, position)
        self.system_records = list(records)
        self.system_star = star

    def approach(self, fraction: float) -> None:
        """Move a fraction of the way from here to the focused host.

        ``fraction = 1`` lands at the host itself. Interpolation happens in
        absolute parsecs and float64, so no intermediate position is ever
        expressed in a frame that cannot represent it.
        """
        if self.system is None:
            raise ValueError("no system is focused; call focus() first")
        start = self._camera_pc
        end = self.system.origin_pc
        self.move_to_pc(start + (end - start) * float(np.clip(fraction, 0.0, 1.0)))

    def enter_system(self, *, standoff_au: float = 3.0) -> None:
        """Fly in until the system frame is active.

        Lands ``standoff_au`` from the host so the camera is inside the
        engage radius by a wide margin and the system fills the view.
        """
        if self.system is None:
            raise ValueError("no system is focused; call focus() first")
        offset = self._camera_pc - self.system.origin_pc
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        standoff_pc = standoff_au * float((1.0 * u.au).to_value(u.pc))
        self.move_to_pc(self.system.origin_pc + direction * standoff_pc)

    def leave_system(self, *, distance_pc: float = 20.0) -> None:
        """Back out until the universe frame is active again."""
        if self.system is None:
            return
        offset = self._camera_pc - self.system.origin_pc
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

        start = self._camera_pc.copy()
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
        if self.system is not None:
            return self.system.origin_pc
        return self.universe.origin_pc

    def _sync_camera(self) -> None:
        """Re-express the camera in the active frame after a move.

        The target is the focused system, which is exactly the origin once
        the system frame is active, so the same expression serves both
        views.
        """
        frame = self.active_frame
        local = frame.from_absolute_pc(self._camera_pc).values
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

    def scene(self, time_bjd: float | None = None) -> SceneDescription:
        """The scene for the current view state."""
        if self.view is ViewState.SYSTEM:
            return self._system_scene(time_bjd)
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
        if self.system is not None:
            distance = self.distance_to_system_pc()
            scene.annotations.append(
                "Focused on {0}: {1:.3f} pc away; the system frame engages "
                "within {2:.3f} pc.".format(
                    self.system.host_name, distance, self.system.engage_radius_pc
                )
            )
        return scene

    def _system_scene(self, time_bjd: float | None) -> SceneDescription:
        anomalies = {}
        if time_bjd is not None and self.system_records:
            from .vertical_slice import SystemSlice

            slice_ = SystemSlice(
                frame=self.system, star=self.system_star, planets=self.system_records
            )
            anomalies = slice_.mean_anomalies(time_bjd)

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
            self.selection = None
            return None

        host = self.system.host_name if self.system is not None else ""
        self.selection = Selection(result.identifier, result.kind, host)
        return self.selection

    def select(self, identifier: str, kind: str = "star") -> Selection:
        """Select by name, e.g. from a search box rather than a click."""
        host = self.system.host_name if self.system is not None else ""
        self.selection = Selection(identifier, kind, host)
        return self.selection

    @property
    def selected_identifier(self) -> str | None:
        return self.selection.identifier if self.selection else None

    # -- labels ----------------------------------------------------------
    def labels(self, scene: SceneDescription, width: int, height: int):
        """Screen-space label placements, selection first.

        The selected body's label is passed as a priority so decluttering
        can never drop it. Nothing here writes to the scene.
        """
        return scene.project_labels(
            self.camera, width, height, priority={self.selected_identifier} - {None}
        )

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
        if self.system is not None:
            distance = self.distance_to_system_pc()
            lines.append(
                "Focused system:    {0} at {1:.4g} pc (engages within {2:.4g} pc)".format(
                    self.system.host_name, distance, self.system.engage_radius_pc
                )
            )
        if self.selection is not None:
            lines.append(
                "Selected:          {0} ({1})".format(
                    self.selection.identifier, self.selection.kind
                )
            )
        else:
            lines.append("Selected:          nothing")
        return lines
