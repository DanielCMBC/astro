"""The renderer's input contract (roadmap section 6).

The golden architecture rule: *the renderer must never own scientific truth*.
This module defines the only types the renderer accepts.  They contain
positions that have already been computed, in display units, plus material
identifiers.  There is deliberately no field for eccentricity, semimajor
axis, anomaly, physical distance, parameter status or molecular detection -
the renderer cannot decide any of those because it is never told them.

:class:`SceneDescription` is produced by the application state layer and
consumed by a GL backend.  Building a scene requires no OpenGL context, so
the contract can be tested headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

__all__ = [
    "PROGRAMS",
    "RenderStar",
    "RenderPlanet",
    "RenderOrbit",
    "SceneDescription",
    "ShaderLibrary",
    "SHADER_DIR",
]

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _as_float32_position(value) -> np.ndarray:
    position = np.asarray(value, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(position)):
        raise ValueError(
            "render positions must be finite; an unknown parameter must be "
            "resolved or excluded before it reaches the renderer"
        )
    return position


@dataclass(frozen=True)
class RenderStar:
    """A star, ready to draw."""

    identifier: str
    position_local: np.ndarray
    radius_display: float
    color: tuple[float, float, float]
    material_id: str = "star"
    #: Set only when the display colour was derived from a temperature.
    temperature_k: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_local", _as_float32_position(self.position_local))


@dataclass(frozen=True)
class RenderPlanet:
    """A planet, ready to draw.

    Note what is absent: no orbital elements, no anomaly, no provenance.
    The position was computed by the physics layer.
    """

    identifier: str
    position_local: np.ndarray
    radius_display: float
    material_id: str = "rocky"
    base_color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    emissive: float = 0.0
    banding: float = 0.0
    roughness: float = 0.8
    label: str = ""
    lod: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_local", _as_float32_position(self.position_local))


@dataclass(frozen=True)
class RenderOrbit:
    """A precomputed orbit path as a batched line strip."""

    identifier: str
    points_local: np.ndarray  # (N, 3) float32
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.35)
    #: True when any element used to draw this path was an assumption.  The
    #: renderer uses it only to pick a dashed style; the *reason* stays in
    #: the UI layer.
    dashed: bool = False

    def __post_init__(self) -> None:
        points = np.asarray(self.points_local, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("orbit points must be an (N, 3) array")
        object.__setattr__(self, "points_local", np.ascontiguousarray(points))

    @property
    def vertex_count(self) -> int:
        return int(self.points_local.shape[0])


@dataclass
class SceneDescription:
    """Everything one frame needs, in display units of the active frame."""

    stars: list[RenderStar] = field(default_factory=list)
    planets: list[RenderPlanet] = field(default_factory=list)
    orbits: list[RenderOrbit] = field(default_factory=list)
    #: Name of the active frame's unit, for the on-screen scale bar.
    unit_label: str = "AU"
    #: Free-text notes the UI overlays, e.g. which values were assumed.
    annotations: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.stars or self.planets or self.orbits)

    def bounding_radius(self) -> float:
        """Largest distance from the origin, for framing the camera."""
        points = [s.position_local for s in self.stars]
        points += [p.position_local for p in self.planets]
        for orbit in self.orbits:
            if orbit.vertex_count:
                points.append(orbit.points_local[np.argmax(np.linalg.norm(orbit.points_local, axis=1))])
        if not points:
            return 1.0
        return float(max(np.linalg.norm(np.asarray(p, dtype=np.float64)) for p in points)) or 1.0

    def instance_buffer(self) -> np.ndarray:
        """Per-instance attributes for one instanced draw of all planets.

        Layout per instance: ``position(3) radius(1) color(3) emissive(1)``.
        Batching this way is what roadmap section 25 means by avoiding
        per-object Python draw overhead.
        """
        if not self.planets:
            return np.zeros((0, 8), dtype=np.float32)
        rows = [
            np.concatenate(
                [
                    planet.position_local,
                    [planet.radius_display],
                    planet.base_color,
                    [planet.emissive],
                ]
            )
            for planet in self.planets
        ]
        return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)

    def assign_lod(self, camera, viewport_height: int) -> None:
        """Choose a sphere subdivision per body from its projected size.

        Per-system LOD (review section 15): a tightly packed system seen
        from outside has inner planets a few pixels across and outer ones
        larger, and there is no reason to spend 5120 triangles on a body
        covering four pixels.
        """
        from .mesh import lod_for_distance

        eye = np.asarray(camera.position, dtype=np.float64)
        for index, planet in enumerate(self.planets):
            distance = float(np.linalg.norm(planet.position_local - eye))
            level = lod_for_distance(
                planet.radius_display, distance, viewport_height, camera.fov_y_rad
            )
            self.planets[index] = replace(planet, lod=int(level))

    def project_labels(
        self, camera, width: int, height: int, *, margin: int = 4, priority=()
    ):
        """Screen positions for body labels (review section 15).

        Returns ``(identifier, x, y, depth, screen_radius)`` for every
        labelled body that is in front of the camera and inside the
        viewport. Placement is computed here rather than in a shader so any
        UI - a GL text pass, a Qt overlay, or PIL in the demo - can draw
        them the same way.

        ``screen_radius`` lets a caller push the text clear of the body
        instead of writing across it, which matters for a host star that
        fills a fair part of the frame.

        ``priority`` names identifiers that must survive decluttering - in
        practice the current selection, which the user must always be able
        to see. They are returned first.

        This method is strictly read-only. Label placement must never touch
        the coordinates that positioned the bodies, so nothing here writes
        back to the scene.
        """
        view_projection = camera.view_projection()
        placements = []

        for body in list(self.stars) + list(self.planets):
            if not body.label:
                continue
            clip = view_projection @ np.append(
                np.asarray(body.position_local, dtype=np.float64), 1.0
            )
            if clip[3] <= 0.0:
                continue  # behind the camera
            ndc = clip[:3] / clip[3]
            if not (-1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0):
                continue
            x = (ndc[0] * 0.5 + 0.5) * width
            y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
            if not (margin <= x <= width - margin and margin <= y <= height - margin):
                continue

            # Projected radius in pixels: r / (d tan(fov/2)) * (height/2).
            radius = getattr(body, "radius_display", 0.0)
            screen_radius = (
                radius / max(float(clip[3]), 1e-12)
                / np.tan(0.5 * camera.fov_y_rad)
                * (height * 0.5)
            )
            placements.append(
                (body.label, float(x), float(y), float(clip[3]), float(screen_radius))
            )

        # Priority first, then nearest, so a collision resolver keeps the
        # selection and drops the far label.
        wanted = set(priority)
        placements.sort(key=lambda item: (item[0] not in wanted, item[3]))
        return placements

    def star_instance_buffer(self) -> np.ndarray:
        """Per-instance attributes for stars: ``position(3) radius(1) color(3)``."""
        if not self.stars:
            return np.zeros((0, 7), dtype=np.float32)
        rows = [
            np.concatenate([star.position_local, [star.radius_display], star.color])
            for star in self.stars
        ]
        return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


#: Program name -> (vertex source, fragment source).  Several materials share
#: one vertex stage, so programs are declared rather than inferred from
#: filenames.
PROGRAMS = {
    "star": ("star.vert", "star.frag"),
    "rocky": ("planet.vert", "rocky.frag"),
    "gas_giant": ("planet.vert", "gas_giant.frag"),
    "atmosphere": ("planet_atmosphere.vert", "atmosphere.frag"),
    "orbit": ("orbit.vert", "orbit.frag"),
}


class ShaderLibrary:
    """Loads GLSL sources from disk.

    Kept separate from any GL backend so shader sources can be validated in
    tests without a context.
    """

    def __init__(self, directory: Path | str = SHADER_DIR):
        self.directory = Path(directory)

    def available(self) -> list[str]:
        """Program names whose vertex and fragment sources both exist."""
        if not self.directory.is_dir():
            return []
        return sorted(
            name
            for name, (vert, frag) in PROGRAMS.items()
            if (self.directory / vert).exists() and (self.directory / frag).exists()
        )

    def source(self, filename: str) -> str:
        path = self.directory / filename
        if not path.exists():
            raise FileNotFoundError("shader {0} not found".format(path))
        return path.read_text(encoding="utf-8")

    def program_sources(self, name: str) -> dict[str, str]:
        """Sources for a declared program, ready for ``ctx.program(**sources)``."""
        if name not in PROGRAMS:
            raise KeyError(
                "unknown shader program {0!r}; declared programs are {1}".format(
                    name, ", ".join(sorted(PROGRAMS))
                )
            )
        vertex_file, fragment_file = PROGRAMS[name]
        return {
            "vertex_shader": self.source(vertex_file),
            "fragment_shader": self.source(fragment_file),
        }
