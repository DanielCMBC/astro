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
from enum import Enum
from pathlib import Path

import numpy as np

__all__ = [
    "PROGRAMS",
    "GuideStyle",
    "RenderStar",
    "RenderPlanet",
    "RenderOrbit",
    "RenderZone",
    "RenderGuide",
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


def _as_float32_ring(value, which: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("{0} ring points must be an (N, 3) array".format(which))
    if points.shape[0] < 3:
        raise ValueError("{0} ring needs at least 3 points to bound a region".format(which))
    if not np.all(np.isfinite(points)):
        raise ValueError(
            "zone rings must be finite; an unknown boundary must be excluded "
            "before it reaches the renderer"
        )
    return np.ascontiguousarray(points)


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


@dataclass(frozen=True)
class RenderZone:
    """A flat annular region, ready to draw.

    Used for the habitable-zone overlay. Both rings arrive already sampled
    in the active frame's display units, so the renderer draws a band
    between two closed loops and is told nothing about what the band means:
    no radii in AU, no luminosity, no temperature, no model name, no
    provenance. Whether the region exists at all was decided upstream, as
    was whether the flat band is a cross-section of something rounder.

    The two rings must have the same vertex count, because the band is
    built by pairing them index for index.

    Both colours are consumed: the fill spans the band, and ``edge_color``
    draws the two boundary loops as lines. The edges exist because the
    boundaries are the scientific content - the fill is only what lies
    between them - so a soft wash with no visible limit understates how
    sharp the model's statement is.
    """

    identifier: str
    inner_points_local: np.ndarray  # (N, 3) float32
    outer_points_local: np.ndarray  # (N, 3) float32
    #: Fill colour. Alpha is a display choice and never a scientific one.
    color: tuple[float, float, float, float] = (0.36, 0.78, 0.52, 0.13)
    #: Colour of the two boundary loops, drawn as lines after the fill.
    edge_color: tuple[float, float, float, float] = (0.45, 0.9, 0.62, 0.5)
    label: str = ""

    def __post_init__(self) -> None:
        inner = _as_float32_ring(self.inner_points_local, "inner")
        outer = _as_float32_ring(self.outer_points_local, "outer")
        if inner.shape[0] != outer.shape[0]:
            raise ValueError(
                "a zone's two rings must have the same vertex count; got "
                "{0} inner and {1} outer".format(inner.shape[0], outer.shape[0])
            )
        object.__setattr__(self, "inner_points_local", inner)
        object.__setattr__(self, "outer_points_local", outer)

    @property
    def vertex_count(self) -> int:
        """Vertices per ring, not the total."""
        return int(self.inner_points_local.shape[0])


class GuideStyle(str, Enum):
    """How a guide line is stroked.

    Two visual states, and only two, because the distinction they carry has
    to survive being seen at a glance and in a still image: a solid line is
    a statement about the sky, a dashed one is not.

    What made a particular guide dashed - an element nobody published, a
    convention the catalogue never stated - is not in here. The renderer
    strokes what it is given; the reason is words, and words belong to the
    UI.
    """

    SOLID = "SOLID"
    DASHED = "DASHED"

    @property
    def is_dashed(self) -> bool:
        return self is GuideStyle.DASHED


@dataclass(frozen=True)
class RenderGuide:
    """A finished orientation guide: a polyline, a stroke style, a label.

    Used for the orbital-orientation overlays - the orbit plane, the orbit
    normal, the line of nodes, the periapsis direction, the inclination
    indicator. Every one of them arrives as points already computed by the
    physics layer from the *same* rotation the propagator uses.

    Note what is absent, and why it matters more here than anywhere else:
    there is no inclination, no ``omega``, no ``Omega``, no eccentricity.
    An overlay that re-read those angles would be a second interpretation of
    the catalogue's conventions, free to disagree with the orbit it is drawn
    against - and it would disagree silently, since both would look
    plausible. So the renderer is given vectors, not angles.

    ``points_local`` is a polyline: consecutive points are joined, and a
    closed ring is closed by repeating its first point at the end. An arrow
    is one polyline too - shaft out to the tip, then back along each barb -
    so a guide is always exactly one primitive.
    """

    identifier: str
    points_local: np.ndarray  # (N, 3) float32
    style: GuideStyle = GuideStyle.SOLID
    color: tuple[float, float, float, float] = (0.72, 0.78, 0.92, 0.85)
    label: str = ""

    def __post_init__(self) -> None:
        points = np.asarray(self.points_local, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("guide points must be an (N, 3) array")
        if points.shape[0] < 2:
            raise ValueError("a guide needs at least 2 points to be a line")
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "guide points must be finite; an unknown orientation must be "
                "resolved or excluded before it reaches the renderer"
            )
        object.__setattr__(self, "points_local", np.ascontiguousarray(points))
        object.__setattr__(self, "style", GuideStyle(self.style))

    @property
    def vertex_count(self) -> int:
        return int(self.points_local.shape[0])


@dataclass
class SceneDescription:
    """Everything one frame needs, in display units of the active frame."""

    stars: list[RenderStar] = field(default_factory=list)
    planets: list[RenderPlanet] = field(default_factory=list)
    orbits: list[RenderOrbit] = field(default_factory=list)
    #: Scientific regions drawn as flat bands, e.g. the habitable zone.
    zones: list[RenderZone] = field(default_factory=list)
    #: Orientation guides for the selected orbit, e.g. the line of nodes.
    guides: list[RenderGuide] = field(default_factory=list)
    #: Name of the active frame's unit, for the on-screen scale bar.
    unit_label: str = "AU"
    #: Free-text notes the UI overlays, e.g. which values were assumed.
    annotations: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.stars or self.planets or self.orbits or self.zones or self.guides
        )

    def bounding_radius(self) -> float:
        """Largest distance from the origin, for framing the camera."""
        points = [s.position_local for s in self.stars]
        points += [p.position_local for p in self.planets]
        for orbit in self.orbits:
            if orbit.vertex_count:
                points.append(orbit.points_local[np.argmax(np.linalg.norm(orbit.points_local, axis=1))])
        for zone in self.zones:
            ring = zone.outer_points_local
            points.append(ring[np.argmax(np.linalg.norm(ring, axis=1))])
        for guide in self.guides:
            line = guide.points_local
            points.append(line[np.argmax(np.linalg.norm(line, axis=1))])
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

        Returns ``(label, x, y, depth, screen_radius, entity_id)`` for every
        labelled body that is in front of the camera and inside the
        viewport. The label is what a front end draws; the entity id is what
        ``priority`` is matched against, so a renamed body keeps its
        priority. Placement is computed here rather than in a shader so any
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
            # Priority is matched on the stable identifier; the label is
            # only ever the text drawn. A renamed body keeps its priority.
            placements.append(
                (
                    body.label,
                    float(x),
                    float(y),
                    float(clip[3]),
                    float(screen_radius),
                    body.identifier,
                )
            )

        # Priority first, then nearest, so a collision resolver keeps the
        # selection and drops the far label.
        wanted = set(priority)
        placements.sort(
            key=lambda item: (item[5] not in wanted and item[0] not in wanted, item[3])
        )
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
    "zone": ("zone.vert", "zone.frag"),
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
