"""Modern OpenGL rendering.

Roadmap section 6: nothing in this package imports from
:mod:`astro_explorer.data`.  :mod:`.scene_builder` is the one module that
reads scientific records, and it emits only render primitives.

Nothing here creates a GL context on import, so the whole package is
testable headlessly.
"""

from .camera import Camera, look_at, orthographic, perspective
from .materials import MATERIALS, MaterialDefinition, atmosphere_uniforms, material_for
from .mesh import Mesh, icosphere, lod_for_distance, orbit_line, uv_sphere
from .picking import PickResult, pick, ray_sphere_intersection
from .renderer import (
    PROGRAMS,
    SHADER_DIR,
    RenderOrbit,
    RenderPlanet,
    RenderStar,
    SceneDescription,
    ShaderLibrary,
)
from .scene_builder import build_system_scene, display_radius_au, orbit_path

__all__ = [
    "MATERIALS",
    "PROGRAMS",
    "SHADER_DIR",
    "Camera",
    "MaterialDefinition",
    "Mesh",
    "PickResult",
    "RenderOrbit",
    "RenderPlanet",
    "RenderStar",
    "SceneDescription",
    "ShaderLibrary",
    "atmosphere_uniforms",
    "build_system_scene",
    "display_radius_au",
    "icosphere",
    "lod_for_distance",
    "look_at",
    "material_for",
    "orbit_line",
    "orbit_path",
    "orthographic",
    "perspective",
    "pick",
    "ray_sphere_intersection",
    "uv_sphere",
]
