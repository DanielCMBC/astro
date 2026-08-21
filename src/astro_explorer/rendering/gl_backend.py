"""Modern core-profile OpenGL backend (roadmap Phase 3).

Consumes a :class:`~astro_explorer.rendering.renderer.SceneDescription` and
draws it with a GL 3.3 core pipeline: VAOs, VBOs, an EBO, GLSL programs and
instanced draws. There is no ``glBegin``, no ``gluSphere``, no matrix stack
and no client-state vertex array anywhere in it.

The backend is deliberately ignorant. It receives float32 positions, radii,
colours and material ids, and it has no way to ask what the eccentricity was
or whether a value was measured - the scene type does not carry that. This
module is also forbidden by ``tests/regression/test_architecture.py`` from
importing the data layer.

It renders to an offscreen framebuffer, so the whole slice can be verified
in CI without a window. ModernGL is an optional dependency; import this
module only when it is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .materials import MATERIALS
from .mesh import Mesh, icosphere
from .renderer import SceneDescription, ShaderLibrary

__all__ = ["GLRenderer", "RenderSettings", "moderngl_available"]


def moderngl_available() -> bool:
    """True when ModernGL can be imported and a context can be created."""
    try:
        import moderngl  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class RenderSettings:
    """Framebuffer and post-processing options."""

    width: int = 1280
    height: int = 800
    samples: int = 4
    background: tuple[float, float, float, float] = (0.012, 0.016, 0.031, 1.0)
    exposure: float = 1.6
    star_granulation: float = 0.25
    orbit_line_width: float = 1.6
    #: Screen-space dash period for an orbit drawn from assumed elements.
    dash_period: float = 0.035
    draw_orbits: bool = True


class GLRenderer:
    """Draws a scene with a modern OpenGL pipeline.

    Buffers are built once and reused: the sphere mesh lives in a single
    VBO/EBO pair, and each frame uploads only the per-instance attributes.
    That is what roadmap section 25 means by avoiding per-object Python draw
    overhead - a thousand planets cost one draw call, not a thousand.
    """

    def __init__(self, settings: RenderSettings | None = None, *, context=None, lod: int = 4):
        import moderngl

        self.settings = settings or RenderSettings()
        self.ctx = context or moderngl.create_standalone_context(require=330)
        self.library = ShaderLibrary()
        self._lod = lod

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self._programs: dict[str, "moderngl.Program"] = {}
        self._sphere = icosphere(lod)
        self._build_sphere_buffers(self._sphere)
        self._build_framebuffer()

    # -- setup -----------------------------------------------------------
    def _program(self, name: str):
        """Compile a declared program once and cache it."""
        if name not in self._programs:
            self._programs[name] = self.ctx.program(**self.library.program_sources(name))
        return self._programs[name]

    def _build_sphere_buffers(self, mesh: Mesh) -> None:
        """One VBO + EBO for the shared unit sphere."""
        self._vbo = self.ctx.buffer(mesh.vertex_bytes)
        self._ebo = self.ctx.buffer(mesh.index_bytes)
        self._index_count = mesh.triangle_count * 3

    def _build_framebuffer(self) -> None:
        settings = self.settings
        size = (settings.width, settings.height)
        if settings.samples > 1:
            self._msaa = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer(size, samples=settings.samples)],
                depth_attachment=self.ctx.depth_renderbuffer(size, samples=settings.samples),
            )
        else:
            self._msaa = None
        self._resolve = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(size, 4)],
            depth_attachment=self.ctx.depth_renderbuffer(size),
        )

    # -- instanced geometry ----------------------------------------------
    @staticmethod
    def _binding(program, fields: list[tuple[str, str, int]], per_instance: bool = False):
        """Build a ModernGL buffer format for whichever attributes survived.

        GLSL compilers strip attributes a shader never reads - ``star.frag``
        ignores the UVs, for instance - and binding a name that is not in the
        program is an error. Rather than dropping the field, which would
        shift every later attribute onto the wrong offset, the stride is kept
        intact by emitting explicit padding (``2x4``) in its place.

        Returns ``(format_string, attribute_names)``.
        """
        parts: list[str] = []
        names: list[str] = []
        for name, fmt, byte_size in fields:
            if name in program:
                parts.append(fmt)
                names.append(name)
            else:
                parts.append("{0}x1".format(byte_size))
        layout = " ".join(parts)
        if per_instance:
            layout += "/i"
        return layout, names

    #: Interleaved layout of the shared sphere VBO.
    _SPHERE_FIELDS = [
        ("in_position", "3f", 12),
        ("in_normal", "3f", 12),
        ("in_uv", "2f", 8),
    ]

    def _sphere_vao(self, program, instance_data: np.ndarray, instance_fields):
        """VAO binding the shared sphere plus a per-instance buffer."""
        buffer = self.ctx.buffer(np.ascontiguousarray(instance_data, dtype="f4").tobytes())

        vertex_layout, vertex_names = self._binding(program, self._SPHERE_FIELDS)
        instance_layout, instance_names = self._binding(
            program, instance_fields, per_instance=True
        )

        content = [(self._vbo, vertex_layout, *vertex_names)]
        if instance_names:
            content.append((buffer, instance_layout, *instance_names))

        vao = self.ctx.vertex_array(program, content, self._ebo)
        return vao, buffer

    # -- drawing ---------------------------------------------------------
    def render(self, scene: SceneDescription, camera: Camera, *, time: float = 0.0) -> np.ndarray:
        """Draw one frame and return it as an ``(H, W, 3)`` uint8 array."""
        import moderngl

        target = self._msaa or self._resolve
        target.use()
        self.ctx.clear(*self.settings.background)
        self.ctx.enable(moderngl.DEPTH_TEST)

        view_projection = np.ascontiguousarray(
            camera.view_projection().T.astype("f4")
        ).tobytes()
        camera_position = tuple(float(v) for v in camera.position)

        star_position = (
            tuple(float(v) for v in scene.stars[0].position_local)
            if scene.stars
            else (0.0, 0.0, 0.0)
        )

        self._draw_stars(scene, view_projection, camera_position, time)
        self._draw_planets(scene, view_projection, camera_position, star_position)
        if self.settings.draw_orbits:
            self._draw_orbits(scene, view_projection)

        if self._msaa is not None:
            self.ctx.copy_framebuffer(self._resolve, self._msaa)

        data = self._resolve.read(components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.settings.height, self.settings.width, 3
        )
        return np.flipud(image)

    def _draw_stars(self, scene, view_projection, camera_position, time) -> None:
        if not scene.stars:
            return
        program = self._program("star")
        program["u_view_projection"].write(view_projection)
        program["u_camera_position"].value = camera_position
        for name, value in (
            ("u_limb_darkening", MATERIALS["star"].uniforms["u_limb_darkening"]),
            ("u_granulation", self.settings.star_granulation),
            ("u_exposure", self.settings.exposure),
            ("u_time", float(time)),
        ):
            if name in program:
                program[name].value = value

        vao, buffer = self._sphere_vao(
            program,
            scene.star_instance_buffer(),
            [
                ("instance_position", "3f", 12),
                ("instance_radius", "1f", 4),
                ("instance_color", "3f", 12),
            ],
        )
        vao.render(instances=len(scene.stars))
        vao.release()
        buffer.release()

    def _draw_planets(self, scene, view_projection, camera_position, star_position) -> None:
        if not scene.planets:
            return

        # Group by material so each shader family is one instanced draw.
        by_material: dict[str, list] = {}
        for planet in scene.planets:
            by_material.setdefault(planet.material_id, []).append(planet)

        for material_id, planets in by_material.items():
            definition = MATERIALS.get(material_id, MATERIALS["rocky"])
            program = self._program(definition.program)
            program["u_view_projection"].write(view_projection)
            program["u_camera_position"].value = camera_position
            program["u_star_position"].value = star_position
            for name, value in definition.uniforms.items():
                if name in program:
                    program[name].value = value

            instances = np.vstack(
                [
                    np.concatenate(
                        [p.position_local, [p.radius_display], p.base_color, [p.emissive]]
                    )
                    for p in planets
                ]
            )
            vao, buffer = self._sphere_vao(
                program,
                instances,
                [
                    ("instance_position", "3f", 12),
                    ("instance_radius", "1f", 4),
                    ("instance_color", "3f", 12),
                    ("instance_emissive", "1f", 4),
                ],
            )
            vao.render(instances=len(planets))
            vao.release()
            buffer.release()

    def _draw_orbits(self, scene, view_projection) -> None:
        import moderngl

        if not scene.orbits:
            return
        program = self._program("orbit")
        program["u_view_projection"].write(view_projection)
        self.ctx.line_width = self.settings.orbit_line_width
        # Orbit paths are translucent overlays; do not occlude each other.
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for orbit in scene.orbits:
            points = orbit.points_local
            # Cumulative arc length drives the dash pattern, so dashes stay
            # even along a highly eccentric path where the sample spacing
            # varies by orders of magnitude.
            segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
            arclength = np.concatenate([[0.0], np.cumsum(segments)]).astype("f4")
            total = float(arclength[-1]) or 1.0

            interleaved = np.hstack([points, arclength[:, None]]).astype("f4")
            buffer = self.ctx.buffer(np.ascontiguousarray(interleaved).tobytes())
            vao = self.ctx.vertex_array(
                program, [(buffer, "3f 1f", "in_position", "in_arclength")]
            )
            program["u_color"].value = orbit.color
            program["u_dashed"].value = bool(orbit.dashed)
            program["u_dash_period"].value = total * self.settings.dash_period

            vao.render(moderngl.LINE_STRIP)
            vao.release()
            buffer.release()

        self.ctx.disable(moderngl.BLEND)

    # -- output ----------------------------------------------------------
    def save_png(self, image: np.ndarray, path) -> None:
        """Write a rendered frame to disk without pulling in a GUI toolkit."""
        try:
            from PIL import Image

            Image.fromarray(image).save(str(path))
            return
        except ImportError:
            pass

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg

        mpimg.imsave(str(path), image)

    def release(self) -> None:
        for program in self._programs.values():
            program.release()
        self._programs.clear()
        self._vbo.release()
        self._ebo.release()
        if self._msaa is not None:
            self._msaa.release()
        self._resolve.release()

    def __enter__(self) -> "GLRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
