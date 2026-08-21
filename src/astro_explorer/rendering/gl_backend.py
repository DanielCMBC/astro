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

__all__ = [
    "GLRenderer",
    "RenderSettings",
    "moderngl_available",
    "create_standalone_context",
    "GL_BACKENDS",
]


def moderngl_available() -> bool:
    """True when ModernGL is importable. Says nothing about a context."""
    try:
        import moderngl  # noqa: F401
    except ImportError:
        return False
    return True


#: Backends tried in order when the default context creation fails. A
#: headless Linux box (CI) normally has no GLX display but does have Mesa's
#: software EGL.
GL_BACKENDS = ("egl", "osmesa")


def create_standalone_context(require: int = 330):
    """A standalone GL context, trying each backend a headless box may offer.

    Returns ``(context, backend_name)``. Raises RuntimeError listing every
    attempt when none succeed.

    Both :class:`GLRenderer` and ``scripts/verify_gl.py`` go through this, so
    CI cannot end up in the state where the verification step succeeds on one
    backend while the test fixture skips on another.
    """
    import moderngl

    failures = []
    try:
        return moderngl.create_standalone_context(require=require), "default"
    except Exception as exc:  # pragma: no cover - depends on the host
        failures.append("default: {0}".format(exc))

    for backend in GL_BACKENDS:
        try:
            return (
                moderngl.create_context(standalone=True, require=require, backend=backend),
                backend,
            )
        except Exception as exc:  # pragma: no cover - depends on the host
            failures.append("{0}: {1}".format(backend, exc))

    raise RuntimeError(
        "no OpenGL {0} context could be created:\n  ".format(require)
        + "\n  ".join(failures)
    )


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
        if context is None:
            self.ctx, self.backend = create_standalone_context(require=330)
        else:
            self.ctx, self.backend = context, "supplied"
        self.library = ShaderLibrary()
        self._lod = lod
        self._default_lod = int(np.clip(lod, 0, 5))

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self._programs: dict[str, "moderngl.Program"] = {}
        self._meshes: dict[int, Mesh] = {}
        self._sphere_buffers: dict[int, tuple] = {}
        self._sphere = self._mesh_for(self._default_lod)
        self._build_framebuffer()

        #: Draw calls issued by the last frame, for the batching tests.
        self.last_planet_draw_calls = 0
        self.last_orbit_draw_calls = 0

    # -- setup -----------------------------------------------------------
    def _program(self, name: str):
        """Compile a declared program once and cache it."""
        if name not in self._programs:
            self._programs[name] = self.ctx.program(**self.library.program_sources(name))
        return self._programs[name]

    def _mesh_for(self, lod: int) -> Mesh:
        """Unit sphere at a subdivision level, built and uploaded once.

        Per-system LOD means several levels can be live simultaneously - a
        close planet at level 4, a distant one at level 1 - so meshes are
        cached by level rather than replaced.
        """
        level = int(np.clip(lod, 0, 5))
        if level not in self._meshes:
            mesh = icosphere(level)
            self._meshes[level] = mesh
            self._sphere_buffers[level] = (
                self.ctx.buffer(mesh.vertex_bytes),
                self.ctx.buffer(mesh.index_bytes),
            )
        return self._meshes[level]

    @property
    def _vbo(self):
        return self._sphere_buffers[self._default_lod][0]

    @property
    def _ebo(self):
        return self._sphere_buffers[self._default_lod][1]

    @property
    def _index_count(self) -> int:
        return self._meshes[self._default_lod].triangle_count * 3

    def _build_framebuffer(self) -> None:
        settings = self.settings
        size = (settings.width, settings.height)

        # A software rasteriser (llvmpipe, which is what CI has) supports
        # fewer samples than a discrete GPU, and asking for more fails with
        # "the number of samples is invalid". Clamp to what this context
        # actually reports; anti-aliasing is a quality knob, not a
        # correctness one, so degrading is right and silence is not.
        supported = int(getattr(self.ctx, "max_samples", 0) or 0)
        if settings.samples > 1 and supported and settings.samples > supported:
            self.samples = supported
        else:
            self.samples = settings.samples
        if self.samples != settings.samples:
            print(
                "GLRenderer: {0}x MSAA unsupported by this context, using {1}x".format(
                    settings.samples, self.samples
                )
            )

        if self.samples > 1:
            self._msaa = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer(size, samples=self.samples)],
                depth_attachment=self.ctx.depth_renderbuffer(size, samples=self.samples),
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

    def _sphere_vao(self, program, instance_data: np.ndarray, instance_fields, lod=None):
        """VAO binding a shared sphere plus a per-instance buffer."""
        level = self._default_lod if lod is None else int(np.clip(lod, 0, 5))
        self._mesh_for(level)
        vbo, ebo = self._sphere_buffers[level]

        buffer = self.ctx.buffer(np.ascontiguousarray(instance_data, dtype="f4").tobytes())

        vertex_layout, vertex_names = self._binding(program, self._SPHERE_FIELDS)
        instance_layout, instance_names = self._binding(
            program, instance_fields, per_instance=True
        )

        content = [(vbo, vertex_layout, *vertex_names)]
        if instance_names:
            content.append((buffer, instance_layout, *instance_names))

        vao = self.ctx.vertex_array(program, content, ebo)
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

        self.last_planet_draw_calls = 0
        self.last_orbit_draw_calls = 0
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

        # Group by (material, LOD) so each shader family at each detail
        # level is a single instanced draw. A six-planet system where the
        # inner three are close and the outer three distant costs at most
        # two draws, not six.
        by_group: dict[tuple, list] = {}
        for planet in scene.planets:
            by_group.setdefault((planet.material_id, int(planet.lod)), []).append(planet)

        self.last_planet_draw_calls = len(by_group)

        for (material_id, lod), planets in by_group.items():
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
                lod=lod,
            )
            vao.render(instances=len(planets))
            vao.release()
            buffer.release()

    def _draw_orbits(self, scene, view_projection) -> None:
        """Every orbit in one indexed draw call.

        Orbit paths are concatenated into a single vertex buffer and drawn
        as ``LINES`` through an index buffer, so a six-planet system costs
        one draw call instead of six. ``LINE_STRIP`` cannot do this - the
        strips would join end to end - and GL 3.3 has no portable primitive
        restart in ModernGL, so the segments are indexed explicitly.

        Per-orbit style therefore travels as vertex attributes rather than
        uniforms: colour, and a dash period of zero meaning "solid".
        """
        import moderngl

        if not scene.orbits:
            return

        vertices, indices = self._batch_orbits(scene)
        if indices.size == 0:
            return

        program = self._program("orbit")
        program["u_view_projection"].write(view_projection)
        self.ctx.line_width = self.settings.orbit_line_width
        # Orbit paths are translucent overlays and must not occlude one
        # another, so depth writes are off while they are drawn.
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_mask = False

        buffer = self.ctx.buffer(vertices.tobytes())
        index_buffer = self.ctx.buffer(indices.tobytes())
        layout, names = self._binding(program, self._ORBIT_FIELDS)
        vao = self.ctx.vertex_array(
            program, [(buffer, layout, *names)], index_buffer
        )
        vao.render(moderngl.LINES)
        self.last_orbit_draw_calls = 1

        vao.release()
        buffer.release()
        index_buffer.release()

        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    #: Interleaved layout of the batched orbit buffer.
    _ORBIT_FIELDS = [
        ("in_position", "3f", 12),
        ("in_arclength", "1f", 4),
        ("in_color", "4f", 16),
        ("in_dash_period", "1f", 4),
    ]

    def _batch_orbits(self, scene):
        """Pack every orbit into one vertex array and one index array."""
        vertex_blocks = []
        index_blocks = []
        offset = 0

        for orbit in scene.orbits:
            points = orbit.points_local
            count = points.shape[0]
            if count < 2:
                continue

            # Cumulative arc length drives the dash pattern, so dashes stay
            # even along a highly eccentric path where the sample spacing
            # varies by orders of magnitude.
            segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
            arclength = np.concatenate([[0.0], np.cumsum(segments)])
            total = float(arclength[-1]) or 1.0
            dash = total * self.settings.dash_period if orbit.dashed else 0.0

            block = np.empty((count, 9), dtype=np.float32)
            block[:, 0:3] = points
            block[:, 3] = arclength
            block[:, 4:8] = orbit.color
            block[:, 8] = dash
            vertex_blocks.append(block)

            # One line segment per consecutive pair, indexed so the strips
            # never join across orbits.
            starts = np.arange(count - 1, dtype=np.uint32) + offset
            index_blocks.append(np.stack([starts, starts + 1], axis=-1).ravel())
            offset += count

        if not vertex_blocks:
            return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.uint32)

        return (
            np.ascontiguousarray(np.vstack(vertex_blocks), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(index_blocks), dtype=np.uint32),
        )

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
        for vbo, ebo in self._sphere_buffers.values():
            vbo.release()
            ebo.release()
        self._sphere_buffers.clear()
        self._meshes.clear()
        if self._msaa is not None:
            self._msaa.release()
        self._resolve.release()

    def __enter__(self) -> "GLRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
