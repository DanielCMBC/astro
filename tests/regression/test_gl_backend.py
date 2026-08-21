"""The OpenGL backend, exercised against a real GL 3.3 core context.

Skipped when no context can be created (headless CI without a GL driver).
Where it does run, it asserts on pixels rather than on "it did not crash":
a renderer that clears the screen and draws nothing would otherwise pass.
"""

from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from astro_explorer.rendering.camera import Camera  # noqa: E402
from astro_explorer.rendering.renderer import PROGRAMS  # noqa: E402
from astro_explorer.rendering.renderer import (  # noqa: E402
    RenderOrbit,
    RenderPlanet,
    RenderStar,
    SceneDescription,
)


@pytest.fixture(scope="module")
def context():
    """A GL 3.3 core context, or a skip.

    Uses the same factory as ``scripts/verify_gl.py`` so CI cannot verify a
    context on one backend while these tests skip on another. In CI the
    verification step runs first and fails hard, so a skip here can only
    mean a developer machine without a usable driver.
    """
    from astro_explorer.rendering.gl_backend import create_standalone_context

    try:
        ctx, backend = create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip("no OpenGL 3.3 core context available: {0}".format(exc))
    print("GL context backend: {0}".format(backend))
    yield ctx
    ctx.release()


@pytest.fixture(scope="module")
def renderer(context):
    from astro_explorer.rendering.gl_backend import GLRenderer, RenderSettings

    instance = GLRenderer(
        RenderSettings(width=320, height=240, samples=4), context=context, lod=3
    )
    yield instance
    instance.release()


def _lit_pixels(image, threshold: int = 24) -> int:
    return int((image.sum(axis=2) > threshold).sum())


# -- context and programs ---------------------------------------------------


def test_the_context_is_core_profile_3_3(context):
    version = context.info["GL_VERSION"]
    assert "3.3" in version or int(version.split(".")[0]) >= 3


def test_every_declared_program_compiles(context):
    from astro_explorer.rendering.renderer import ShaderLibrary

    library = ShaderLibrary()
    for name in PROGRAMS:
        program = context.program(**library.program_sources(name))
        assert program is not None
        program.release()


def test_the_pipeline_uses_indexed_buffers_not_immediate_mode(renderer):
    """A VBO and an EBO exist and are non-empty."""
    assert renderer._vbo.size > 0
    assert renderer._ebo.size > 0
    assert renderer._index_count == renderer._sphere.triangle_count * 3


# -- drawing ----------------------------------------------------------------


def test_an_empty_scene_renders_the_background_only(renderer):
    camera = Camera(target=np.zeros(3), distance=5.0, aspect=320 / 240)
    image = renderer.render(SceneDescription(), camera)
    assert image.shape == (240, 320, 3)
    assert _lit_pixels(image) == 0


def test_a_star_actually_appears(renderer):
    scene = SceneDescription(stars=[RenderStar("S", [0, 0, 0], 1.0, (1.0, 0.9, 0.8))])
    camera = Camera(target=np.zeros(3), distance=4.0, aspect=320 / 240, pitch=0.0)
    image = renderer.render(scene, camera)
    assert _lit_pixels(image) > 500


def test_a_planet_actually_appears(renderer):
    scene = SceneDescription(
        stars=[RenderStar("S", [-4, 0, 0], 0.2, (1, 1, 1))],
        planets=[RenderPlanet("P", [0, 0, 0], 1.0, base_color=(0.8, 0.6, 0.4))],
    )
    camera = Camera(target=np.zeros(3), distance=4.0, aspect=320 / 240, pitch=0.0)
    image = renderer.render(scene, camera)
    assert _lit_pixels(image) > 500


def test_the_lit_hemisphere_follows_the_star(renderer):
    """A day/night terminator must exist and track the light source."""
    camera = Camera(target=np.zeros(3), distance=3.0, yaw=0.0, pitch=0.0, aspect=320 / 240)

    def lit_side(star_x: float) -> str:
        scene = SceneDescription(
            stars=[RenderStar("S", [star_x, 0, 0], 0.05, (1, 1, 1))],
            planets=[RenderPlanet("P", [0, 0, 0], 1.0, base_color=(0.9, 0.9, 0.9))],
        )
        luminance = renderer.render(scene, camera).sum(axis=2).astype(float)
        height, width = luminance.shape
        strip = luminance[height // 2 - 10 : height // 2 + 10, :].mean(axis=0)
        disc = np.where(strip > 8)[0]
        low, high = int(disc.min()), int(disc.max())
        span = high - low
        left = strip[low + int(0.15 * span) : low + int(0.35 * span)].mean()
        right = strip[low + int(0.65 * span) : low + int(0.85 * span)].mean()
        return "LEFT" if left > right else "RIGHT"

    assert lit_side(-4.0) == "LEFT"
    assert lit_side(4.0) == "RIGHT"


def test_orbits_are_drawn(renderer):
    angle = np.linspace(0.0, 2.0 * np.pi, 256)
    points = np.stack([np.cos(angle), np.zeros_like(angle), np.sin(angle)], axis=-1)
    scene = SceneDescription(orbits=[RenderOrbit("o", points, color=(1, 1, 1, 1.0))])
    camera = Camera(target=np.zeros(3), distance=3.0, aspect=320 / 240, pitch=1.4)
    assert _lit_pixels(renderer.render(scene, camera)) > 100


def test_a_dashed_orbit_draws_fewer_pixels_than_a_solid_one(renderer):
    """The dash is how an assumed orbit is signalled; it must be visible."""
    angle = np.linspace(0.0, 2.0 * np.pi, 512)
    points = np.stack([np.cos(angle), np.zeros_like(angle), np.sin(angle)], axis=-1)
    camera = Camera(target=np.zeros(3), distance=3.0, aspect=320 / 240, pitch=1.4)

    solid = SceneDescription(orbits=[RenderOrbit("o", points, color=(1, 1, 1, 1.0))])
    dashed = SceneDescription(
        orbits=[RenderOrbit("o", points, color=(1, 1, 1, 1.0), dashed=True)]
    )
    solid_pixels = _lit_pixels(renderer.render(solid, camera))
    dashed_pixels = _lit_pixels(renderer.render(dashed, camera))
    assert 0 < dashed_pixels < solid_pixels


def test_many_planets_are_one_instanced_draw(renderer):
    """Instancing is what keeps a large system from costing N draw calls."""
    rng = np.random.default_rng(0)
    planets = [
        RenderPlanet("p{0}".format(i), rng.uniform(-2, 2, 3), 0.08)
        for i in range(200)
    ]
    scene = SceneDescription(
        stars=[RenderStar("S", [0, 0, 0], 0.1, (1, 1, 1))], planets=planets
    )
    # One buffer holds all 200 instances.
    assert scene.instance_buffer().shape == (200, 8)

    camera = Camera(target=np.zeros(3), distance=8.0, aspect=320 / 240)
    assert _lit_pixels(renderer.render(scene, camera)) > 200


def test_depth_testing_hides_the_far_body(renderer):
    """A body behind another must not paint over it."""
    camera = Camera(target=np.zeros(3), distance=6.0, yaw=0.0, pitch=0.0, aspect=320 / 240)
    front_only = SceneDescription(
        stars=[RenderStar("near", [0, 0, 2.0], 0.8, (1, 1, 1))]
    )
    both = SceneDescription(
        stars=[
            RenderStar("near", [0, 0, 2.0], 0.8, (1, 1, 1)),
            RenderStar("far", [0, 0, -2.0], 0.8, (1, 1, 1)),
        ]
    )
    # The far star is completely occluded, so the pixel count barely moves.
    a = _lit_pixels(renderer.render(front_only, camera))
    b = _lit_pixels(renderer.render(both, camera))
    assert a > 0
    assert b <= a * 1.2


# -- the vertical slice through real GL -------------------------------------


def test_the_hd80606b_slice_renders(renderer):
    from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
    from astro_explorer.rendering.scene_builder import build_frame_scene

    try:
        catalog = load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")

    slice_ = build_slice("HD 80606", catalog)
    scene = build_frame_scene(
        slice_.frame, slice_.star, slice_.planets,
        mean_anomalies=slice_.mean_anomalies(2458882.344),
    )

    points = scene.orbits[0].points_local.astype(float)
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    extent = float(np.linalg.norm(points - centre, axis=1).max())

    camera = Camera(target=centre, aspect=320 / 240, pitch=1.2)
    camera.frame_object(extent, margin=1.3)
    image = renderer.render(scene, camera)

    assert image.shape == (240, 320, 3)
    assert _lit_pixels(image) > 50


# -- batching and LOD through a real context (review section 15) ------------


def test_a_six_planet_system_costs_one_orbit_draw_call(renderer):
    """Review section 15: batched orbit geometry."""
    from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
    from astro_explorer.rendering.scene_builder import build_frame_scene

    try:
        catalog = load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")

    system = build_slice("Kepler-11", catalog)
    scene = build_frame_scene(
        system.frame, system.star, system.planets,
        mean_anomalies=system.mean_anomalies(2455590.0),
    )
    assert len(scene.orbits) == 6

    camera = Camera(target=np.zeros(3), distance=1.5, aspect=320 / 240, pitch=1.1)
    image = renderer.render(scene, camera)

    # Six orbits, one draw call - the point of batching them.
    assert renderer.last_orbit_draw_calls == 1

    # Planets cost one instanced draw per (material, LOD) group, never one
    # per planet. Kepler-11's six span two material classes at one LOD.
    materials = {planet.material_id for planet in scene.planets}
    levels = {int(planet.lod) for planet in scene.planets}
    assert renderer.last_planet_draw_calls == len(materials) * len(levels)
    assert renderer.last_planet_draw_calls < len(scene.planets)
    assert _lit_pixels(image) > 100


def test_mixed_lod_costs_one_draw_call_per_level(renderer):
    from astro_explorer.rendering.renderer import RenderPlanet, SceneDescription

    planets = [
        RenderPlanet("near", [0.0, 0.0, 0.0], 0.5, lod=4),
        RenderPlanet("mid", [2.0, 0.0, 0.0], 0.2, lod=2),
        RenderPlanet("far", [4.0, 0.0, 0.0], 0.05, lod=1),
        RenderPlanet("far2", [5.0, 0.0, 0.0], 0.05, lod=1),
    ]
    scene = SceneDescription(
        stars=[RenderStar("S", [-6, 0, 0], 0.3, (1, 1, 1))], planets=planets
    )
    camera = Camera(target=np.array([2.5, 0, 0]), distance=9.0, aspect=320 / 240)
    renderer.render(scene, camera)

    # Three distinct LOD levels, one material -> three draws, not four.
    assert renderer.last_planet_draw_calls == 3


def test_every_lod_level_renders_something(renderer):
    from astro_explorer.rendering.renderer import RenderPlanet, SceneDescription

    camera = Camera(target=np.zeros(3), distance=3.0, aspect=320 / 240, pitch=0.0)
    for lod in range(5):
        scene = SceneDescription(
            stars=[RenderStar("S", [-8, 0, 0], 0.05, (1, 1, 1))],
            planets=[RenderPlanet("P", [0, 0, 0], 1.0, base_color=(0.9, 0.9, 0.9), lod=lod)],
        )
        assert _lit_pixels(renderer.render(scene, camera)) > 100, lod


def test_a_coarse_lod_uses_fewer_triangles(renderer):
    assert renderer._mesh_for(1).triangle_count < renderer._mesh_for(4).triangle_count
    assert renderer._mesh_for(0).triangle_count == 20


def test_labels_can_be_composited_onto_a_frame(renderer):
    from astro_explorer.rendering.labels import draw_labels
    from astro_explorer.rendering.renderer import RenderPlanet, SceneDescription

    scene = SceneDescription(
        stars=[RenderStar("S", [0, 0, 0], 0.3, (1, 1, 1), label="Host")],
        planets=[RenderPlanet("P", [1.5, 0, 0], 0.1, label="Host b")],
    )
    camera = Camera(target=np.zeros(3), distance=4.0, aspect=320 / 240, pitch=0.6)
    image = renderer.render(scene, camera)
    placements = scene.project_labels(camera, 320, 240)
    assert placements

    labelled = draw_labels(image, placements, header=["a header line"])
    assert labelled.shape == image.shape
    assert labelled.dtype == np.uint8
    # Text adds lit pixels without touching the original array.
    assert _lit_pixels(labelled) > _lit_pixels(image)
    assert not np.array_equal(labelled, image)
