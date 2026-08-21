"""Architecture and rendering-contract tests.

Roadmap section 6 states the golden rule as prose.  Here it is enforced
mechanically, so a future change that lets the renderer reach into the data
layer fails the build rather than passing review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "astro_explorer"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()

    package_parts = path.relative_to(SRC).parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Resolve a relative import to an absolute package path.
                base = ["astro_explorer", *package_parts]
                trimmed = base[: len(base) - node.level + 1]
                target = ".".join(trimmed + ([node.module] if node.module else []))
                modules.add(target)
            elif node.module:
                modules.add(node.module)
    return modules


def _python_files(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _code_only(path: Path) -> str:
    """Source with comments and string literals removed.

    Needed because these modules *document* the constructs they forbid -
    ``gl_backend`` explains that it uses no ``glBegin``, and
    ``system_frame`` explains the ``0.005`` bug. Scanning raw text would
    flag the explanation as the offence.
    """
    import io
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


# -- the golden architecture rule (roadmap section 6) ------------------------


def test_physics_does_not_import_data_ui_or_rendering():
    """The scientific core must stand alone."""
    for path in _python_files("physics"):
        for module in _imported_modules(path):
            assert not module.startswith("astro_explorer.data"), path
            assert not module.startswith("astro_explorer.ui"), path
            assert not module.startswith("astro_explorer.rendering"), path


def test_coordinates_does_not_import_data_ui_or_rendering():
    for path in _python_files("coordinates"):
        for module in _imported_modules(path):
            assert not module.startswith("astro_explorer.data"), path
            assert not module.startswith("astro_explorer.ui"), path
            assert not module.startswith("astro_explorer.rendering"), path


def test_spectroscopy_does_not_import_ui_or_rendering():
    for path in _python_files("spectroscopy"):
        for module in _imported_modules(path):
            assert not module.startswith("astro_explorer.ui"), path
            assert not module.startswith("astro_explorer.rendering"), path


def test_the_renderer_never_imports_the_data_layer():
    """Roadmap section 6: the renderer must not own scientific truth."""
    for path in _python_files("rendering"):
        for module in _imported_modules(path):
            assert not module.startswith("astro_explorer.data"), (
                "{0} imports the data layer".format(path.name)
            )


def test_only_the_scene_builder_reads_science_inside_rendering():
    """Every other rendering module works on plain numbers."""
    for path in _python_files("rendering"):
        if path.name == "scene_builder.py":
            continue
        for module in _imported_modules(path):
            assert not module.startswith("astro_explorer.physics"), path
            assert not module.startswith("astro_explorer.provenance"), path


def test_no_module_hard_codes_a_physical_constant():
    """Roadmap 3.8: constants come from Astropy, not from literals."""
    forbidden = ("6.62607015e-34", "1.380649e-23", "299792458", "5.670374419e-08")
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "constants.py":
            continue
        text = path.read_text(encoding="utf-8")
        for literal in forbidden:
            if literal in text:
                offenders.append("{0}: {1}".format(path.name, literal))
    assert not offenders, offenders


def test_no_module_uses_the_wrong_au_to_parsec_factor():
    """Roadmap 4.2: the 0.005 scaling must never come back."""
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "0.005" in line and "pc" in line.lower() and not line.strip().startswith("#"):
                offenders.append("{0}: {1}".format(path.name, line.strip()))
    assert not offenders, offenders


# -- the renderer's input contract -------------------------------------------


def test_render_primitives_carry_no_scientific_fields():
    """The renderer cannot decide what it is never told."""
    from astro_explorer.rendering.renderer import RenderOrbit, RenderPlanet, RenderStar

    forbidden = {
        "eccentricity",
        "semimajor_axis",
        "mean_anomaly",
        "true_anomaly",
        "distance_pc",
        "status",
        "provenance",
        "molecule",
        "detection_status",
    }
    for cls in (RenderStar, RenderPlanet, RenderOrbit):
        assert not (set(cls.__dataclass_fields__) & forbidden), cls.__name__


def test_render_primitives_reject_non_finite_positions():
    from astro_explorer.rendering.renderer import RenderPlanet

    with pytest.raises(ValueError, match="finite"):
        RenderPlanet("x", [np.nan, 0.0, 0.0], 1.0)


def test_scene_instance_buffers_have_the_declared_layout():
    from astro_explorer.rendering.renderer import RenderPlanet, RenderStar, SceneDescription

    scene = SceneDescription(
        stars=[RenderStar("S", [0, 0, 0], 1.0, (1.0, 0.9, 0.8))],
        planets=[RenderPlanet("P", [2, 0, 0], 0.1), RenderPlanet("Q", [3, 0, 0], 0.2)],
    )
    assert scene.star_instance_buffer().shape == (1, 7)
    assert scene.instance_buffer().shape == (2, 8)
    assert scene.instance_buffer().dtype == np.float32


# -- shaders -----------------------------------------------------------------


def test_every_declared_shader_program_has_both_stages():
    from astro_explorer.rendering.renderer import PROGRAMS, ShaderLibrary

    library = ShaderLibrary()
    assert sorted(library.available()) == sorted(PROGRAMS)
    for name in PROGRAMS:
        sources = library.program_sources(name)
        assert sources["vertex_shader"].startswith("#version 330 core")
        assert sources["fragment_shader"].startswith("#version 330 core")


def test_shaders_use_no_fixed_function_constructs():
    """Roadmap 4.8: no glBegin, no gluSphere, no built-in matrix stack."""
    from astro_explorer.rendering.renderer import SHADER_DIR

    forbidden = ("gl_ModelViewMatrix", "gl_ProjectionMatrix", "varying ", "attribute ", "ftransform")
    for path in sorted(SHADER_DIR.glob("*")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, "{0} uses {1}".format(path.name, token)


def test_the_atmosphere_shader_can_be_switched_off():
    """A planet with no evidence of an atmosphere gets no haze."""
    from astro_explorer.rendering.materials import atmosphere_uniforms

    assert atmosphere_uniforms(has_evidence=False)["u_enabled"] is False
    assert atmosphere_uniforms(has_evidence=True)["u_enabled"] is True


def test_material_uniform_typos_are_rejected():
    from astro_explorer.rendering.materials import MATERIALS

    with pytest.raises(KeyError):
        MATERIALS["rocky"].with_values(u_rougness=0.5)


# -- meshes and camera -------------------------------------------------------


def test_sphere_normals_are_unit_length():
    from astro_explorer.rendering.mesh import icosphere, uv_sphere

    for mesh in (uv_sphere(12, 24), icosphere(2)):
        normals = mesh.vertices[:, 3:6]
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_sphere_vertices_lie_on_the_sphere():
    from astro_explorer.rendering.mesh import icosphere

    mesh = icosphere(3, radius=2.5)
    radii = np.linalg.norm(mesh.vertices[:, :3], axis=1)
    assert np.allclose(radii, 2.5, atol=1e-5)


def test_mesh_indices_are_in_range():
    from astro_explorer.rendering.mesh import uv_sphere

    mesh = uv_sphere(8, 16)
    assert mesh.indices.max() < mesh.vertex_count


def test_projection_matrix_maps_the_near_plane_to_minus_one():
    from astro_explorer.rendering.camera import perspective

    matrix = perspective(np.radians(45.0), 1.5, 0.1, 100.0)
    point = matrix @ np.array([0.0, 0.0, -0.1, 1.0])
    assert np.isclose(point[2] / point[3], -1.0)


def test_view_matrix_places_the_target_on_the_axis():
    from astro_explorer.rendering.camera import Camera

    camera = Camera(target=np.array([1.0, 2.0, 3.0]), distance=10.0)
    view = camera.view_matrix()
    transformed = view @ np.append(camera.target, 1.0)
    assert np.allclose(transformed[:2], 0.0, atol=1e-9)
    assert np.isclose(transformed[2], -10.0)


def test_picking_finds_the_nearer_body():
    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.picking import pick
    from astro_explorer.rendering.renderer import RenderPlanet, SceneDescription

    camera = Camera(target=np.zeros(3), distance=10.0, yaw=0.0, pitch=0.0, aspect=1.0)
    scene = SceneDescription(
        planets=[
            RenderPlanet("near", [0.0, 0.0, 5.0], 0.5),
            RenderPlanet("far", [0.0, 0.0, -5.0], 0.5),
        ]
    )
    result = pick(scene, camera, 200, 200, 400, 400)
    assert result is not None and result.identifier == "near"


def test_ray_sphere_is_stable_at_astronomical_range():
    """A camera parsecs away from a small body must still get a hit."""
    from astro_explorer.rendering.picking import ray_sphere_intersection

    origin = np.array([0.0, 0.0, 1.0e8])
    direction = np.array([0.0, 0.0, -1.0])
    hit = ray_sphere_intersection(origin, direction, np.zeros(3), 1.0)
    assert hit is not None
    assert np.isclose(hit, 1.0e8 - 1.0, rtol=1e-6)


def test_a_ray_pointing_away_misses():
    from astro_explorer.rendering.picking import ray_sphere_intersection

    assert ray_sphere_intersection([0, 0, 10], [0, 0, 1], [0, 0, 0], 1.0) is None


# -- guards added with the 3D vertical slice --------------------------------


def test_the_orientation_module_is_pure_numeric():
    """The 3D transform must be testable without units or provenance.

    Keeping it free of astropy and of the provenance layer is what lets
    ``tests/physics/test_orientation.py`` check it against the expanded
    scalar equations with nothing else in the way.
    """
    modules = _imported_modules(SRC / "physics" / "orientation.py")
    assert not any(m.startswith("astro_explorer.provenance") for m in modules)
    assert not any(m.startswith("astropy") for m in modules)
    assert "numpy" in modules


def test_the_gl_backend_never_imports_science_or_application_code():
    """The backend receives primitives; it must not be able to fetch more."""
    modules = _imported_modules(SRC / "rendering" / "gl_backend.py")
    for module in modules:
        assert not module.startswith("astro_explorer.data"), module
        assert not module.startswith("astro_explorer.app"), module
        assert not module.startswith("astro_explorer.physics"), module
        assert not module.startswith("astro_explorer.provenance"), module


def test_the_gl_backend_uses_no_fixed_function_calls():
    """Roadmap 4.8: no glBegin, gluSphere, matrix stack or client arrays."""
    source = _code_only(SRC / "rendering" / "gl_backend.py")
    for token in (
        "glBegin", "glEnd", "gluSphere", "glMatrixMode", "glTranslatef",
        "glRotatef", "glVertex", "EnableClientState",
    ):
        assert token not in source, token


def test_the_system_frame_module_defines_no_conversion_literals():
    """Every factor must come from astropy, not from a typed-in number."""
    source = _code_only(SRC / "coordinates" / "system_frame.py")
    for literal in ("206264.806", "4.8481368e-06", "1.495978707e8", "0.005"):
        assert literal not in source, "hard-coded conversion factor: " + literal


def test_only_the_scene_builder_and_slice_bridge_frames_and_primitives():
    """FramedPosition must not leak into the renderer's own modules."""
    for path in _python_files("rendering"):
        if path.name == "scene_builder.py":
            continue
        assert "FramedPosition" not in _code_only(path), path.name


# -- review section 6: display radius is not a physical quantity ------------


SCIENCE_PACKAGES = ("physics", "data", "spectroscopy", "classification", "coordinates")


def test_display_radius_never_reaches_the_science_layers():
    """Review section 6: display radius is strictly a rendering parameter.

    It must never enter gravitational, transit-depth, density, collision,
    orbital-distance or stellar-radius calculations, nor any scientific
    plot. The cheapest way to guarantee that is for the science packages
    never to see the symbol at all.
    """
    offenders = []
    for package in SCIENCE_PACKAGES:
        for path in _python_files(package):
            code = _code_only(path)
            for symbol in ("display_radius", "DisplayScale", "radius_display"):
                if symbol in code:
                    offenders.append("{0}/{1}: {2}".format(package, path.name, symbol))
    assert not offenders, offenders


def test_the_physical_radius_is_the_one_the_science_uses():
    """Density and gravity must come from the catalogued radius."""
    import astropy.units as u

    from astro_explorer.data.schema import build_planet_record

    record = build_planet_record(
        {"pl_name": "T b", "hostname": "T", "pl_rade": 1.0, "pl_bmasse": 1.0}
    )
    # Earth's values, from the physical radius - not from anything a
    # renderer chose.
    assert record.bulk_density.value_in(u.g / u.cm**3) == pytest.approx(5.495, rel=1e-3)
    assert record.surface_gravity.value_in(u.m / u.s**2) == pytest.approx(9.798, rel=1e-3)


def test_scaling_the_display_does_not_move_the_planet():
    """Exaggerating radii must not perturb any orbital distance."""
    import numpy as np

    from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
    from astro_explorer.rendering.scene_builder import build_frame_scene

    try:
        catalog = load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")

    slice_ = build_slice("HD 80606", catalog)
    anomalies = slice_.mean_anomalies(2458882.344)

    exaggerated = build_frame_scene(
        slice_.frame, slice_.star, slice_.planets, mean_anomalies=anomalies
    )
    true_scale = build_frame_scene(
        slice_.frame, slice_.star, slice_.planets, mean_anomalies=anomalies,
        exaggerate=False,
    )

    assert exaggerated.planets[0].radius_display != true_scale.planets[0].radius_display
    assert np.allclose(
        exaggerated.planets[0].position_local, true_scale.planets[0].position_local
    )
    assert np.allclose(
        exaggerated.orbits[0].points_local, true_scale.orbits[0].points_local
    )


def test_the_true_scale_mode_says_it_is_to_scale():
    from astro_explorer.rendering.scene_builder import DisplayScale

    assert "common scale" in DisplayScale.for_system(0.005, 0.03, exaggerate=False).describe()
    assert "not to scale" in DisplayScale.for_system(0.005, 0.03).describe(11.6)
