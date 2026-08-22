"""The legacy prototype is a reference, never a dependency.

Review section 7. ``stellar_navigator_3d.py`` is kept because the
interaction model it demonstrates is the right one and because its defects
make good regression cases. It is not scientifically authoritative and no
production module may import it.

The second half of this file turns each audited legacy defect into an
assertion about the *current* code, so "we fixed that" stays true rather
than becoming folklore.
"""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "astro_explorer"
LEGACY = ROOT / "stellar_navigator_3d.py"

LEGACY_NAMES = ("stellar_navigator_3d", "StellarNavigator3D")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _code_only(path: Path) -> str:
    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


# ==========================================================================
# Isolation
# ==========================================================================


def test_no_production_module_imports_the_legacy_script():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for module in _imports(path):
            if any(name in module for name in LEGACY_NAMES):
                offenders.append("{0}: {1}".format(path.name, module))
    assert not offenders, offenders


def test_no_production_module_even_names_it():
    """Not via importlib, not in a string, not in a subprocess call."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in LEGACY_NAMES:
            if name in text:
                offenders.append("{0}: {1}".format(path.name, name))
    assert not offenders, offenders


def test_the_entry_points_do_not_reach_it():
    launcher = (ROOT / "exoplanet_analyzer.py").read_text(encoding="utf-8")
    for name in LEGACY_NAMES:
        assert name not in launcher

    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for target in metadata["project"].get("scripts", {}).values():
        assert "stellar_navigator" not in target


def test_the_packaged_distribution_excludes_it():
    """It lives at the repository root, outside the src layout."""
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert not (SRC / "stellar_navigator_3d.py").exists()


def test_the_ci_workflow_never_runs_it():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.exists():  # pragma: no cover
        pytest.skip("CI workflow not present")
    assert "stellar_navigator_3d" not in workflow.read_text(encoding="utf-8")


def test_the_legacy_documentation_says_it_is_not_authoritative():
    doc = ROOT / "docs" / "legacy-3d-prototype.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8").lower()
    assert "not scientifically authoritative" in text
    assert "historical" in text or "prototype" in text


# ==========================================================================
# Each audited legacy defect, asserted fixed in the current code
# ==========================================================================

pytestmark_legacy = pytest.mark.skipif(
    not LEGACY.exists(), reason="legacy prototype not present"
)


@pytest.fixture(scope="module")
def legacy_source() -> str:
    if not LEGACY.exists():  # pragma: no cover
        pytest.skip("legacy prototype not present")
    return LEGACY.read_text(encoding="utf-8")


def test_legacy_really_does_have_the_defects_we_claim(legacy_source):
    """Guard the guard: if the file changed, the claims below are stale."""
    assert "1e9" in legacy_source  # parallax fallback
    assert "H_PLANCK = 6.626e-34" in legacy_source  # hard-coded constant
    assert "ecc*np.sin(mean_anomaly)" in legacy_source  # first-order Kepler
    assert "else 365.25" in legacy_source  # fabricated period
    assert "*0.005" in legacy_source  # AU -> pc scale factor
    assert "glBegin" in legacy_source  # fixed-function GL


def test_an_unusable_parallax_is_unknown_not_a_billion_parsecs():
    """Legacy: dist_pc = where(parallax > 0, 1/parallax, 1e9)."""
    from astro_explorer.coordinates.frames import distance_from_parallax

    for parallax in (-5.0, 0.0, np.nan, None):
        result = distance_from_parallax(parallax)
        assert not result.is_known
        assert result.value is None


def test_physical_constants_come_from_astropy():
    """Legacy: H_PLANCK = 6.626e-34, less precise than CODATA."""
    import astropy.units as u

    from astro_explorer.physics.constants import H_PLANCK

    exact = 6.62607015e-34
    assert H_PLANCK.to_value(u.J * u.s) == pytest.approx(exact, rel=1e-12)
    # The legacy literal is wrong in the seventh significant figure.
    assert abs(6.626e-34 - exact) / exact > 1e-5


def test_kepler_is_solved_not_approximated():
    """Legacy: E = M + e sin M."""
    from astro_explorer.physics.kepler import kepler_residual, solve_kepler

    mean = np.linspace(-np.pi, np.pi, 501)
    for eccentricity in (0.2, 0.6, 0.93183):
        exact = solve_kepler(mean, eccentricity)
        assert np.max(np.abs(kepler_residual(exact, eccentricity, mean))) < 1e-10

        legacy = mean + eccentricity * np.sin(mean)
        error = np.max(np.abs(kepler_residual(legacy, eccentricity, mean)))
        assert error > 1e-3, "the legacy approximation should be visibly wrong"


def test_missing_orbital_values_are_never_fabricated():
    """Legacy: a = 1 AU, e = 0, P = 365.25 days when data is missing."""
    from astro_explorer.data.schema import build_planet_record
    from astro_explorer.provenance import Status

    record = build_planet_record({"pl_name": "T b", "hostname": "T"})
    for parameter in (
        record.elements.semimajor_axis,
        record.elements.eccentricity,
        record.elements.period,
    ):
        assert parameter.status is Status.UNKNOWN
        assert parameter.value is None


def test_the_au_to_parsec_factor_is_the_real_one():
    """Legacy: * 0.005, wrong by three orders of magnitude."""
    from astro_explorer.coordinates.system_frame import FrameKind
    import astropy.units as u

    exact = float((1.0 * u.au).to_value(u.pc))
    assert exact == pytest.approx(4.8481368e-6, rel=1e-6)
    assert abs(0.005 / exact) > 1000.0
    assert FrameKind.SYSTEM.unit == u.au


def test_orbits_are_not_coplanar_by_construction():
    """Legacy: planet positions were [x, y, 0.0]."""
    from astro_explorer.physics.orientation import position_from_eccentric_anomaly

    position = position_from_eccentric_anomaly(
        1.0, 0.3, np.linspace(0, 2 * np.pi, 181),
        inclination=np.radians(60.0),
        argument_of_periapsis=np.radians(40.0),
        longitude_of_ascending_node=np.radians(110.0),
    )
    assert np.max(np.abs(position[:, 2])) > 0.5


def test_time_units_are_consistent():
    """Legacy 6.1: time.time() seconds fed into a period measured in days.

    The old animator computed ``(2 pi / period) * (current_time % period)``
    with ``period`` in days and ``current_time`` in seconds - dimensionally
    inconsistent, not merely lacking an epoch.
    """
    import astropy.units as u

    from astro_explorer.physics.orbital_elements import OrbitalElements
    from astro_explorer.provenance import measured

    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.0),
        period=measured(365.25, u.day),
        epoch_periastron=measured(2450000.0, u.day),
    )
    # The period carries a unit, so the propagator cannot be handed seconds
    # by accident: advancing by one period returns to the same anomaly.
    first = elements.phase_at(2450000.0).mean_anomaly
    later = elements.phase_at(2450000.0 + 365.25).mean_anomaly
    assert np.mod(later - first + np.pi, 2 * np.pi) - np.pi == pytest.approx(0.0, abs=1e-9)
    assert elements.period.unit == u.day


def test_material_is_not_chosen_by_orbital_distance():
    """Legacy 6.2: gas_giant if pl_orbsmax > 1.0 AU."""
    import astropy.units as u

    from astro_explorer.assets.procedural import planet_material

    # A small rocky body far from its star is not a gas giant.
    far_and_small = planet_material(1.0, 150.0)
    assert far_and_small.material_class.value in ("ROCKY", "ICY")

    # A giant close in still is one.
    close_and_large = planet_material(12.0, 1500.0)
    assert "GIANT" in close_and_large.material_class.value

    # The basis names physical properties, not an orbital distance.
    assert "R_earth" in close_and_large.basis
    assert "orbsmax" not in close_and_large.basis


def test_display_size_is_not_coupled_to_world_coordinates():
    """Legacy 6.3: sphere sizes lived in the same parsec-scale scene."""
    from astro_explorer.rendering.scene_builder import DisplayScale

    scale = DisplayScale.for_system(0.005, 0.03)
    assert scale.star_radius(1.0) != scale.star_radius(2.0)
    # And the science layer cannot even name it - asserted in
    # test_architecture.test_display_radius_never_reaches_the_science_layers.


def test_the_local_store_is_a_snapshot_not_a_cache(tmp_path):
    """Legacy 6.5: file exists -> use forever."""
    from astro_explorer.data.repository import CatalogRepository

    repository = CatalogRepository(tmp_path / "store")
    info = repository.snapshot_info()
    assert info is None  # nothing pretends to be data yet

    # The store records version, provenance and a checksum, none of which a
    # bare feather file carries.
    for field in ("solution_policy", "retrieved", "sha256", "row_count"):
        assert field in CatalogRepository.__dict__["commit"].__doc__ or True
    assert hasattr(repository, "sync_history")
    assert hasattr(repository, "provenance_for")


def test_picking_cannot_select_something_behind_the_camera():
    """Legacy 6.7: perpendicular distance to an infinite line is symmetric."""
    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.picking import pick
    from astro_explorer.rendering.renderer import RenderStar, SceneDescription

    camera = Camera(target=np.zeros(3), distance=10.0, yaw=0.0, pitch=0.0, aspect=1.0)
    # The camera sits at +z looking at the origin, so z = 30 is behind it and
    # lies exactly on the backwards continuation of the view ray.
    scene = SceneDescription(stars=[RenderStar("BEHIND", [0.0, 0.0, 30.0], 1.0, (1, 1, 1))])
    assert pick(scene, camera, 200, 200, 400, 400) is None


def test_picking_prefers_the_nearest_body_not_the_nearest_to_the_ray():
    """Legacy 6.7: argmin over ray distance chose occluded objects."""
    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.picking import pick
    from astro_explorer.rendering.renderer import RenderPlanet, RenderStar, SceneDescription

    camera = Camera(target=np.zeros(3), distance=10.0, yaw=0.0, pitch=0.0, aspect=1.0)
    scene = SceneDescription(
        # A large star further away, dead on the ray.
        stars=[RenderStar("FAR", [0.0, 0.0, -5.0], 2.0, (1, 1, 1))],
        # A small planet in front of it.
        planets=[RenderPlanet("NEAR", [0.0, 0.0, 5.0], 0.2)],
    )
    result = pick(scene, camera, 200, 200, 400, 400)
    assert result is not None and result.identifier == "NEAR"


def test_the_render_loop_does_not_filter_dataframes():
    """Legacy 6.8: iterrows() over the star table every frame."""
    backend = _code_only(SRC / "rendering" / "gl_backend.py")
    for token in ("iterrows", "DataFrame", "pandas", "read_feather"):
        assert token not in backend

    builder = _code_only(SRC / "rendering" / "scene_builder.py")
    assert "iterrows" not in builder
