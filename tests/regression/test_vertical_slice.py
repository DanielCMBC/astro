"""The one-star-one-planet vertical slice, end to end.

Reference target: **HD 80606 b**, ``e = 0.93183``. The whole chain runs from
the committed local snapshot to float32 render primitives, and every stage is
checked against a value that can be derived independently.

The extreme eccentricity is the point. At ``e = 0.93`` the apoapsis distance
is 28 times the periapsis distance and the periapsis speed is 28 times the
apoapsis speed, so an error anywhere in the chain - the Kepler solver, the
rotation order, the propagation rate - shows up as a large, unmistakable
discrepancy rather than a rounding difference.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.vertical_slice import (
    PRIMARY_TARGET,
    build_slice,
    load_reference_catalog,
)
from astro_explorer.coordinates.system_frame import FrameKind, SystemFrame
from astro_explorer.data.nasa_archive import SolutionPolicy
from astro_explorer.physics.state_vectors import (
    expected_specific_energy,
    specific_orbital_energy,
)
from astro_explorer.provenance import Status
from astro_explorer.rendering.scene_builder import build_frame_scene

# Published values, Pearson et al. 2022, as carried in the snapshot.
A_AU = 0.4603
ECC = 0.93183
INCLINATION_DEG = 89.24
ARG_PERIAPSIS_DEG = -58.887
PERIOD_DAYS = 111.436765
T_PERIASTRON = 2458882.344
STELLAR_MASS = 1.05


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def slice_(catalog):
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def planet(slice_):
    record = slice_.planet(PRIMARY_TARGET)
    assert record is not None
    return record


# ==========================================================================
# Stage 1: local data -> records, with units and provenance
# ==========================================================================


def test_the_snapshot_is_a_default_solution_not_a_composite(catalog):
    """HD 80606 b has eight published solutions in `ps`; exactly one default."""
    assert catalog.attrs["solution_policy"] == SolutionPolicy.DEFAULT_SOLUTION.value
    rows = catalog[catalog["pl_name"] == PRIMARY_TARGET]
    assert len(rows) == 1
    if "default_flag" in rows:
        assert int(rows.iloc[0]["default_flag"]) == 1


def test_elements_are_measured_with_units(planet):
    elements = planet.elements
    for parameter in (elements.semimajor_axis, elements.eccentricity, elements.period):
        assert parameter.status is Status.MEASURED

    assert np.isclose(elements.semimajor_axis.value_in(u.au), A_AU)
    assert np.isclose(elements.eccentricity.value, ECC)
    assert np.isclose(elements.period.value_in(u.day), PERIOD_DAYS)


def test_angles_were_converted_from_degrees_once(planet):
    elements = planet.elements
    assert np.isclose(elements.inclination.value_in(u.deg), INCLINATION_DEG)
    assert np.isclose(elements.argument_of_periastron.value_in(u.deg), ARG_PERIAPSIS_DEG)
    # Stored in radians internally.
    assert elements.inclination.unit == u.rad


def test_the_provenance_names_the_publication(slice_):
    text = "\n".join(slice_.describe_provenance())
    assert "Pearson" in text
    assert "default solution" in text.lower()


# ==========================================================================
# Stage 2: unknown Omega must not be zero Omega
# ==========================================================================


def test_the_ascending_node_is_unknown_not_zero(planet):
    """The headline policy of this milestone."""
    node = planet.elements.longitude_of_ascending_node
    assert node.status is Status.UNKNOWN
    assert node.value is None
    assert not planet.elements.orientation_known


def test_inclination_and_argument_of_periapsis_are_genuinely_measured(planet):
    assert planet.elements.inclination.status is Status.MEASURED
    assert planet.elements.argument_of_periastron.status is Status.MEASURED


def test_display_normalisation_is_zero_and_is_labelled(planet):
    display = planet.elements.for_display()
    node = display.longitude_of_ascending_node
    assert node.value == 0.0
    assert node.status is Status.ASSUMED_FOR_VISUALIZATION
    assert not node.is_scientific


def test_normalisation_does_not_contaminate_the_record(planet):
    planet.elements.for_display()
    assert planet.elements.longitude_of_ascending_node.status is Status.UNKNOWN


def test_the_report_states_both_the_unknown_and_the_normalisation(slice_, planet):
    text = "\n".join(slice_.describe_orbit(planet, T_PERIASTRON))
    assert "Asc. node" in text
    assert "UNKNOWN" in text
    assert "display normalisation" in text
    assert "assumed for visualisation" in text
    # Measured angles are shown as measured, with their uncertainty and
    # without a normalisation line.
    assert "89.24" in text and "deg" in text
    inclination_line = next(line for line in text.splitlines() if "Inclination" in line)
    assert "UNKNOWN" not in inclination_line


# ==========================================================================
# Stage 3: Kepler propagation and the 3D transform
# ==========================================================================


def test_at_periastron_the_planet_is_exactly_at_periapsis(slice_, planet):
    """T_peri is published, so this is a hard check on the phase model."""
    state = slice_.state(planet, T_PERIASTRON)
    assert state is not None
    assert np.isclose(float(state.radius), A_AU * (1 - ECC), rtol=1e-9)


def test_half_a_period_later_the_planet_is_at_apoapsis(slice_, planet):
    state = slice_.state(planet, T_PERIASTRON + PERIOD_DAYS / 2.0)
    assert np.isclose(float(state.radius), A_AU * (1 + ECC), rtol=1e-6)


def test_a_full_period_returns_to_the_same_position(slice_, planet):
    first = slice_.state(planet, T_PERIASTRON)
    later = slice_.state(planet, T_PERIASTRON + PERIOD_DAYS)
    assert np.allclose(first.position, later.position, atol=1e-9)


def test_the_apoapsis_to_periapsis_ratio_is_extreme(planet):
    ratio = planet.elements.apoapsis.value / planet.elements.periapsis.value
    assert np.isclose(ratio, (1 + ECC) / (1 - ECC), rtol=1e-9)
    assert ratio > 28.0


def test_the_orbit_is_genuinely_three_dimensional(slice_, planet):
    """i = 89.24 deg: the orbit must be nearly edge-on, not flat in XY."""
    times = T_PERIASTRON + np.linspace(0.0, PERIOD_DAYS, 400)
    positions = np.array([slice_.state(planet, t).position for t in times])
    out_of_plane = np.max(np.abs(positions[:, 2]))
    in_plane = np.max(np.abs(positions[:, 0]))
    assert out_of_plane > 0.5 * A_AU
    assert out_of_plane > in_plane  # nearly edge-on


def test_the_orbit_normal_matches_the_measured_inclination(slice_, planet):
    times = T_PERIASTRON + np.linspace(0.0, PERIOD_DAYS, 200)
    positions = np.array([slice_.state(planet, t).position for t in times])
    normal = np.cross(positions[0], positions[50])
    normal /= np.linalg.norm(normal)
    tilt = np.degrees(np.arccos(abs(np.dot(normal, [0.0, 0.0, 1.0]))))
    assert np.isclose(tilt, INCLINATION_DEG, atol=0.01)


# ==========================================================================
# Stage 4: velocity and energy
# ==========================================================================


def test_the_gravitational_parameter_uses_the_published_stellar_mass(slice_):
    from astro_explorer.physics.state_vectors import MU_SUN_AU3_PER_DAY2

    assert slice_.mu is not None
    assert slice_.mu / MU_SUN_AU3_PER_DAY2 == pytest.approx(STELLAR_MASS, rel=5e-3)


def test_energy_is_conserved_around_the_whole_orbit(slice_, planet):
    times = T_PERIASTRON + np.linspace(0.0, PERIOD_DAYS, 200)
    axis = planet.elements.semimajor_axis.value_in(u.au)
    expected = expected_specific_energy(axis, slice_.mu)
    for time in times:
        state = slice_.state(planet, time)
        assert np.isclose(specific_orbital_energy(state, slice_.mu), expected, rtol=1e-10)


def test_the_energy_check_is_reported_and_tiny(slice_, planet):
    check = slice_.energy_check(planet, T_PERIASTRON)
    assert check is not None
    assert check["relative_error"] < 1e-10


def test_periastron_speed_matches_the_analytic_value(slice_, planet):
    """v_peri = sqrt(mu (1+e) / (a (1-e))) - about 240 km/s here."""
    state = slice_.state(planet, T_PERIASTRON)
    expected = np.sqrt(slice_.mu * (1 + ECC) / (A_AU * (1 - ECC)))
    assert np.isclose(float(state.speed), expected, rtol=1e-9)

    speed_kms = float((float(state.speed) * u.au / u.day).to_value(u.km / u.s))
    assert 230.0 < speed_kms < 250.0


def test_the_speed_ratio_matches_the_eccentricity(slice_, planet):
    fast = slice_.state(planet, T_PERIASTRON).speed
    slow = slice_.state(planet, T_PERIASTRON + PERIOD_DAYS / 2.0).speed
    assert np.isclose(float(fast / slow), (1 + ECC) / (1 - ECC), rtol=1e-5)


# ==========================================================================
# Stage 5: the SystemFrame
# ==========================================================================


def test_the_slice_renders_in_a_system_frame(slice_):
    assert isinstance(slice_.frame, SystemFrame)
    assert slice_.frame.kind is FrameKind.SYSTEM
    assert slice_.frame.unit == u.au


def test_the_star_sits_at_the_frame_origin(slice_):
    assert np.array_equal(slice_.frame.star_position().values, np.zeros(3))


def test_the_framed_position_equals_the_raw_orbital_vector(slice_, planet):
    """No scale factor is applied between physics and frame."""
    state = slice_.state(planet, T_PERIASTRON)
    framed = slice_.framed_position(planet, T_PERIASTRON)
    assert np.array_equal(framed.values, state.position)
    assert framed.kind is FrameKind.SYSTEM


def test_the_frame_knows_where_the_system_is_without_using_it_locally(slice_):
    """Distance from Earth is recorded but never enters the local geometry."""
    assert slice_.frame.origin_pc is not None
    assert np.linalg.norm(slice_.frame.origin_pc) > 60.0  # ~66.5 pc
    assert np.array_equal(slice_.frame.star_position().values, np.zeros(3))


# ==========================================================================
# Stage 6: the render boundary
# ==========================================================================


@pytest.fixture()
def scene(slice_):
    return build_frame_scene(
        slice_.frame,
        slice_.star,
        slice_.planets,
        mean_anomalies=slice_.mean_anomalies(T_PERIASTRON),
    )


def test_the_scene_has_one_star_one_planet_and_one_orbit(scene):
    assert len(scene.stars) == 1
    assert len(scene.planets) == 1
    assert len(scene.orbits) == 1
    assert scene.unit_label == "AU"


def test_render_positions_are_float32_and_finite(scene):
    for body in list(scene.stars) + list(scene.planets):
        assert body.position_local.dtype == np.float32
        assert np.all(np.isfinite(body.position_local))
    assert scene.orbits[0].points_local.dtype == np.float32


def test_the_rendered_planet_position_is_the_orbital_vector(scene, slice_, planet):
    state = slice_.state(planet, T_PERIASTRON)
    rendered = scene.planets[0].position_local
    assert np.allclose(rendered, state.position, rtol=1e-6)
    assert np.isclose(np.linalg.norm(rendered), A_AU * (1 - ECC), rtol=1e-5)


def test_the_orbit_is_dashed_because_the_node_is_assumed(scene):
    assert scene.orbits[0].dashed


def test_the_scene_carries_no_orbital_elements(scene):
    """The renderer is told a position, never how to compute one."""
    forbidden = {"eccentricity", "semimajor_axis", "period", "mean_anomaly", "inclination"}
    for body in list(scene.stars) + list(scene.planets):
        assert not (set(type(body).__dataclass_fields__) & forbidden)


def test_the_planet_is_never_drawn_larger_than_its_star(scene):
    """A Jupiter-radius planet must not dwarf a solar-radius host."""
    star_radius = scene.stars[0].radius_display
    for planet_primitive in scene.planets:
        assert planet_primitive.radius_display < star_radius


def test_the_star_is_not_drawn_wider_than_periapsis(scene, planet):
    """Otherwise the planet appears to orbit inside its host."""
    assert scene.stars[0].radius_display < planet.elements.periapsis.value


def test_the_size_exaggeration_is_disclosed(scene):
    text = " ".join(scene.annotations)
    assert "exaggerated" in text
    assert "not to scale" in text


def test_the_assumed_orientation_is_disclosed(scene):
    assert any("assumed" in note for note in scene.annotations)


# ==========================================================================
# Sanity system: WASP-39 b, where the eccentricity is unpublished
# ==========================================================================


def test_wasp39b_has_a_measured_inclination_but_no_eccentricity(catalog):
    other = build_slice("WASP-39", catalog)
    record = other.planet("WASP-39 b")
    assert record.elements.inclination.status is Status.MEASURED
    assert record.elements.eccentricity.status is Status.UNKNOWN

    display = record.elements.for_display()
    assert display.eccentricity.value == 0.0
    assert display.eccentricity.status is Status.ASSUMED_FOR_VISUALIZATION


def test_wasp39b_still_produces_a_renderable_scene(catalog):
    other = build_slice("WASP-39", catalog)
    scene = build_frame_scene(
        other.frame, other.star, other.planets,
        mean_anomalies=other.mean_anomalies(2455343.0),
    )
    assert len(scene.planets) == 1
    assert scene.orbits[0].dashed  # circular assumption + unknown node
    assert any("assumed" in note for note in scene.annotations)


def test_a_circular_assumption_gives_a_constant_radius(catalog):
    other = build_slice("WASP-39", catalog)
    record = other.planet("WASP-39 b")
    times = 2455343.0 + np.linspace(0.0, 4.055, 50)
    radii = [float(other.state(record, t).radius) for t in times]
    assert np.allclose(radii, radii[0], rtol=1e-9)
