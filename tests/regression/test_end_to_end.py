"""End-to-end tests: catalogue row to rendered scene, without a GUI or GL.

This is the acceptance test from roadmap section 27:

    No value shown as measured may actually be a visualization fallback.
    No orbital distance may depend on an arbitrary scale factor.
    No atmospheric spectrum may rely on positional column guesses.
    The program must work from a validated local snapshot with the network
    disabled.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astro_explorer.app.state import AppState
from astro_explorer.coordinates.floating_origin import AU_TO_PC, SceneGraph
from astro_explorer.data.nasa_archive import SolutionPolicy
from astro_explorer.data.repository import CatalogRepository
from astro_explorer.data.synchronizer import synchronize
from astro_explorer.physics.ephemeris import TimeController, TimeMode
from astro_explorer.provenance import Status
from astro_explorer.rendering.scene_builder import build_system_scene, orbit_path

# A realistic, well-characterised system: HD 219134 style, plus a planet with
# deliberately missing elements.
CATALOG_ROWS = [
    {
        "pl_name": "Test b",
        "hostname": "Test",
        "pl_orbper": 3.0937,
        "pl_orbsmax": 0.03876,
        "pl_orbeccen": 0.0,
        "pl_orbincl": 85.05,
        "pl_orblper": 90.0,
        "pl_tranmid": 2457126.7,
        "pl_rade": 1.602,
        "pl_bmasse": 4.74,
        "pl_eqt": 1015.0,
        "st_teff": 4699.0,
        "st_rad": 0.778,
        "st_mass": 0.81,
        "ra": 348.34,
        "dec": 57.16,
        "sy_dist": 6.53,
        "sy_plx": 153.0,
        "discoverymethod": "Transit",
        "disc_year": 2015,
        "disc_facility": "Spitzer",
    },
    {
        "pl_name": "Test c",
        "hostname": "Test",
        # No semimajor axis and no eccentricity published; a period and a
        # stellar mass are available, so a is derivable but e is not.
        "pl_orbper": 6.765,
        "pl_rade": 1.511,
        "st_teff": 4699.0,
        "st_rad": 0.778,
        "st_mass": 0.81,
        "ra": 348.34,
        "dec": 57.16,
        "sy_dist": 6.53,
        "sy_plx": 153.0,
        "discoverymethod": "Radial Velocity",
        "disc_year": 2015,
    },
    {
        "pl_name": "Test d",
        "hostname": "Test",
        # Nothing usable: no period, no axis, no stellar mass reference.
        "pl_rade": 2.1,
        "st_teff": 4699.0,
        "st_rad": 0.778,
        "ra": 348.34,
        "dec": 57.16,
        "sy_dist": 6.53,
    },
]


@pytest.fixture()
def state() -> AppState:
    frame = pd.DataFrame(CATALOG_ROWS)
    return AppState(catalog=frame, policy=SolutionPolicy.COMPOSITE)


# -- acceptance criteria -----------------------------------------------------


def test_no_value_shown_as_measured_is_a_visualisation_fallback(state):
    for name in ("Test b", "Test c", "Test d"):
        record = state.record(name)
        for parameter in (
            record.elements.semimajor_axis,
            record.elements.eccentricity,
            record.elements.period,
            record.radius_earth,
            record.mass_earth,
            record.host.effective_temperature,
        ):
            if parameter.status is Status.MEASURED:
                assert parameter.is_known
                assert not parameter.is_assumed


def test_derived_and_measured_axes_are_distinguishable(state):
    assert state.record("Test b").elements.semimajor_axis.status is Status.MEASURED
    assert state.record("Test c").elements.semimajor_axis.status is Status.DERIVED
    assert not state.record("Test d").elements.semimajor_axis.is_known


def test_no_orbital_distance_depends_on_an_arbitrary_scale_factor(state):
    """Positions must be reproducible from the published axis alone."""
    record = state.record("Test b")
    path = orbit_path(record.elements)
    radii = np.linalg.norm(path, axis=1)
    axis = record.elements.semimajor_axis.value_in(u.au)
    assert np.isclose(radii.mean(), axis, rtol=0.05)
    assert np.isclose(radii.max(), axis, rtol=1e-6)  # circular orbit


def test_the_scene_places_planets_in_au_not_scaled_parsecs(state):
    records = state.system_records("Test")
    anomalies = {"Test b": 0.0, "Test c": np.pi}
    scene = build_system_scene(records, mean_anomalies=anomalies)

    positions = {planet.identifier: planet.position_local for planet in scene.planets}
    distance = float(np.linalg.norm(positions["Test b"]))
    assert np.isclose(distance, 0.03876, rtol=1e-4)
    # The prototype's factor would have produced a number ~1000x too large.
    assert distance < 1.0


# -- scene construction ------------------------------------------------------


def test_a_planet_without_an_axis_is_not_drawn_and_is_explained(state):
    records = state.system_records("Test")
    scene = build_system_scene(records, mean_anomalies={"Test b": 0.0, "Test c": 0.0})

    drawn = {planet.identifier for planet in scene.planets}
    assert "Test d" not in drawn
    assert any("Test d" in note and "not drawn" in note for note in scene.annotations)


def test_an_assumed_orbit_is_dashed_and_annotated(state):
    records = state.system_records("Test")
    scene = build_system_scene(records, mean_anomalies={"Test c": 1.0})

    orbits = {orbit.identifier: orbit for orbit in scene.orbits}
    assert orbits["Test c:orbit"].dashed  # unknown eccentricity and orientation
    assert any("assumed" in note for note in scene.annotations)


def test_size_exaggeration_is_disclosed(state):
    scene = build_system_scene(state.system_records("Test"), mean_anomalies={"Test b": 0.0})
    assert any("exaggerated" in note for note in scene.annotations)


def test_a_planet_with_no_phase_gets_an_orbit_but_no_body(state):
    records = [state.record("Test b")]
    scene = build_system_scene(records, mean_anomalies={})
    assert scene.orbits
    assert not scene.planets
    assert any("phase not constrained" in note for note in scene.annotations)


def test_the_star_colour_comes_from_the_effective_temperature(state):
    scene = build_system_scene(state.system_records("Test"), mean_anomalies={"Test b": 0.0})
    star = scene.stars[0]
    assert star.temperature_k == 4699.0
    # A 4700 K star is orange: red channel above blue.
    assert star.color[0] > star.color[2]


def test_the_scene_graph_never_mixes_au_and_parsec_buffers():
    graph = SceneGraph()
    host = np.array([2.0, 0.0, 0.0])
    graph.enter_system(host)
    planet = graph.planet_render_position(host, [1.0, 0.0, 0.0])
    assert np.isclose(planet[0], 1.0)

    graph.enter_galaxy()
    planet_galaxy = graph.planet_render_position(host, [1.0, 0.0, 0.0])
    assert np.isclose(planet_galaxy[0], 2.0 + AU_TO_PC, atol=1e-5)


# -- time models -------------------------------------------------------------


def _angular_rate(controller, elements, at=1.0, step=1e-3):
    """Observed dM/dt, wrapped so it survives the +/-pi boundary."""
    before, _ = controller.mean_anomaly(elements, at)
    after, _ = controller.mean_anomaly(elements, at + step)
    delta = np.mod(after - before + np.pi, 2.0 * np.pi) - np.pi
    return delta / step


def test_physical_time_gives_different_planets_different_rates(state):
    """The original animation gave every planet the same visual period."""
    inner = state.record("Test b").elements
    outer = state.record("Test c").elements
    controller = TimeController(mode=TimeMode.SCALED, scale_days_per_second=1.0)

    inner_rate = _angular_rate(controller, inner)
    outer_rate = _angular_rate(controller, outer)
    assert not np.isclose(inner_rate, outer_rate)

    # The rates must be in inverse proportion to the orbital periods.
    expected = outer.period.value_in(u.day) / inner.period.value_in(u.day)
    assert np.isclose(inner_rate / outer_rate, expected, rtol=1e-4)


def test_normalized_mode_is_flagged_as_assumed(state):
    controller = TimeController(mode=TimeMode.NORMALIZED, normalized_period_seconds=8.0)
    anomaly, assumed = controller.mean_anomaly(state.record("Test b").elements, 4.0)
    assert assumed
    assert np.isclose(anomaly, np.pi)
    assert "not physical" in controller.describe()


def test_normalized_mode_gives_every_planet_the_same_period(state):
    """It is an educational mode and says so; it is not the default."""
    controller = TimeController(mode=TimeMode.NORMALIZED, normalized_period_seconds=8.0)
    inner, _ = controller.mean_anomaly(state.record("Test b").elements, 2.0)
    outer, _ = controller.mean_anomaly(state.record("Test c").elements, 2.0)
    assert np.isclose(inner, outer)
    assert TimeController().mode is not TimeMode.NORMALIZED


def test_real_time_mode_requires_a_date(state):
    controller = TimeController(mode=TimeMode.REAL)
    with pytest.raises(ValueError, match="BJD"):
        controller.simulated_bjd(0.0)


def test_transit_epoch_yields_a_computable_position(state):
    record = state.record("Test b")
    assert record.elements.can_compute_current_position
    controller = TimeController(mode=TimeMode.SCALED)
    anomaly, assumed = controller.mean_anomaly(record.elements, 0.0, now_bjd=None)
    assert anomaly is not None
    assert not assumed


# -- offline operation -------------------------------------------------------


def test_the_program_runs_from_a_snapshot_with_the_network_disabled(tmp_path):
    repository = CatalogRepository(tmp_path / "store")
    frame = pd.DataFrame(
        [dict(row, pl_name="P{0}".format(i), hostname="H{0}".format(i)) for i, row in
         enumerate(CATALOG_ROWS * 2000)]
    )
    result = synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: frame)
    assert result.succeeded

    def no_network(_policy):
        raise ConnectionError("network disabled")

    reopened = CatalogRepository(repository.root)
    offline = reopened.load()
    assert offline is not None and len(offline) == len(frame)

    failed = synchronize(reopened, SolutionPolicy.COMPOSITE, fetcher=no_network)
    assert not failed.succeeded
    assert reopened.load() is not None

    state = AppState(catalog=offline, offline=True)
    assert state.is_loaded
    assert state.record("P0") is not None


# -- reporting ---------------------------------------------------------------


def test_the_overview_never_presents_an_assumption_as_a_fact(state):
    text = "\n".join(state.record("Test c").describe())
    assert "Eccentricity:      unknown" in text
    assert "derived" in text  # the semimajor axis says so


def test_the_data_report_names_the_solution_policy(state):
    text = "\n".join(state.data_report())
    assert "pscomppars" in text
    assert "different publications" in text


def test_kepler_residual_is_reported_for_a_published_axis(state):
    residual = state.record("Test b").kepler_residual
    assert residual is not None
    assert abs(residual) < 0.05


def test_no_residual_is_reported_when_the_axis_was_derived(state):
    assert state.record("Test c").kepler_residual is None
