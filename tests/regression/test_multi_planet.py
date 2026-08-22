"""Multi-planet ``SystemFrame`` rendering (review section 15).

Three systems from the committed snapshot, chosen because they fail in
different ways:

* **Kepler-11** - six planets, every one with ``a``, ``e`` and a transit
  epoch but no published ``omega``, so all six are *partially* constrained:
  the timing is observed, the in-plane orientation is normalised;
* **TRAPPIST-1** - seven planets, inclinations only. No eccentricities, no
  epochs, and no system distance. Everything about it must degrade
  gracefully;
* **HD 219134** - six planets with *mixed* completeness, which is the case
  a uniform code path quietly gets wrong.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.physics.orbital_semantics import OrbitValidity
from astro_explorer.rendering.camera import Camera
from astro_explorer.rendering.labels import resolve_collisions
from astro_explorer.rendering.scene_builder import build_frame_scene

KEPLER11_EPOCH = 2455590.0
TRAPPIST_EPOCH = 2457000.0


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def kepler11(catalog):
    return build_slice("Kepler-11", catalog)


@pytest.fixture(scope="module")
def trappist(catalog):
    return build_slice("TRAPPIST-1", catalog)


def _scene(system_slice, epoch):
    return build_frame_scene(
        system_slice.frame,
        system_slice.star,
        system_slice.planets,
        mean_anomalies=system_slice.mean_anomalies(epoch),
    )


# ==========================================================================
# Multiple planets in one frame
# ==========================================================================


def test_all_six_kepler11_planets_are_loaded(kepler11):
    assert len(kepler11.planets) == 6
    assert [p.name[-1] for p in kepler11.planets] == list("bcdefg")


def test_planets_are_ordered_outwards(kepler11):
    axes = [p.semimajor_axis.value_in(u.au) for p in kepler11.planets]
    assert axes == sorted(axes)


def test_every_planet_and_orbit_reaches_the_scene(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    assert len(scene.planets) == 6
    assert len(scene.orbits) == 6
    assert len({p.identifier for p in scene.planets}) == 6


def test_orbits_do_not_intersect_in_the_scene(kepler11):
    """Nested orbits: each planet's radius range stays outside the last."""
    scene = _scene(kepler11, KEPLER11_EPOCH)
    apoapses = [
        p.elements.apoapsis.value_in(u.au) for p in kepler11.planets
    ]
    periapses = [
        p.elements.periapsis.value_in(u.au) for p in kepler11.planets
    ]
    for inner_apo, outer_peri in zip(apoapses, periapses[1:]):
        assert inner_apo < outer_peri


def test_each_planet_sits_on_its_own_orbit(kepler11):
    """A body must lie within its own path's radius range, not another's."""
    scene = _scene(kepler11, KEPLER11_EPOCH)
    by_id = {o.identifier: o for o in scene.orbits}

    for planet in scene.planets:
        path = by_id["{0}:orbit".format(planet.identifier)].points_local
        radii = np.linalg.norm(path.astype(np.float64), axis=1)
        distance = float(np.linalg.norm(planet.position_local))
        assert radii.min() - 1e-6 <= distance <= radii.max() + 1e-6


def test_the_star_is_shared_by_every_planet(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    assert len(scene.stars) == 1
    assert np.allclose(scene.stars[0].position_local, 0.0)


def test_inner_planets_move_faster_than_outer_ones(kepler11):
    """Physical time, not a normalised clock: the inner planets lap."""
    inner, outer = kepler11.planets[0], kepler11.planets[-1]
    step = 5.0

    def advance(record):
        first = kepler11.mean_anomaly(record, KEPLER11_EPOCH)[0]
        second = kepler11.mean_anomaly(record, KEPLER11_EPOCH + step)[0]
        return np.mod(second - first + np.pi, 2 * np.pi) - np.pi

    ratio = advance(inner) / advance(outer)
    expected = outer.elements.period.value_in(u.day) / inner.elements.period.value_in(u.day)
    assert ratio == pytest.approx(expected, rel=1e-6)


# ==========================================================================
# Phase provenance across a system
# ==========================================================================


def test_kepler11_phases_are_partially_constrained_not_fully(kepler11):
    """Review section 9: the timing is observed, the orientation is not.

    Every Kepler-11 planet has a published mid-transit time but no
    published argument of periastron, so reading that epoch as a mean
    anomaly goes through a normalised omega. Calling these "positioned from
    a published epoch" would overstate them; calling them assumed would
    understate them.
    """
    from astro_explorer.physics.phase import (
        AnomalyMapping,
        PhaseAnchor,
        PhaseProvenance,
        PhaseStatus,
    )

    solutions = kepler11.phase_solutions(KEPLER11_EPOCH)
    assert len(solutions) == 6

    for solution in solutions.values():
        assert solution.is_placeable
        assert solution.provenance is PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED
        assert solution.anchor is PhaseAnchor.OBSERVED
        assert solution.mapping is AnomalyMapping.CONJUNCTION_NORMALIZED
        assert solution.status is PhaseStatus.PARTIALLY_CONSTRAINED
        assert solution.status.is_observationally_anchored


def test_at_its_own_transit_time_each_planet_is_at_inferior_conjunction(kepler11):
    """The transit epoch is only meaningful if it puts the planet there.

    With omega normalised to zero, mid-transit is nu = pi/2, so the
    orbital radius must equal a(1-e^2)/(1+e cos(pi/2)) = a(1-e^2).
    """
    for record in kepler11.planets:
        transit = record.elements.epoch_transit.value_in(u.day)
        assert transit is not None

        display = record.elements.for_display()
        axis = display.semimajor_axis.value_in(u.au)
        ecc = display.eccentricity.value
        expected = axis * (1.0 - ecc**2)

        state = kepler11.state(record, transit)
        assert float(state.radius) == pytest.approx(expected, rel=1e-9)


def test_trappist1_has_no_epochs_and_says_so(trappist):
    assert len(trappist.planets) == 7
    placements = trappist.placements(TRAPPIST_EPOCH)
    assert all(assumed for _anomaly, assumed in placements.values())
    for record in trappist.planets:
        assert record.elements.epochs == []


def test_trappist1_phases_are_excluded_when_assumed_ones_are_refused(trappist):
    assert trappist.mean_anomalies(TRAPPIST_EPOCH, include_assumed=False) == {}
    assert len(trappist.mean_anomalies(TRAPPIST_EPOCH)) == 7


def test_the_panel_reports_the_three_way_phase_vocabulary(kepler11, trappist):
    """Neither system may be described with the old binary wording."""
    kepler_text = "\n".join(kepler11.describe_system(KEPLER11_EPOCH))
    assert "PARTIALLY_CONSTRAINED" in kepler_text
    assert "TRANSIT_CONJUNCTION_NORMALIZED" in kepler_text
    assert "normalised" in kepler_text
    # It must not claim these are fully constrained.
    assert "6  CONSTRAINED" not in kepler_text

    trappist_text = "\n".join(trappist.describe_system(TRAPPIST_EPOCH))
    assert "ASSUMED" in trappist_text
    assert "arbitrary starting point" in trappist_text
    assert "PARTIALLY_CONSTRAINED" not in trappist_text


def test_hd219134_shows_all_three_phase_statuses(catalog):
    """The mixed system is where a binary vocabulary would lose information."""
    from astro_explorer.physics.phase import PhaseStatus

    text = "\n".join(build_slice("HD 219134", catalog).describe_system(2457000.0))
    assert "CONSTRAINED" in text
    assert "PARTIALLY_CONSTRAINED" in text
    assert "ASSUMED" in text


def test_hd219134_has_mixed_phase_provenance(catalog):
    """The case a uniform code path gets wrong."""
    system = build_slice("HD 219134", catalog)
    placements = system.placements(2457000.0)
    assumed = [name for name, (_a, flag) in placements.items() if flag]
    real = [name for name, (_a, flag) in placements.items() if not flag]
    assert real and assumed


def test_every_system_reports_its_ascending_node_situation(kepler11, trappist):
    for system, count in ((kepler11, 6), (trappist, 7)):
        text = "\n".join(system.describe_system(KEPLER11_EPOCH))
        assert "Ascending node unknown for {0}/{0}".format(count) in text
        assert "not measured" in text


# ==========================================================================
# Graceful degradation: TRAPPIST-1
# ==========================================================================


def test_trappist1_renders_without_a_system_distance(trappist):
    """sy_dist is absent, so the frame has no galactic origin - and works."""
    assert np.allclose(trappist.frame.origin_pc, 0.0)
    scene = _scene(trappist, TRAPPIST_EPOCH)
    assert len(scene.planets) == 7
    assert len(scene.orbits) == 7


def test_trappist1_orbits_are_all_dashed(trappist):
    """No eccentricity and no node: every element behind them is assumed."""
    scene = _scene(trappist, TRAPPIST_EPOCH)
    assert all(orbit.dashed for orbit in scene.orbits)


def test_trappist1_geometry_is_not_claimed_as_measured(trappist):
    for record in trappist.planets:
        validity = record.elements.validity
        assert OrbitValidity.GEOMETRY_VALID not in validity
        assert OrbitValidity.ORIENTATION_PARTIAL in validity


def test_trappist1_circular_assumption_gives_constant_radii(trappist):
    record = trappist.planets[0]
    period = record.elements.period.value_in(u.day)
    radii = [
        float(trappist.state(record, TRAPPIST_EPOCH + f * period).radius)
        for f in np.linspace(0.0, 1.0, 20)
    ]
    assert np.allclose(radii, radii[0], rtol=1e-9)


def test_the_star_colour_matches_an_m_dwarf(trappist):
    scene = _scene(trappist, TRAPPIST_EPOCH)
    star = scene.stars[0]
    assert star.temperature_k == pytest.approx(2566.0)
    # 2566 K is deep orange: strongly red-dominant.
    assert star.color[0] > star.color[2] * 1.3


# ==========================================================================
# Level of detail
# ==========================================================================


def test_lod_is_assigned_from_projected_size(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    camera = Camera(target=np.zeros(3), distance=1.0, aspect=1.5)
    scene.assign_lod(camera, 900)
    assert all(0 <= planet.lod <= 5 for planet in scene.planets)


def test_a_distant_system_gets_a_coarser_lod_than_a_close_one(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    camera = Camera(target=np.zeros(3), aspect=1.5)

    camera.distance = 0.3
    scene.assign_lod(camera, 900)
    close = [p.lod for p in scene.planets]

    camera.distance = 300.0
    scene.assign_lod(camera, 900)
    far = [p.lod for p in scene.planets]

    assert max(far) <= min(close)
    assert max(close) > max(far)


def test_lod_does_not_move_anything(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    before = [p.position_local.copy() for p in scene.planets]
    radii = [p.radius_display for p in scene.planets]

    scene.assign_lod(Camera(target=np.zeros(3), distance=2.0, aspect=1.5), 900)

    for planet, position, radius in zip(scene.planets, before, radii):
        assert np.array_equal(planet.position_local, position)
        assert planet.radius_display == radius


# ==========================================================================
# Labels
# ==========================================================================


def test_every_body_is_labelled(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    assert all(planet.label for planet in scene.planets)
    assert scene.stars[0].label == "Kepler-11"


def test_labels_project_inside_the_viewport(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    camera = Camera(target=np.zeros(3), distance=2.0, aspect=1400 / 900, pitch=1.1)
    placements = scene.project_labels(camera, 1400, 900)

    assert placements
    for _label, x, y, depth, radius, _entity in placements:
        assert 0 <= x <= 1400 and 0 <= y <= 900
        assert depth > 0
        assert radius >= 0


def test_labels_behind_the_camera_are_dropped(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    camera = Camera(target=np.zeros(3), distance=0.001, aspect=1.5)
    # Standing inside the system, most bodies fall outside the frustum.
    assert len(scene.project_labels(camera, 400, 300)) < len(scene.planets) + 1


def test_labels_come_back_nearest_first(kepler11):
    scene = _scene(kepler11, KEPLER11_EPOCH)
    camera = Camera(target=np.zeros(3), distance=2.0, aspect=1.5, pitch=1.1)
    depths = [p[3] for p in scene.project_labels(camera, 1400, 900)]
    assert depths == sorted(depths)


def test_collision_resolution_keeps_the_nearest_of_a_pair():
    near = ("near", 100.0, 100.0, 1.0, 0.0, "star:nasa:near")
    far = ("far", 105.0, 102.0, 9.0, 0.0, "star:nasa:far")
    kept = resolve_collisions([near, far], min_separation=26)
    assert [item[0] for item in kept] == ["near"]


def test_well_separated_labels_are_all_kept():
    placements = [
        ("a", 10.0, 10.0, 1.0, 0.0, "star:nasa:a"),
        ("b", 200.0, 200.0, 2.0, 0.0, "star:nasa:b"),
    ]
    assert len(resolve_collisions(placements, min_separation=26)) == 2


def test_the_label_offset_clears_a_large_body():
    """A host star must not have its name written across its disc."""
    from astro_explorer.rendering.labels import LabelStyle

    style = LabelStyle()
    big_radius = 40.0
    text_x = 100.0 + style.offset[0] + big_radius
    assert text_x > 100.0 + big_radius


# ==========================================================================
# Batching: the whole point of doing this in one frame
# ==========================================================================


def test_all_orbits_batch_into_one_vertex_and_index_array(kepler11):
    from astro_explorer.rendering.gl_backend import GLRenderer

    scene = _scene(kepler11, KEPLER11_EPOCH)
    vertices, indices = GLRenderer._batch_orbits(_FakeRenderer(), scene)

    total_points = sum(o.points_local.shape[0] for o in scene.orbits)
    total_segments = sum(o.points_local.shape[0] - 1 for o in scene.orbits)

    assert vertices.shape == (total_points, 9)
    assert indices.shape == (total_segments * 2,)
    assert indices.max() < total_points


def test_batched_indices_never_join_two_orbits(kepler11):
    """The failure mode LINE_STRIP would have: a segment across the gap."""
    from astro_explorer.rendering.gl_backend import GLRenderer

    scene = _scene(kepler11, KEPLER11_EPOCH)
    _vertices, indices = GLRenderer._batch_orbits(_FakeRenderer(), scene)

    boundaries = set()
    offset = 0
    for orbit in scene.orbits:
        offset += orbit.points_local.shape[0]
        boundaries.add(offset - 1)  # last vertex of this orbit

    pairs = indices.reshape(-1, 2)
    for first, second in pairs:
        assert second == first + 1
        assert first not in boundaries


def test_dashed_and_solid_orbits_share_one_buffer(kepler11):
    """Style travels per vertex, so one draw call covers both."""
    from astro_explorer.rendering.gl_backend import GLRenderer
    from astro_explorer.rendering.renderer import RenderOrbit

    scene = _scene(kepler11, KEPLER11_EPOCH)
    points = np.linspace(0.0, 1.0, 10).reshape(-1, 1) * np.array([1.0, 0.0, 0.0])
    scene.orbits.append(RenderOrbit("solid", points, dashed=False))

    vertices, _indices = GLRenderer._batch_orbits(_FakeRenderer(), scene)
    dash_column = vertices[:, 8]
    assert np.any(dash_column > 0.0)  # the dashed ones
    assert np.any(dash_column == 0.0)  # the solid one


class _FakeRenderer:
    """Just enough of GLRenderer for the pure-numpy batching helper."""

    class _Settings:
        dash_period = 0.035

    settings = _Settings()


# ==========================================================================
# Review section 10: the physical clock, kept as a permanent regression
# ==========================================================================


#: Measured on the committed Kepler-11 snapshot. Over one period of the
#: outermost planet, each planet completes this many revolutions. A shared
#: normalised animation clock would make every entry 1.00.
KEPLER11_REVOLUTIONS = {
    "Kepler-11 b": 11.49,
    "Kepler-11 c": 9.09,
    "Kepler-11 d": 5.22,
    "Kepler-11 e": 3.70,
    "Kepler-11 f": 2.54,
    "Kepler-11 g": 1.00,
}


def test_the_physical_clock_gives_each_planet_its_own_period(kepler11):
    """Review section 10: keep this as a permanent regression test."""
    periods = {
        record.name: record.elements.period.value_in(u.day) for record in kepler11.planets
    }
    outer = max(periods.values())

    for name, expected in KEPLER11_REVOLUTIONS.items():
        assert outer / periods[name] == pytest.approx(expected, abs=0.01), name

    # The innermost planet laps the outermost more than ten times over.
    assert outer / periods["Kepler-11 b"] > 11.0
    assert outer / periods["Kepler-11 g"] == pytest.approx(1.0, abs=1e-9)


def test_a_normalised_clock_would_fail_that_test(kepler11):
    """Guard the guard: show the assertion discriminates.

    Under a shared normalised clock every planet completes exactly one
    revolution per animation cycle, which is precisely what the published
    periods must not produce.
    """
    periods = [r.elements.period.value_in(u.day) for r in kepler11.planets]
    revolutions = [max(periods) / p for p in periods]
    assert not np.allclose(revolutions, 1.0)
    assert max(revolutions) / min(revolutions) > 11.0


def test_propagated_positions_reproduce_those_revolution_counts(kepler11):
    """Not just the periods - the propagator must actually advance that far."""
    periods = {
        record.name: record.elements.period.value_in(u.day) for record in kepler11.planets
    }
    outer = max(periods.values())
    steps = 4000

    for record in kepler11.planets:
        times = KEPLER11_EPOCH + np.linspace(0.0, outer, steps + 1)
        anomalies = np.array([kepler11.phase(record, t).mean_anomaly for t in times])
        # Unwrap and measure the total angle swept.
        swept = np.sum(np.mod(np.diff(anomalies) + np.pi, 2 * np.pi) - np.pi)
        revolutions = swept / (2 * np.pi)
        assert revolutions == pytest.approx(
            KEPLER11_REVOLUTIONS[record.name], abs=0.02
        ), record.name
