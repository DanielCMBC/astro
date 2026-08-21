"""Coordinate tests (roadmap section 22, "Coordinates")."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.coordinates.floating_origin import (
    AU_TO_KM,
    AU_TO_PC,
    KM_TO_AU,
    PC_TO_AU,
    FloatingOrigin,
    Scale,
    SceneGraph,
)
from astro_explorer.coordinates.frames import Frame, cartesian_pc, separation_pc, sky_position
from astro_explorer.coordinates.transforms import (
    au_to_pc,
    describe_distance,
    ly_to_pc,
    pc_to_au,
    pc_to_ly,
)
from astro_explorer.provenance import measured

# Proxima Centauri: RA 217.4289 deg, Dec -62.6795 deg, parallax 768.07 mas.
PROXIMA = (217.4289, -62.6795, 768.0665)


# -- unit bridges ------------------------------------------------------------


def test_au_to_pc_is_correct_and_not_the_prototype_constant():
    assert np.isclose(AU_TO_PC, 4.8481368e-6, rtol=1e-6)
    assert AU_TO_PC / 0.005 < 1e-2  # three orders of magnitude apart


def test_unit_round_trips():
    assert np.isclose(AU_TO_PC * PC_TO_AU, 1.0)
    assert np.isclose(AU_TO_KM * KM_TO_AU, 1.0)
    assert np.isclose(pc_to_ly(ly_to_pc(4.2)), 4.2)
    assert np.isclose(au_to_pc(pc_to_au(3.0)), 3.0)


def test_parsec_to_lightyear():
    assert np.isclose(float(pc_to_ly(1.0)), 3.26156, rtol=1e-5)


def test_au_to_km():
    assert np.isclose(AU_TO_KM, 1.495978707e8, rtol=1e-9)


# -- ICRS geometry -----------------------------------------------------------


def test_known_object_cartesian_round_trip():
    ra, dec, parallax = PROXIMA
    position = sky_position("Proxima", ra, dec, parallax_mas=parallax)
    distance = position.distance.value_in(u.pc)
    assert np.isclose(distance, 1.3020, rtol=1e-3)

    cartesian = position.cartesian_pc()
    assert np.isclose(np.linalg.norm(cartesian), distance, rtol=1e-9)


def test_cartesian_recovers_ra_and_dec():
    ra, dec, parallax = PROXIMA
    cartesian = cartesian_pc(ra, dec, 1.0 / (parallax / 1000.0))[0]
    recovered_ra = np.degrees(np.arctan2(cartesian[1], cartesian[0])) % 360.0
    recovered_dec = np.degrees(np.arcsin(cartesian[2] / np.linalg.norm(cartesian)))
    assert np.isclose(recovered_ra, ra, atol=1e-6)
    assert np.isclose(recovered_dec, dec, atol=1e-6)


def test_galactic_transform_preserves_distance():
    ra, dec, parallax = PROXIMA
    icrs = cartesian_pc(ra, dec, 1.302)[0]
    galactic = cartesian_pc(ra, dec, 1.302, Frame.GALACTIC)[0]
    assert np.isclose(np.linalg.norm(icrs), np.linalg.norm(galactic), rtol=1e-9)


def test_rows_without_a_distance_become_nan_not_a_guess():
    result = cartesian_pc([10.0, 20.0], [0.0, 0.0], [5.0, np.nan])
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isnan(result[1]))


def test_separation_between_two_objects():
    a = sky_position("A", 0.0, 0.0, catalog_distance_pc=10.0)
    b = sky_position("B", 0.0, 0.0, catalog_distance_pc=15.0)
    assert np.isclose(separation_pc(a, b).value, 5.0, rtol=1e-9)


def test_separation_is_unknown_when_a_distance_is_missing():
    a = sky_position("A", 0.0, 0.0, catalog_distance_pc=10.0)
    b = sky_position("B", 30.0, 10.0, parallax_mas=-1.0)
    result = separation_pc(a, b)
    assert not result.is_known
    assert "distance" in result.note


def test_distance_description_lists_several_units():
    position = sky_position("X", 0.0, 0.0, catalog_distance_pc=10.0)
    text = "\n".join(describe_distance(position.distance))
    assert "pc" in text and "lyr" in text and "km" in text
    assert "Light travel time" in text


def test_description_does_not_invent_precision():
    """Roadmap 10.1: avoid fake precision beyond the catalogue uncertainty."""
    distance = measured(12.3456789, u.pc, error_plus=0.5, error_minus=0.5)
    line = describe_distance(distance)[0]
    assert "12.35" in line or "12.3" in line
    assert "12.3456789" not in line


# -- floating origin ---------------------------------------------------------


def test_world_to_local_round_trip():
    origin = FloatingOrigin(origin_pc=np.array([10.0, -4.0, 3.0]), scale=Scale.GALAXY)
    world = np.array([12.0, -1.0, 3.5])
    local = origin.world_to_local(world)
    assert np.allclose(origin.local_to_world(local), world)


def test_system_frame_expresses_offsets_in_au():
    origin = FloatingOrigin(origin_pc=np.zeros(3), scale=Scale.SYSTEM)
    one_au_away = np.array([AU_TO_PC, 0.0, 0.0])
    assert np.allclose(origin.world_to_local(one_au_away), [1.0, 0.0, 0.0], atol=1e-9)


def test_planet_frame_expresses_offsets_in_km():
    origin = FloatingOrigin(origin_pc=np.zeros(3), scale=Scale.PLANET)
    one_au_away = np.array([AU_TO_PC, 0.0, 0.0])
    assert np.allclose(origin.world_to_local(one_au_away), [AU_TO_KM, 0.0, 0.0], rtol=1e-9)


def test_rebasing_moves_the_origin_only_when_needed():
    origin = FloatingOrigin(scale=Scale.GALAXY, rebase_threshold=100.0)
    assert not origin.maybe_rebase(np.array([1.0, 0.0, 0.0]))
    assert origin.maybe_rebase(np.array([500.0, 0.0, 0.0]))
    assert np.isclose(origin.origin_pc[0], 500.0)


def test_render_space_is_float32_and_bounded():
    origin = FloatingOrigin(origin_pc=np.zeros(3), scale=Scale.SYSTEM)
    rendered = origin.to_render_space(np.array([AU_TO_PC, 0.0, 0.0]))
    assert rendered.dtype == np.float32


def test_render_space_refuses_to_silently_lose_precision():
    """A coordinate too large for float32 must raise, not degrade."""
    origin = FloatingOrigin(origin_pc=np.zeros(3), scale=Scale.PLANET)
    with pytest.raises(ValueError, match="float32"):
        origin.to_render_space(np.array([1.0, 0.0, 0.0]))  # one parsec in km


# -- the scene graph ---------------------------------------------------------


def test_planet_offset_uses_au_directly_in_the_system_frame():
    """The 0.005 bug made systems roughly a thousand times too large."""
    graph = SceneGraph()
    host = np.array([1.2, -3.4, 0.7])
    graph.enter_system(host)
    position = graph.planet_render_position(host, [0.05, 0.0, 0.0])
    assert np.allclose(position, [0.05, 0.0, 0.0], atol=1e-6)
    # The wrong scaling would have produced 0.05 * 0.005 / AU_TO_PC.
    assert not np.isclose(position[0], 0.05 * 0.005 / AU_TO_PC)


def test_planet_offset_is_negligible_but_exact_in_the_galaxy_frame():
    graph = SceneGraph()
    host = np.array([100.0, 0.0, 0.0])
    graph.enter_galaxy()
    star = graph.host_render_position(host)
    planet = graph.planet_render_position(host, [1.0, 0.0, 0.0])
    separation = float(np.linalg.norm(np.asarray(planet, dtype=np.float64) - star))
    assert separation < 1e-4  # one AU in parsecs is tiny
    assert separation < 10.0 * AU_TO_PC


def test_entering_a_planet_frame_puts_the_planet_at_the_origin():
    graph = SceneGraph()
    host = np.array([2.0, 0.0, 0.0])
    offset = np.array([0.4, 0.0, 0.0])
    graph.enter_planet(host, offset)
    planet = graph.planet_render_position(host, offset)
    assert np.allclose(planet, [0.0, 0.0, 0.0], atol=1.0)  # within a km
    assert graph.origin.scale is Scale.PLANET
