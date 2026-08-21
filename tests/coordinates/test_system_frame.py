"""Hierarchical reference frames.

The requirement is that parsecs, AU and kilometres are never combined in one
OpenGL coordinate space. These tests check that the *type system* prevents
it, not merely that the current call sites happen to get it right.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.coordinates.system_frame import (
    FLOAT32_SAFE_MAGNITUDE,
    FrameKind,
    FrameMismatchError,
    FramedPosition,
    PlanetFrame,
    PrecisionError,
    SystemFrame,
    UniverseFrame,
)

AU_IN_PC = float((1.0 * u.au).to_value(u.pc))
AU_IN_KM = float((1.0 * u.au).to_value(u.km))


# -- frame identity ---------------------------------------------------------


def test_each_frame_declares_its_unit():
    assert UniverseFrame().unit == u.pc
    assert SystemFrame.for_host("X").unit == u.au
    assert PlanetFrame().unit == u.km


def test_frame_kinds_do_not_share_a_unit():
    units = {kind.unit for kind in FrameKind}
    assert len(units) == 3


def test_a_position_knows_its_frame_and_unit():
    frame = SystemFrame.for_host("HD 80606")
    position = frame.at([1.0, 0.0, 0.0])
    assert position.kind is FrameKind.SYSTEM
    assert position.unit == u.au
    assert position.quantity().unit == u.au


# -- the anti-mixing rule ---------------------------------------------------


def test_adding_positions_from_different_frames_raises():
    system = SystemFrame.for_host("X")
    universe = UniverseFrame()
    with pytest.raises(FrameMismatchError):
        system.at([1, 0, 0]) + universe.at([1, 0, 0])


def test_subtracting_across_frames_raises():
    with pytest.raises(FrameMismatchError):
        SystemFrame.for_host("X").at([1, 0, 0]) - PlanetFrame().at([1, 0, 0])


def test_distance_across_frames_raises():
    """The exact shape of the 0.005 bug: a pc and an AU number subtracted."""
    system = SystemFrame.for_host("X")
    universe = UniverseFrame()
    with pytest.raises(FrameMismatchError):
        system.at([0.05, 0, 0]).distance_to(universe.at([66.47, 0, 0]))


def test_the_error_names_both_units():
    system = SystemFrame.for_host("HD 80606")
    universe = UniverseFrame()
    with pytest.raises(FrameMismatchError, match="AU"):
        system.at([1, 0, 0]) + universe.at([1, 0, 0])


def test_rendering_a_foreign_position_raises():
    system = SystemFrame.for_host("X")
    other = SystemFrame.for_host("Y", [10.0, 0.0, 0.0])
    with pytest.raises(FrameMismatchError):
        system.to_render(other.at([1, 0, 0]))


def test_same_frame_arithmetic_is_allowed():
    frame = SystemFrame.for_host("X")
    total = frame.at([1, 0, 0]) + frame.at([0, 2, 0])
    assert np.allclose(total.values, [1, 2, 0])
    assert total.frame is frame


# -- conversions ------------------------------------------------------------


def test_conversion_goes_through_absolute_parsecs():
    universe = UniverseFrame()
    system = SystemFrame.for_host("X", [10.0, 0.0, 0.0])

    one_au = system.at([1.0, 0.0, 0.0])
    in_pc = universe.convert(one_au)
    assert np.isclose(in_pc.values[0], 10.0 + AU_IN_PC, rtol=1e-12)
    assert in_pc.kind is FrameKind.UNIVERSE


def test_conversion_round_trips():
    universe = UniverseFrame()
    system = SystemFrame.for_host("X", [66.4711, -3.0, 12.0])
    original = system.at([0.4603, -0.12, 0.31])
    back = system.convert(universe.convert(original))
    assert np.allclose(back.values, original.values, rtol=1e-9, atol=1e-9)


def test_system_to_planet_frame_uses_the_exact_au_to_km_factor():
    system = SystemFrame.for_host("X")
    planet = PlanetFrame()
    converted = planet.convert(system.at([1.0, 0.0, 0.0]))
    assert np.isclose(converted.values[0], AU_IN_KM, rtol=1e-9)


def test_the_wrong_historical_factor_is_nowhere_near_the_real_one():
    """0.005 pc per AU was wrong by three orders of magnitude."""
    assert abs(AU_IN_PC / 0.005) < 1e-2


# -- the SystemFrame guarantees ---------------------------------------------


def test_the_star_is_exactly_at_the_origin():
    frame = SystemFrame.for_host("HD 80606", [66.47, 0.0, 0.0])
    star = frame.star_position()
    assert np.array_equal(star.values, np.zeros(3))
    assert np.array_equal(star.to_render(), np.zeros(3, dtype=np.float32))


def test_placing_a_planet_applies_no_scale_factor_at_all():
    """The orbital vector must reach the GPU unchanged."""
    frame = SystemFrame.for_host("HD 80606", [66.47, -12.0, 3.0])
    orbital_vector = np.array([0.01621421, -0.00035634, -0.02686246])
    placed = frame.place_planet(orbital_vector)
    assert np.array_equal(placed.values, orbital_vector)
    assert np.allclose(placed.to_render(), orbital_vector, rtol=1e-6)


def test_the_frame_works_without_knowing_where_the_system_is():
    """A host with an unusable parallax must still be renderable."""
    frame = SystemFrame.for_host("Unknown distance")
    assert np.array_equal(frame.origin_pc, np.zeros(3))
    assert np.allclose(frame.place_planet([0.5, 0, 0]).values, [0.5, 0, 0])


def test_planet_distance_from_the_star_is_the_orbital_radius():
    frame = SystemFrame.for_host("HD 80606")
    planet = frame.place_planet([0.01621421, -0.00035634, -0.02686246])
    assert np.isclose(planet.distance_to(frame.star_position()), 0.03137865, rtol=1e-6)


def test_placing_an_array_of_positions():
    frame = SystemFrame.for_host("X")
    path = np.random.default_rng(0).normal(size=(64, 3))
    placed = frame.place_planet(path)
    assert placed.values.shape == (64, 3)
    assert placed.to_render().shape == (64, 3)


# -- the GPU boundary -------------------------------------------------------


def test_render_output_is_float32():
    frame = SystemFrame.for_host("X")
    rendered = frame.place_planet([0.46, 0.0, 0.0]).to_render()
    assert rendered.dtype == np.float32


def test_cpu_storage_stays_float64():
    frame = SystemFrame.for_host("X")
    assert frame.place_planet([0.46, 0.0, 0.0]).values.dtype == np.float64
    assert frame.origin_pc.dtype == np.float64


def test_narrowing_refuses_to_lose_precision():
    """One parsec expressed in km cannot survive float32."""
    planet_frame = PlanetFrame()
    huge = planet_frame.at([1.0 / AU_IN_PC * AU_IN_KM, 0.0, 0.0])
    with pytest.raises(PrecisionError, match="float32"):
        huge.to_render()


def test_a_coordinate_just_inside_the_limit_is_allowed():
    frame = PlanetFrame()
    ok = frame.at([FLOAT32_SAFE_MAGNITUDE * 0.5, 0.0, 0.0])
    assert ok.to_render().dtype == np.float32


def test_non_finite_coordinates_are_refused():
    frame = SystemFrame.for_host("X")
    with pytest.raises((PrecisionError, ValueError)):
        frame.at([np.nan, 0.0, 0.0]).to_render()


# -- frame transitions ------------------------------------------------------


def test_entering_a_system_puts_the_star_at_the_origin():
    universe = UniverseFrame()
    host = universe.at([66.4711, 0.0, 0.0])
    system = universe.enter_system(host, "HD 80606")

    assert system.kind is FrameKind.SYSTEM
    assert np.allclose(system.star_position().values, 0.0)
    # ...and the star is still in the right place absolutely.
    assert np.allclose(system.to_absolute_pc(system.star_position()), [66.4711, 0.0, 0.0])


def test_entering_a_planet_puts_the_planet_at_the_origin():
    universe = UniverseFrame()
    system = universe.enter_system(universe.at([10.0, 0.0, 0.0]), "X")
    planet_position = system.place_planet([0.46, 0.0, 0.0])
    planet_frame = system.enter_planet(planet_position)

    assert planet_frame.kind is FrameKind.PLANET
    back = planet_frame.convert(planet_position)
    assert np.allclose(back.values, 0.0, atol=1e-6)  # within a metre


def test_rebasing_the_universe_frame_moves_the_floating_origin():
    universe = UniverseFrame()
    target = universe.at([500.0, 0.0, 0.0])
    rebased = universe.rebase_to(target)
    assert np.allclose(rebased.origin_pc, [500.0, 0.0, 0.0])
    assert np.allclose(rebased.convert(target).values, 0.0)


def test_a_system_far_away_still_renders_at_the_origin():
    """Distance from Earth must not degrade local precision."""
    universe = UniverseFrame()
    system = universe.enter_system(universe.at([8000.0, -2000.0, 300.0]), "far")
    rendered = system.place_planet([0.05, 0.0, 0.0]).to_render()
    assert np.allclose(rendered, [0.05, 0.0, 0.0], rtol=1e-6)
