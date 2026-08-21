"""Kepler's second law and specific orbital energy.

Reference: ``ORBITAL_MECHANICS_FORMULAS_3D_EXOPLANET.md`` sections 13-17, 22.

These are the tests that catch the class of animation error a shape test
cannot see. An orbit can trace a perfect ellipse and still move along it
wrongly; equal-area-in-equal-time is what pins the *timing*, and the energy
identity pins position and velocity together.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_explorer.physics.kepler import solve_kepler
from astro_explorer.physics.orientation import specific_angular_momentum
from astro_explorer.physics.state_vectors import (
    MU_SUN_AU3_PER_DAY2,
    areal_velocity,
    expected_specific_energy,
    gravitational_parameter,
    specific_orbital_energy,
    state_at_mean_anomaly,
    swept_area,
    swept_area_binned,
    vis_viva_speed,
)

DEG = np.pi / 180.0

#: (label, a_AU, e, i, omega, Omega, M_star)
ORBITS = [
    ("circular", 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("earth-like", 1.0, 0.0167, 0.0, 0.0, 0.0, 1.0),
    ("mercury-like", 0.387, 0.2056, 7.0 * DEG, 29.1 * DEG, 48.3 * DEG, 1.0),
    ("eccentric", 1.0, 0.8, 45 * DEG, 60 * DEG, 120 * DEG, 1.0),
    ("hd-80606-b", 0.4603, 0.93183, 89.24 * DEG, -58.887 * DEG, 0.0, 1.05),
    ("extreme", 2.0, 0.99, 30 * DEG, 200 * DEG, 300 * DEG, 0.8),
]


def _elements(case):
    _label, a, e, inclination, omega, node, mass = case
    mu = gravitational_parameter(mass)
    return a, e, dict(
        inclination=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=node,
    ), mu


# ==========================================================================
# Kepler's second law: dA/dt = constant
# ==========================================================================


#: Substeps used to turn the polygonal chord estimate into an accurate
#: sector area. The chord error falls as substeps^-2; 200 keeps even the
#: e = 0.99 case two orders of magnitude inside the 1e-3 tolerance.
SUBSTEPS = 200
INTERVALS = 100


def _positions_over_one_orbit(a, e, angles, mu, intervals=INTERVALS, substeps=SUBSTEPS):
    """Equal time steps around a full revolution.

    Mean anomaly advances linearly with time, so equal steps in ``M`` *are*
    equal steps in ``t``. Sampling that way and then solving Kepler's
    equation is exactly what the animation does, which makes this a
    regression test for the animation and not only for the geometry.
    """
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, intervals * substeps + 1)
    return state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles).position


@pytest.mark.parametrize("case", ORBITS, ids=[c[0] for c in ORBITS])
def test_equal_times_sweep_equal_areas(case):
    """100 equal time intervals around a full orbit sweep equal areas."""
    a, e, angles, mu = _elements(case)
    positions = _positions_over_one_orbit(a, e, angles, mu)

    areas = swept_area_binned(positions, INTERVALS)
    assert areas.size == INTERVALS

    relative_variation = float(np.std(areas) / np.mean(areas))
    assert relative_variation < 1e-3, "dA/dt varies by {0:.3e}".format(relative_variation)


@pytest.mark.parametrize("case", ORBITS, ids=[c[0] for c in ORBITS])
def test_swept_area_matches_the_analytic_areal_velocity(case):
    """dA/dt must equal h/2, not merely be constant.

    A propagator running at the wrong *rate* would still produce equal
    areas; comparing against ``h/2`` catches that too.
    """
    a, e, angles, mu = _elements(case)
    period_days = 2.0 * np.pi * np.sqrt(a**3 / mu)

    positions = _positions_over_one_orbit(a, e, angles, mu)
    interval_seconds = period_days / INTERVALS
    measured = swept_area_binned(positions, INTERVALS) / interval_seconds

    assert np.allclose(measured, areal_velocity(a, e, mu), rtol=1e-3)


@pytest.mark.parametrize("eccentricity", [0.8, 0.93183, 0.99])
def test_the_residual_is_discretisation_not_physics(eccentricity):
    """The chord estimate must converge at second order.

    If the remaining spread came from a propagation error it would not
    shrink with step size. Quadrupling the substeps must cut the variation
    by roughly four, which is what distinguishes "the polygon is coarse"
    from "the orbit is wrong".
    """
    a, mu = 1.0, gravitational_parameter(1.0)
    angles = dict(inclination=0.4, argument_of_periapsis=1.0, longitude_of_ascending_node=2.0)

    variations = []
    for substeps in (25, 100, 400):
        positions = _positions_over_one_orbit(a, eccentricity, angles, mu, substeps=substeps)
        areas = swept_area_binned(positions, INTERVALS)
        variations.append(float(np.std(areas) / np.mean(areas)))

    for coarse, fine in zip(variations, variations[1:]):
        ratio = coarse / fine
        assert 12.0 < ratio < 20.0, "expected ~16x (second order), got {0:.1f}".format(ratio)


def test_a_coarse_polygon_under_reads_the_periapsis_sector():
    """Document the bias the substepping exists to remove.

    This is not a defect in the propagator; it is why the tests above
    subdivide, and it is worth pinning so nobody "simplifies" it away.
    """
    a, e, mu = 1.0, 0.99, gravitational_parameter(1.0)
    coarse = swept_area(state_at_mean_anomaly(
        a, e, np.linspace(0.0, 2.0 * np.pi, INTERVALS + 1), mu=mu
    ).position)
    exact_sector = np.pi * a * a * np.sqrt(1.0 - e**2) / INTERVALS

    # The first interval straddles periapsis, where the chord cuts the most.
    assert coarse[0] / exact_sector < 0.7
    # Far from periapsis the same polygon is essentially exact.
    assert np.isclose(coarse[INTERVALS // 2] / exact_sector, 1.0, rtol=1e-3)


@pytest.mark.parametrize("case", ORBITS, ids=[c[0] for c in ORBITS])
def test_total_swept_area_is_the_ellipse_area(case):
    """Summing the sectors over one revolution must give pi a b."""
    a, e, angles, mu = _elements(case)
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 20001)
    state = state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles)

    total = float(np.sum(swept_area(state.position)))
    expected = np.pi * a * a * np.sqrt(1.0 - e**2)
    assert np.isclose(total, expected, rtol=1e-4)


def test_uniform_angular_stepping_would_fail_this_test():
    """Guard the guard: show the test detects the classic animation bug.

    Advancing *true* anomaly uniformly - a common shortcut - traces the
    correct ellipse but at the wrong speed, and the areas then vary wildly.
    """
    a, e = 1.0, 0.8
    true_anomaly = np.linspace(0.0, 2.0 * np.pi, 101)
    radius = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    positions = np.stack(
        [radius * np.cos(true_anomaly), radius * np.sin(true_anomaly), np.zeros_like(radius)],
        axis=-1,
    )
    areas = swept_area(positions)
    assert float(np.std(areas) / np.mean(areas)) > 0.5


def test_angular_momentum_vector_is_conserved():
    """h = r x v must be constant in magnitude *and* direction."""
    a, e, angles, mu = _elements(ORBITS[4])  # HD 80606 b
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 997)
    state = state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles)

    momentum = state.angular_momentum()
    magnitude = np.linalg.norm(momentum, axis=-1)
    assert np.allclose(magnitude, specific_angular_momentum(a, e, mu), rtol=1e-10)

    direction = momentum / magnitude[:, None]
    assert np.allclose(direction, direction[0], atol=1e-10)


# ==========================================================================
# Specific orbital energy: epsilon = v^2/2 - mu/r = -mu/(2a)
# ==========================================================================


@pytest.mark.parametrize("case", ORBITS, ids=[c[0] for c in ORBITS])
def test_specific_energy_is_constant_and_correct(case):
    """The headline energy test, at many phases around the orbit."""
    a, e, angles, mu = _elements(case)

    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 512)
    state = state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles)

    energy = specific_orbital_energy(state, mu)
    expected = expected_specific_energy(a, mu)

    assert np.allclose(energy, expected, rtol=1e-10), (
        "energy drifts by up to {0:.3e} relative".format(
            float(np.max(np.abs(energy / expected - 1.0)))
        )
    )


@pytest.mark.parametrize("case", ORBITS, ids=[c[0] for c in ORBITS])
def test_speed_matches_the_vis_viva_equation(case):
    """v = sqrt(mu (2/r - 1/a)) at every phase (reference section 16)."""
    a, e, angles, mu = _elements(case)
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 512)
    state = state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles)

    assert np.allclose(state.speed, vis_viva_speed(state.radius, a, mu), rtol=1e-12)


def test_energy_is_independent_of_orbital_orientation():
    """Rotating the orbit cannot change its energy."""
    a, e, mu = 0.4603, 0.93183, gravitational_parameter(1.05)
    reference = None
    for inclination, omega, node in [
        (0.0, 0.0, 0.0),
        (89.24 * DEG, -58.887 * DEG, 0.0),
        (145 * DEG, 200 * DEG, 300 * DEG),
    ]:
        state = state_at_mean_anomaly(
            a, e, np.linspace(0, 2 * np.pi, 101), mu=mu,
            inclination=inclination,
            argument_of_periapsis=omega,
            longitude_of_ascending_node=node,
        )
        energy = specific_orbital_energy(state, mu)
        assert np.allclose(energy, expected_specific_energy(a, mu), rtol=1e-10)
        if reference is None:
            reference = energy
        else:
            assert np.allclose(energy, reference, rtol=1e-12)


def test_periapsis_is_fastest_and_apoapsis_slowest():
    """The physical content of the second law, stated as speeds."""
    a, e, mu = 1.0, 0.8, gravitational_parameter(1.0)
    at_periapsis = state_at_mean_anomaly(a, e, 0.0, mu=mu)
    at_apoapsis = state_at_mean_anomaly(a, e, np.pi, mu=mu)

    assert at_periapsis.speed > at_apoapsis.speed
    # v_peri / v_apo = (1+e)/(1-e) for any ellipse.
    assert np.isclose(
        float(at_periapsis.speed / at_apoapsis.speed), (1 + e) / (1 - e), rtol=1e-10
    )


def test_earth_orbital_speed_is_about_thirty_kilometres_per_second():
    """An external sanity anchor for the units."""
    import astropy.units as u

    state = state_at_mean_anomaly(1.0, 0.0167, 0.0, mu=MU_SUN_AU3_PER_DAY2)
    speed = (float(state.speed) * u.au / u.day).to_value(u.km / u.s)
    assert 29.0 < speed < 31.0


def test_hd_80606b_speed_ratio_is_extreme():
    """e = 0.93 means periapsis is roughly 28 times faster than apoapsis."""
    a, e, angles, mu = _elements(ORBITS[4])
    fast = state_at_mean_anomaly(a, e, 0.0, mu=mu, **angles).speed
    slow = state_at_mean_anomaly(a, e, np.pi, mu=mu, **angles).speed
    assert np.isclose(float(fast / slow), (1 + e) / (1 - e), rtol=1e-10)
    assert float(fast / slow) > 25.0


def test_energy_requires_a_velocity():
    """No stellar mass means no mu, which means no energy - not a guess."""
    state = state_at_mean_anomaly(1.0, 0.1, 0.5, mu=None)
    assert not state.has_velocity
    with pytest.raises(ValueError, match="needs a velocity"):
        specific_orbital_energy(state, MU_SUN_AU3_PER_DAY2)


def test_gravitational_parameter_is_unknown_without_a_stellar_mass():
    assert gravitational_parameter(None) is None
    assert gravitational_parameter(0.0) is None
    assert gravitational_parameter(np.nan) is None


def test_gravitational_parameter_includes_the_planet_mass():
    star_only = gravitational_parameter(1.0)
    with_planet = gravitational_parameter(1.0, 0.001)
    assert with_planet > star_only
    assert np.isclose(with_planet / star_only, 1.001, rtol=1e-12)


def test_kepler_third_law_falls_out_of_mu():
    """P = 2 pi sqrt(a^3/mu) must reproduce Earth's year."""
    period = 2.0 * np.pi * np.sqrt(1.0**3 / MU_SUN_AU3_PER_DAY2)
    assert np.isclose(period, 365.25, rtol=1e-3)


def test_solver_and_propagator_agree_on_radius():
    """r from the propagated vector must equal a(1 - e cos E)."""
    a, e, angles, mu = _elements(ORBITS[4])
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 333)
    state = state_at_mean_anomaly(a, e, mean_anomaly, mu=mu, **angles)
    ecc_anomaly = solve_kepler(mean_anomaly, e)
    assert np.allclose(state.radius, a * (1 - e * np.cos(ecc_anomaly)), rtol=1e-12)
