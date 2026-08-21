"""Orbital mechanics tests (roadmap section 22, "Orbital mechanics")."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.physics.kepler import (
    eccentric_from_true_anomaly,
    kepler_residual,
    mean_anomaly_from_eccentric,
    solve_kepler,
    true_anomaly_from_eccentric,
)
from astro_explorer.physics.orbital_elements import (
    OrbitalElements,
    orbital_radius,
    perifocal_position,
    position_at_eccentric_anomaly,
    rotation_perifocal_to_reference,
)
from astro_explorer.provenance import Status, measured

TOLERANCE = 1e-10

# circular, Earth, Mercury-like, high, very high, near-parabolic
ECCENTRICITIES = [0.0, 0.0167, 0.2056, 0.6, 0.9, 0.97, 0.999]


@pytest.mark.parametrize("eccentricity", ECCENTRICITIES)
def test_kepler_equation_is_actually_solved(eccentricity):
    """E - e sin E must equal M everywhere, not approximately."""
    mean = np.linspace(-np.pi, np.pi, 2001)
    ecc_anomaly = solve_kepler(mean, eccentricity)
    assert np.max(np.abs(kepler_residual(ecc_anomaly, eccentricity, mean))) < TOLERANCE


@pytest.mark.parametrize("eccentricity", ECCENTRICITIES)
def test_round_trip_through_true_anomaly(eccentricity):
    ecc_anomaly = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 401)
    nu = true_anomaly_from_eccentric(ecc_anomaly, eccentricity)
    recovered = eccentric_from_true_anomaly(nu, eccentricity)
    assert np.allclose(recovered, ecc_anomaly, atol=1e-9)


def test_first_order_approximation_is_not_good_enough():
    """The 3D prototype used E ~ M + e sin M; show why that had to go."""
    eccentricity = 0.9
    mean = np.linspace(-np.pi, np.pi, 501)
    approximate = mean + eccentricity * np.sin(mean)
    exact = solve_kepler(mean, eccentricity)
    # The approximation is wrong by more than a third of a radian.
    assert np.max(np.abs(approximate - exact)) > 0.3


def test_circular_orbit_is_exactly_uniform():
    mean = np.linspace(0.0, 2.0 * np.pi, 101)
    assert np.allclose(solve_kepler(mean, 0.0), np.mod(mean + np.pi, 2 * np.pi) - np.pi)


def _angular_difference(a, b):
    """Signed difference wrapped into (-pi, pi], so +pi and -pi agree."""
    return np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2.0 * np.pi) - np.pi


def test_forward_and_inverse_kepler_agree():
    ecc_anomaly = np.linspace(-np.pi, np.pi, 257)
    for eccentricity in ECCENTRICITIES:
        mean = mean_anomaly_from_eccentric(ecc_anomaly, eccentricity)
        recovered = solve_kepler(mean, eccentricity)
        assert np.max(np.abs(_angular_difference(recovered, ecc_anomaly))) < 1e-9


def test_parabolic_and_hyperbolic_are_rejected_not_clamped():
    """The original code clipped e to 0.98; silently changing the orbit."""
    with pytest.raises(ValueError):
        solve_kepler(0.5, 1.0)
    with pytest.raises(ValueError):
        solve_kepler(0.5, 1.4)
    with pytest.raises(ValueError):
        solve_kepler(0.5, -0.1)


def test_scalar_input_returns_scalar():
    assert isinstance(solve_kepler(0.5, 0.3), float)


def test_broadcasting_over_arrays():
    result = solve_kepler([0.1, 0.2, 0.3], [0.0, 0.5, 0.9])
    assert result.shape == (3,)


# -- geometry ----------------------------------------------------------------


def test_perifocal_matches_the_conic_equation():
    """x = a(cos E - e), y = a sqrt(1-e^2) sin E must satisfy r = a(1-e^2)/(1+e cos nu)."""
    a, e = 2.5, 0.42
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 361)
    position = perifocal_position(a, e, ecc_anomaly)
    radius = np.linalg.norm(position, axis=-1)

    nu = true_anomaly_from_eccentric(ecc_anomaly, e)
    assert np.allclose(radius, orbital_radius(a, e, nu), atol=1e-12)


def test_periapsis_and_apoapsis_distances():
    a, e = 1.5, 0.3
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 2001)
    radius = np.linalg.norm(perifocal_position(a, e, ecc_anomaly), axis=-1)
    assert np.isclose(radius.min(), a * (1 - e), atol=1e-9)
    assert np.isclose(radius.max(), a * (1 + e), atol=1e-9)


def test_kepler_second_law_equal_areas():
    """A planet must sweep equal areas in equal times."""
    a, e = 1.0, 0.6
    mean = np.linspace(0.0, 2.0 * np.pi, 2001)
    position = perifocal_position(a, e, solve_kepler(mean, e))[:, :2]

    # Triangle areas between consecutive radius vectors.
    cross = position[:-1, 0] * position[1:, 1] - position[:-1, 1] * position[1:, 0]
    areas = 0.5 * np.abs(cross)
    assert np.std(areas) / np.mean(areas) < 1e-3


def test_rotation_matrix_is_orthonormal():
    rotation = rotation_perifocal_to_reference(0.7, 1.1, 2.3)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_zero_angles_leave_the_orbit_in_the_xy_plane():
    rotation = rotation_perifocal_to_reference(0.0, 0.0, 0.0)
    assert np.allclose(rotation, np.eye(3))


def test_inclination_actually_lifts_the_orbit_out_of_the_plane():
    """Roadmap 4.3: orbits were [x, y, 0.0], so every system was coplanar."""
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.0),
        inclination=measured(np.radians(60.0), u.rad),
        argument_of_periastron=measured(0.0, u.rad),
        longitude_of_ascending_node=measured(0.0, u.rad),
    )
    position = position_at_eccentric_anomaly(elements, np.linspace(0, 2 * np.pi, 181))
    assert np.max(np.abs(position[:, 2])) > 0.5


def test_inclined_orbit_keeps_its_radius():
    """Rotation must not change distances."""
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.3),
        inclination=measured(np.radians(75.0), u.rad),
        argument_of_periastron=measured(np.radians(40.0), u.rad),
        longitude_of_ascending_node=measured(np.radians(110.0), u.rad),
    )
    ecc_anomaly = np.linspace(0, 2 * np.pi, 361)
    rotated = np.linalg.norm(position_at_eccentric_anomaly(elements, ecc_anomaly), axis=-1)
    flat = np.linalg.norm(perifocal_position(1.0, 0.3, ecc_anomaly), axis=-1)
    assert np.allclose(rotated, flat, atol=1e-12)


def test_unknown_node_is_normalised_and_labelled_not_invented():
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.1),
        inclination=measured(np.radians(89.0), u.rad),
    )
    assert not elements.orientation_known

    display = elements.for_display()
    assert display.longitude_of_ascending_node.status is Status.ASSUMED_FOR_VISUALIZATION
    assert display.longitude_of_ascending_node.value == 0.0
    # The original element set is untouched.
    assert elements.longitude_of_ascending_node.status is Status.UNKNOWN

    lines = "\n".join(display.describe_orientation())
    assert "Ascending node: unknown" in lines
