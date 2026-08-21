"""Numerical tests of the 3D orbital orientation transform.

Reference: ``ORBITAL_MECHANICS_FORMULAS_3D_EXOPLANET.md`` sections 8-11, 22.

Nothing here is checked by eye. Every case asserts on vector components,
and the combined-rotation case is cross-checked against the expanded scalar
equations of reference section 11, written out independently below so that a
mistake in the matrix composition cannot hide behind the same mistake in the
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_explorer.physics.orientation import (
    node_vector,
    orbit_normal,
    orbital_state,
    perifocal_position,
    perifocal_velocity,
    position_from_eccentric_anomaly,
    radius_from_eccentric_anomaly,
    rotation_perifocal_to_inertial,
    rotation_x,
    rotation_z,
)

ATOL = 1e-12
DEG = np.pi / 180.0


# --------------------------------------------------------------------------
# Independent reference implementation (reference section 11).
#
# Written from the expanded scalar equations rather than from matrices, so it
# shares no code with the implementation under test.
# --------------------------------------------------------------------------
def reference_position(a, e, ecc_anomaly, inclination, arg_periapsis, node):
    """Expanded form of R_z(Omega) R_x(i) R_z(omega) r_p."""
    x_p = a * (np.cos(ecc_anomaly) - e)
    y_p = a * np.sqrt(1.0 - e**2) * np.sin(ecc_anomaly)

    cos_O, sin_O = np.cos(node), np.sin(node)
    cos_w, sin_w = np.cos(arg_periapsis), np.sin(arg_periapsis)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)

    x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_p + (
        -cos_O * sin_w - sin_O * cos_w * cos_i
    ) * y_p
    y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_p + (
        -sin_O * sin_w + cos_O * cos_w * cos_i
    ) * y_p
    z = (sin_w * sin_i) * x_p + (cos_w * sin_i) * y_p
    return np.stack([x, y, z], axis=-1)


# -- rotation primitives ----------------------------------------------------


def test_rotation_z_matches_the_reference_matrix():
    theta = 0.7
    expected = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(rotation_z(theta), expected, atol=ATOL)


def test_rotation_x_matches_the_reference_matrix():
    angle = 0.4
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    assert np.allclose(rotation_x(angle), expected, atol=ATOL)


def test_rotation_z_by_90_degrees_maps_x_to_y():
    assert np.allclose(rotation_z(90 * DEG) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=ATOL)


def test_rotation_x_by_90_degrees_maps_y_to_z():
    assert np.allclose(rotation_x(90 * DEG) @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=ATOL)


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.3, 1.1, 2.4), (np.pi / 2, np.pi / 3, np.pi / 6), (2.9, -1.2, 0.8)],
)
def test_the_composed_rotation_is_a_proper_rotation(angles):
    rotation = rotation_perifocal_to_inertial(*angles)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=ATOL)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=ATOL)


def test_all_angles_zero_is_the_identity():
    assert np.allclose(rotation_perifocal_to_inertial(0.0, 0.0, 0.0), np.eye(3), atol=ATOL)


# -- case 1: i = 0, the orbit stays in the XY plane -------------------------


def test_zero_inclination_keeps_the_orbit_in_the_xy_plane():
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 361)
    position = position_from_eccentric_anomaly(
        1.5, 0.4, ecc_anomaly, inclination=0.0, argument_of_periapsis=0.9,
        longitude_of_ascending_node=2.1,
    )
    assert np.max(np.abs(position[:, 2])) < ATOL


def test_zero_inclination_normal_is_the_reference_pole():
    for node in (0.0, 1.0, np.pi):
        assert np.allclose(orbit_normal(0.0, node), [0.0, 0.0, 1.0], atol=ATOL)


def test_zero_inclination_with_zero_angles_reproduces_the_perifocal_frame():
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 91)
    perifocal = perifocal_position(2.0, 0.3, ecc_anomaly)
    rotated = position_from_eccentric_anomaly(2.0, 0.3, ecc_anomaly)
    assert np.allclose(rotated, perifocal, atol=ATOL)


# -- case 2: i = 90 degrees, the plane becomes perpendicular ----------------


def test_ninety_degree_inclination_puts_the_orbit_in_the_xz_plane():
    """With Omega = 0 and omega = 0, R_x(90) maps the orbital y axis onto z."""
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 361)
    position = position_from_eccentric_anomaly(
        1.0, 0.5, ecc_anomaly, inclination=90 * DEG
    )
    # The orbit is edge-on: it has no extent along y at all.
    assert np.max(np.abs(position[:, 1])) < ATOL
    # ...and it genuinely occupies z.
    assert np.max(np.abs(position[:, 2])) > 0.5


def test_ninety_degree_inclination_normal_lies_in_the_reference_plane():
    """An edge-on orbit's angular momentum is perpendicular to the pole."""
    normal = orbit_normal(90 * DEG, 0.0)
    assert np.isclose(np.dot(normal, [0.0, 0.0, 1.0]), 0.0, atol=ATOL)
    assert np.allclose(normal, [0.0, -1.0, 0.0], atol=ATOL)


def test_inclination_tilts_the_normal_by_exactly_that_angle():
    for inclination in (0.0, 15 * DEG, 45 * DEG, 89.24 * DEG, 90 * DEG):
        normal = orbit_normal(inclination, 1.234)
        cos_tilt = float(np.dot(normal, [0.0, 0.0, 1.0]))
        assert np.isclose(cos_tilt, np.cos(inclination), atol=ATOL)


def test_the_orbit_normal_does_not_depend_on_the_argument_of_periapsis():
    """Rotating periapsis inside the plane cannot tilt the plane."""
    base = orbit_normal(0.7, 1.9)
    for omega in (0.0, 1.0, 2.5, 6.0):
        rotation = rotation_perifocal_to_inertial(0.7, omega, 1.9)
        assert np.allclose(rotation @ [0.0, 0.0, 1.0], base, atol=ATOL)


# -- case 3: omega = 90 degrees, periapsis rotates in the plane -------------


def test_argument_of_periapsis_rotates_periapsis_within_the_orbital_plane():
    """At E = 0 the planet is at periapsis, on the perifocal +x axis."""
    a, e = 1.0, 0.6
    periapsis_distance = a * (1.0 - e)

    at_zero = position_from_eccentric_anomaly(a, e, 0.0)
    assert np.allclose(at_zero, [periapsis_distance, 0.0, 0.0], atol=ATOL)

    # omega = 90 degrees with i = 0 swings periapsis onto +y.
    at_ninety = position_from_eccentric_anomaly(
        a, e, 0.0, argument_of_periapsis=90 * DEG
    )
    assert np.allclose(at_ninety, [0.0, periapsis_distance, 0.0], atol=ATOL)


def test_argument_of_periapsis_of_180_degrees_flips_periapsis():
    a, e = 1.0, 0.6
    flipped = position_from_eccentric_anomaly(a, e, 0.0, argument_of_periapsis=np.pi)
    assert np.allclose(flipped, [-a * (1.0 - e), 0.0, 0.0], atol=ATOL)


def test_omega_measures_the_angle_from_the_node_to_periapsis():
    """The defining property of omega, checked as an angle."""
    for omega in (0.0, 30 * DEG, 90 * DEG, 200 * DEG):
        for inclination in (0.0, 40 * DEG, 89.24 * DEG):
            node = node_vector(1.1)
            periapsis = position_from_eccentric_anomaly(
                1.0, 0.5, 0.0,
                inclination=inclination,
                argument_of_periapsis=omega,
                longitude_of_ascending_node=1.1,
            )
            unit_periapsis = periapsis / np.linalg.norm(periapsis)
            angle = np.arccos(np.clip(np.dot(node, unit_periapsis), -1.0, 1.0))
            expected = omega if omega <= np.pi else 2 * np.pi - omega
            assert np.isclose(angle, expected, atol=1e-9)


def test_omega_with_ninety_degree_inclination_lifts_periapsis_to_the_pole():
    """i = 90 and omega = 90 place periapsis on the +z axis."""
    a, e = 1.0, 0.4
    position = position_from_eccentric_anomaly(
        a, e, 0.0, inclination=90 * DEG, argument_of_periapsis=90 * DEG
    )
    assert np.allclose(position, [0.0, 0.0, a * (1.0 - e)], atol=ATOL)


# -- case 4: Omega = 90 degrees, the node rotates about the reference pole ---


def test_node_rotation_moves_periapsis_onto_the_y_axis():
    a, e = 1.0, 0.5
    position = position_from_eccentric_anomaly(
        a, e, 0.0, longitude_of_ascending_node=90 * DEG
    )
    assert np.allclose(position, [0.0, a * (1.0 - e), 0.0], atol=ATOL)


def test_node_rotation_spins_the_normal_about_the_pole():
    """Omega rotates the angular momentum about z, leaving its tilt alone."""
    inclination = 50 * DEG
    base = orbit_normal(inclination, 0.0)
    rotated = orbit_normal(inclination, 90 * DEG)
    assert np.allclose(rotated, rotation_z(90 * DEG) @ base, atol=ATOL)
    assert np.isclose(base[2], rotated[2], atol=ATOL)


def test_the_ascending_node_lies_in_the_reference_plane_at_omega_capital():
    for node in (0.0, 30 * DEG, 90 * DEG, 217 * DEG):
        vector = node_vector(node)
        assert np.isclose(vector[2], 0.0, atol=ATOL)
        assert np.isclose(np.arctan2(vector[1], vector[0]) % (2 * np.pi), node % (2 * np.pi))


def test_the_orbit_actually_crosses_the_reference_plane_at_the_node():
    """The physical meaning of Omega: where the orbit rises through z = 0."""
    a, e = 1.0, 0.35
    inclination, omega, node = 60 * DEG, 25 * DEG, 110 * DEG

    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 200001)
    position, _ = orbital_state(
        a, e, ecc_anomaly,
        inclination=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=node,
    )
    z = position[:, 2]
    # Ascending crossing: z goes from negative to positive.
    crossings = np.where((z[:-1] < 0.0) & (z[1:] >= 0.0))[0]
    assert crossings.size >= 1

    index = crossings[0]
    direction = position[index, :2] / np.linalg.norm(position[index, :2])
    measured = np.arctan2(direction[1], direction[0]) % (2 * np.pi)
    assert np.isclose(measured, node, atol=1e-4)


# -- case 5: combined Omega / i / omega vs the reference calculation --------


COMBINED_CASES = [
    # (a, e, i, omega, Omega) - the last is HD 80606 b's measured geometry.
    (1.0, 0.0, 30 * DEG, 40 * DEG, 50 * DEG),
    (2.5, 0.3, 60 * DEG, 120 * DEG, 200 * DEG),
    (0.05, 0.9, 89.0 * DEG, 300 * DEG, 15 * DEG),
    (0.4603, 0.93183, 89.24 * DEG, -58.887 * DEG, 0.0),
    (1.7, 0.55, 145 * DEG, -30 * DEG, 275 * DEG),
]


@pytest.mark.parametrize("a,e,inclination,omega,node", COMBINED_CASES)
def test_combined_rotation_matches_the_expanded_reference_equations(
    a, e, inclination, omega, node
):
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 501)
    computed = position_from_eccentric_anomaly(
        a, e, ecc_anomaly,
        inclination=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=node,
    )
    expected = reference_position(a, e, ecc_anomaly, inclination, omega, node)
    assert np.allclose(computed, expected, atol=1e-13)


@pytest.mark.parametrize("a,e,inclination,omega,node", COMBINED_CASES)
def test_rotation_preserves_orbital_radius(a, e, inclination, omega, node):
    """A rotation cannot change a distance; if it does, it is not a rotation."""
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 501)
    rotated = position_from_eccentric_anomaly(
        a, e, ecc_anomaly,
        inclination=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=node,
    )
    radii = np.linalg.norm(rotated, axis=-1)
    assert np.allclose(radii, radius_from_eccentric_anomaly(a, e, ecc_anomaly), atol=1e-13)
    assert np.isclose(radii.min(), a * (1 - e), atol=1e-12)
    assert np.isclose(radii.max(), a * (1 + e), atol=1e-12)


@pytest.mark.parametrize("a,e,inclination,omega,node", COMBINED_CASES)
def test_the_orbit_stays_planar_after_the_full_rotation(a, e, inclination, omega, node):
    """Every point must lie in the plane whose normal is the orbit normal."""
    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, 501)
    position = position_from_eccentric_anomaly(
        a, e, ecc_anomaly,
        inclination=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=node,
    )
    normal = orbit_normal(inclination, node)
    # The focus is at the origin and lies in the orbital plane, so every
    # position vector must be perpendicular to the normal.
    assert np.max(np.abs(position @ normal)) < 1e-13


def test_rotation_order_matters():
    """R_z(Omega) R_x(i) R_z(omega) is not the same as any other order."""
    inclination, omega, node = 40 * DEG, 70 * DEG, 110 * DEG
    correct = rotation_perifocal_to_inertial(inclination, omega, node)
    swapped = rotation_z(omega) @ rotation_x(inclination) @ rotation_z(node)
    assert not np.allclose(correct, swapped, atol=1e-6)


# -- velocity ---------------------------------------------------------------


def test_perifocal_velocity_is_perpendicular_to_position_at_the_apsides():
    """At periapsis and apoapsis the motion is purely transverse."""
    a, e, mu = 1.0, 0.6, 4.0 * np.pi**2 / 365.25**2
    for ecc_anomaly in (0.0, np.pi):
        position = perifocal_position(a, e, ecc_anomaly)
        velocity = perifocal_velocity(a, e, ecc_anomaly, mu)
        assert np.isclose(np.dot(position, velocity), 0.0, atol=1e-15)


def test_velocity_is_rotated_by_the_same_matrix_as_position():
    a, e, mu = 1.0, 0.4, 2.959e-4
    angles = dict(
        inclination=35 * DEG, argument_of_periapsis=80 * DEG,
        longitude_of_ascending_node=150 * DEG,
    )
    position, velocity = orbital_state(a, e, 1.1, mu=mu, **angles)
    rotation = rotation_perifocal_to_inertial(
        angles["inclination"],
        angles["argument_of_periapsis"],
        angles["longitude_of_ascending_node"],
    )
    assert np.allclose(position, rotation @ perifocal_position(a, e, 1.1), atol=ATOL)
    assert np.allclose(velocity, rotation @ perifocal_velocity(a, e, 1.1, mu), atol=ATOL)


def test_velocity_requires_an_elliptical_orbit():
    with pytest.raises(ValueError):
        perifocal_velocity(1.0, 1.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        perifocal_position(1.0, 1.2, 0.5)
