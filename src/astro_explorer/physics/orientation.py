"""The full 3D orbital orientation transform.

Implements sections 8-11 and 15 of
``ORBITAL_MECHANICS_FORMULAS_3D_EXOPLANET.md``, as a standalone module with
no dependency on the provenance or data layers, so it can be tested purely
numerically.

The chain this module owns is::

    E  ->  r_perifocal  ->  R_z(Omega) R_x(i) R_z(omega)  ->  r_inertial

Perifocal frame convention (the standard one):

* ``+x`` points at periapsis;
* ``+y`` lies in the orbital plane, 90 degrees ahead of periapsis in the
  direction of motion;
* ``+z`` is along the orbital angular momentum, so motion is counter-
  clockwise seen from ``+z``.

Reference frame convention: a right-handed inertial frame in which the
reference plane is ``z = 0`` and the ascending node is measured from ``+x``.
For an exoplanet the reference plane is the sky plane, so ``i = 90`` degrees
is an edge-on (transiting) orbit.

Everything here takes plain floats and NumPy arrays in radians and AU. Units
and provenance belong to the layer above; mixing them in here would make the
transform impossible to test in isolation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rotation_z",
    "rotation_x",
    "rotation_perifocal_to_inertial",
    "perifocal_position",
    "perifocal_velocity",
    "orbital_state_perifocal",
    "orbital_state",
    "position_from_eccentric_anomaly",
    "radius_from_eccentric_anomaly",
    "specific_angular_momentum",
    "node_vector",
    "orbit_normal",
]


def rotation_z(angle) -> np.ndarray:
    """Right-handed rotation about the z axis (reference section 9).

    .. math::
        R_z(\\theta) = \\begin{bmatrix}
        \\cos\\theta & -\\sin\\theta & 0 \\\\
        \\sin\\theta & \\cos\\theta & 0 \\\\
        0 & 0 & 1 \\end{bmatrix}
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_x(angle) -> np.ndarray:
    """Right-handed rotation about the x axis (reference section 10).

    .. math::
        R_x(i) = \\begin{bmatrix}
        1 & 0 & 0 \\\\
        0 & \\cos i & -\\sin i \\\\
        0 & \\sin i & \\cos i \\end{bmatrix}
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ],
        dtype=np.float64,
    )


def rotation_perifocal_to_inertial(
    inclination: float,
    argument_of_periapsis: float,
    longitude_of_ascending_node: float,
) -> np.ndarray:
    """Build ``R = R_z(Omega) R_x(i) R_z(omega)`` (reference section 8).

    Parameters
    ----------
    inclination:
        ``i``, radians. 0 leaves the orbit in the reference plane; ``pi/2``
        makes it edge-on.
    argument_of_periapsis:
        ``omega``, radians, measured in the orbital plane from the ascending
        node to periapsis.
    longitude_of_ascending_node:
        ``Omega``, radians, measured in the reference plane from ``+x`` to
        the ascending node.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` proper rotation: orthonormal with determinant ``+1``.

    Notes
    -----
    The three rotations are applied right to left, so ``omega`` acts first,
    inside the orbital plane, then the plane is tilted by ``i``, and finally
    the whole configuration is spun about the reference pole by ``Omega``.
    Composing them in any other order gives a different, wrong, orbit.
    """
    return (
        rotation_z(longitude_of_ascending_node)
        @ rotation_x(inclination)
        @ rotation_z(argument_of_periapsis)
    )


def perifocal_position(semimajor_axis: float, eccentricity: float, eccentric_anomaly):
    """Position in the orbital plane from ``E`` (reference section 4).

    .. math::
        x_p = a(\\cos E - e), \\quad
        y_p = a\\sqrt{1-e^2}\\sin E, \\quad
        z_p = 0

    Returns an array of shape ``(..., 3)`` in the same length unit as ``a``.
    """
    ecc_anomaly = np.asarray(eccentric_anomaly, dtype=np.float64)
    eccentricity = float(eccentricity)
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("perifocal_position handles elliptical orbits only (0 <= e < 1)")

    x = semimajor_axis * (np.cos(ecc_anomaly) - eccentricity)
    y = semimajor_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(ecc_anomaly)
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


def perifocal_velocity(
    semimajor_axis: float,
    eccentricity: float,
    eccentric_anomaly,
    mu: float,
):
    """Velocity in the orbital plane from ``E`` (reference section 15).

    .. math::
        \\mathbf v_p = \\frac{na}{1 - e\\cos E}
        \\begin{bmatrix} -\\sin E \\\\ \\sqrt{1-e^2}\\cos E \\\\ 0 \\end{bmatrix},
        \\quad n = \\sqrt{\\mu / a^3}

    ``mu`` must be consistent with ``a``: pass ``mu`` in AU^3/day^2 to get
    AU/day, for example. :mod:`astro_explorer.physics.state_vectors` provides
    ``gravitational_parameter`` for that.
    """
    ecc_anomaly = np.asarray(eccentric_anomaly, dtype=np.float64)
    eccentricity = float(eccentricity)
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("perifocal_velocity handles elliptical orbits only (0 <= e < 1)")
    if semimajor_axis <= 0.0 or mu <= 0.0:
        raise ValueError("semimajor axis and gravitational parameter must be positive")

    mean_motion = np.sqrt(mu / semimajor_axis**3)
    # r = a(1 - e cos E); the factor na/r is the standard result of
    # differentiating the perifocal position with respect to time.
    factor = mean_motion * semimajor_axis / (1.0 - eccentricity * np.cos(ecc_anomaly))

    vx = -factor * np.sin(ecc_anomaly)
    vy = factor * np.sqrt(1.0 - eccentricity**2) * np.cos(ecc_anomaly)
    return np.stack([vx, vy, np.zeros_like(vx)], axis=-1)


def radius_from_eccentric_anomaly(semimajor_axis: float, eccentricity: float, eccentric_anomaly):
    """``r = a(1 - e cos E)`` (reference section 5, second form)."""
    ecc_anomaly = np.asarray(eccentric_anomaly, dtype=np.float64)
    return semimajor_axis * (1.0 - eccentricity * np.cos(ecc_anomaly))


def _apply(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate ``(..., 3)`` vectors by a ``(3, 3)`` matrix."""
    return vectors @ rotation.T


def orbital_state_perifocal(
    semimajor_axis: float,
    eccentricity: float,
    eccentric_anomaly,
    mu: float | None = None,
):
    """Perifocal position and, when ``mu`` is given, velocity."""
    position = perifocal_position(semimajor_axis, eccentricity, eccentric_anomaly)
    if mu is None:
        return position, None
    velocity = perifocal_velocity(semimajor_axis, eccentricity, eccentric_anomaly, mu)
    return position, velocity


def orbital_state(
    semimajor_axis: float,
    eccentricity: float,
    eccentric_anomaly,
    *,
    inclination: float = 0.0,
    argument_of_periapsis: float = 0.0,
    longitude_of_ascending_node: float = 0.0,
    mu: float | None = None,
):
    """Full inertial state from orbital elements and ``E``.

    This is the single function the rest of the program should call for a
    3D orbital position. It performs, in order:

    1. perifocal position (and velocity, if ``mu`` is supplied);
    2. rotation by ``R_z(Omega) R_x(i) R_z(omega)``.

    Returns
    -------
    (position, velocity)
        ``position`` has shape ``(..., 3)``; ``velocity`` is ``None`` unless
        ``mu`` was supplied. Both are rotated by the same matrix, so the
        state stays self-consistent.
    """
    rotation = rotation_perifocal_to_inertial(
        inclination, argument_of_periapsis, longitude_of_ascending_node
    )
    position_pf, velocity_pf = orbital_state_perifocal(
        semimajor_axis, eccentricity, eccentric_anomaly, mu
    )
    position = _apply(rotation, position_pf)
    velocity = None if velocity_pf is None else _apply(rotation, velocity_pf)
    return position, velocity


def position_from_eccentric_anomaly(
    semimajor_axis: float,
    eccentricity: float,
    eccentric_anomaly,
    *,
    inclination: float = 0.0,
    argument_of_periapsis: float = 0.0,
    longitude_of_ascending_node: float = 0.0,
):
    """Inertial position only; a convenience wrapper over :func:`orbital_state`."""
    position, _ = orbital_state(
        semimajor_axis,
        eccentricity,
        eccentric_anomaly,
        inclination=inclination,
        argument_of_periapsis=argument_of_periapsis,
        longitude_of_ascending_node=longitude_of_ascending_node,
    )
    return position


def specific_angular_momentum(semimajor_axis: float, eccentricity: float, mu: float) -> float:
    """``h = sqrt(mu a (1 - e^2))`` (reference section 14)."""
    if semimajor_axis <= 0.0 or mu <= 0.0:
        raise ValueError("semimajor axis and gravitational parameter must be positive")
    return float(np.sqrt(mu * semimajor_axis * (1.0 - eccentricity**2)))


def orbit_normal(inclination: float, longitude_of_ascending_node: float) -> np.ndarray:
    """Unit vector along the orbital angular momentum.

    This is the perifocal ``+z`` axis carried through the rotation, and it is
    independent of ``omega`` - rotating periapsis within the orbital plane
    cannot tilt the plane. Tests use that invariance.
    """
    rotation = rotation_perifocal_to_inertial(inclination, 0.0, longitude_of_ascending_node)
    return rotation @ np.array([0.0, 0.0, 1.0])


def node_vector(longitude_of_ascending_node: float) -> np.ndarray:
    """Unit vector towards the ascending node, in the reference plane.

    By definition the node lies at ``Omega`` measured from ``+x``, in the
    ``z = 0`` plane, whatever the inclination.
    """
    return np.array(
        [
            np.cos(longitude_of_ascending_node),
            np.sin(longitude_of_ascending_node),
            0.0,
        ]
    )
