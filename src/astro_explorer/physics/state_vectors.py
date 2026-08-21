"""Position and velocity propagation, and the conserved quantities.

Implements sections 14-17 of ``ORBITAL_MECHANICS_FORMULAS_3D_EXOPLANET.md``.

Velocity matters even though nothing is rendered from it yet: a position
propagator can be wrong in ways a shape test cannot see, and the specific
orbital energy

.. math:: \\epsilon = \\frac{v^2}{2} - \\frac{\\mu}{r} = -\\frac{\\mu}{2a}

tests position and velocity *together*. It is also the state an eventual
N-body handoff needs (roadmap section 17).

Working units in this module are AU, days and solar masses, so ``mu`` is in
AU^3/day^2 and velocities come out in AU/day.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from ..provenance import Parameter
from .constants import G
from .kepler import solve_kepler
from .orientation import orbital_state, specific_angular_momentum

__all__ = [
    "MU_SUN_AU3_PER_DAY2",
    "gravitational_parameter",
    "StateVector",
    "state_at_eccentric_anomaly",
    "state_at_mean_anomaly",
    "specific_orbital_energy",
    "expected_specific_energy",
    "vis_viva_speed",
    "swept_area",
    "swept_area_binned",
]

#: Heliocentric gravitational parameter expressed in the module's working
#: units. Derived from Astropy's G and solar mass, never typed in by hand.
MU_SUN_AU3_PER_DAY2 = float((G * (1.0 * u.M_sun)).to_value(u.au**3 / u.day**2))


def gravitational_parameter(
    stellar_mass_solar: Parameter | float | None,
    planet_mass_solar: Parameter | float | None = None,
) -> float | None:
    """``mu = G(M* + Mp)`` in AU^3/day^2, or None when the mass is unknown.

    Returning ``None`` rather than a solar default keeps the missing-data
    policy intact: an orbit whose host mass is unpublished has no defined
    velocity, and the caller must say so instead of pretending.
    """
    if isinstance(stellar_mass_solar, Parameter):
        star = stellar_mass_solar.value_in(u.M_sun)
    else:
        star = stellar_mass_solar
    if star is None or not np.isfinite(star) or star <= 0.0:
        return None

    if isinstance(planet_mass_solar, Parameter):
        planet = planet_mass_solar.value_in(u.M_sun, 0.0)
    else:
        planet = planet_mass_solar or 0.0
    if planet is None or not np.isfinite(planet) or planet < 0.0:
        planet = 0.0

    return MU_SUN_AU3_PER_DAY2 * (star + planet)


@dataclass(frozen=True)
class StateVector:
    """An inertial position and, when computable, velocity.

    ``position`` is in AU and ``velocity`` in AU/day, both shaped
    ``(..., 3)``. ``velocity`` is ``None`` when the gravitational parameter
    was unavailable, which happens whenever the stellar mass is unpublished.
    """

    position: np.ndarray
    velocity: np.ndarray | None = None
    eccentric_anomaly: np.ndarray | None = None

    @property
    def radius(self) -> np.ndarray:
        """Distance from the focus, in AU."""
        return np.linalg.norm(self.position, axis=-1)

    @property
    def speed(self) -> np.ndarray | None:
        if self.velocity is None:
            return None
        return np.linalg.norm(self.velocity, axis=-1)

    @property
    def has_velocity(self) -> bool:
        return self.velocity is not None

    def angular_momentum(self) -> np.ndarray | None:
        """``h = r x v``, the orbit's conserved vector (reference section 14)."""
        if self.velocity is None:
            return None
        return np.cross(self.position, self.velocity)


def state_at_eccentric_anomaly(
    semimajor_axis: float,
    eccentricity: float,
    eccentric_anomaly,
    *,
    inclination: float = 0.0,
    argument_of_periapsis: float = 0.0,
    longitude_of_ascending_node: float = 0.0,
    mu: float | None = None,
) -> StateVector:
    """Inertial state from ``E`` and the orbital elements."""
    position, velocity = orbital_state(
        semimajor_axis,
        eccentricity,
        eccentric_anomaly,
        inclination=inclination,
        argument_of_periapsis=argument_of_periapsis,
        longitude_of_ascending_node=longitude_of_ascending_node,
        mu=mu,
    )
    return StateVector(
        position=position,
        velocity=velocity,
        eccentric_anomaly=np.asarray(eccentric_anomaly, dtype=np.float64),
    )


def state_at_mean_anomaly(
    semimajor_axis: float,
    eccentricity: float,
    mean_anomaly,
    *,
    inclination: float = 0.0,
    argument_of_periapsis: float = 0.0,
    longitude_of_ascending_node: float = 0.0,
    mu: float | None = None,
) -> StateVector:
    """Inertial state from ``M``: solves Kepler's equation, then transforms."""
    ecc_anomaly = solve_kepler(mean_anomaly, eccentricity)
    return state_at_eccentric_anomaly(
        semimajor_axis,
        eccentricity,
        ecc_anomaly,
        inclination=inclination,
        argument_of_periapsis=argument_of_periapsis,
        longitude_of_ascending_node=longitude_of_ascending_node,
        mu=mu,
    )


def vis_viva_speed(radius, semimajor_axis: float, mu: float):
    """``v = sqrt(mu (2/r - 1/a))`` (reference section 16)."""
    radius = np.asarray(radius, dtype=np.float64)
    return np.sqrt(mu * (2.0 / radius - 1.0 / semimajor_axis))


def specific_orbital_energy(state: StateVector, mu: float):
    """``epsilon = v^2/2 - mu/r`` from an actual propagated state.

    This is the *measured* energy: it uses the position and velocity the
    propagator produced, so it catches an error in either one.
    """
    if not state.has_velocity:
        raise ValueError("specific orbital energy needs a velocity; mu was not supplied")
    speed = state.speed
    return 0.5 * speed**2 - mu / state.radius


def expected_specific_energy(semimajor_axis: float, mu: float) -> float:
    """``epsilon = -mu / (2a)`` (reference section 17), the analytic value."""
    if semimajor_axis <= 0.0:
        raise ValueError("semimajor axis must be positive for a bound orbit")
    return -mu / (2.0 * semimajor_axis)


def swept_area(positions: np.ndarray) -> np.ndarray:
    """Areas of the triangles swept between consecutive positions.

    Kepler's second law (reference section 13) says these are equal for
    equal time steps. Using the full 3D cross product rather than a 2D
    determinant means the result is correct for an inclined orbit too, where
    a projected area would shrink with the tilt and hide a real error.

    This is a *polygonal* estimate: each straight chord cuts the corner of a
    curved sector, so it under-reads by O(dnu^2). The bias is negligible far
    from periapsis and severe close to it on an eccentric orbit - at
    ``e = 0.99`` a hundred-step polygon under-reads the periapsis sector by
    nearly 40%. Use :func:`swept_area_binned` with enough substeps when the
    absolute value matters; the error falls as the square of the step count.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("swept_area expects an (N, 3) array of positions")
    cross = np.cross(positions[:-1], positions[1:])
    return 0.5 * np.linalg.norm(cross, axis=-1)


def swept_area_binned(positions: np.ndarray, intervals: int) -> np.ndarray:
    """Swept area per interval, from a finely sampled position track.

    ``positions`` must hold ``intervals * substeps + 1`` points sampled at
    equal time steps. The chord areas within each interval are summed, which
    converges on the true sector area as the substep count rises.

    Returns an array of length ``intervals``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    total_steps = positions.shape[0] - 1
    if intervals <= 0 or total_steps % intervals:
        raise ValueError(
            "expected intervals * substeps + 1 positions; got {0} points for "
            "{1} intervals".format(positions.shape[0], intervals)
        )
    substeps = total_steps // intervals
    return swept_area(positions).reshape(intervals, substeps).sum(axis=1)


def areal_velocity(semimajor_axis: float, eccentricity: float, mu: float) -> float:
    """``dA/dt = h/2``, the constant of Kepler's second law."""
    return 0.5 * specific_angular_momentum(semimajor_axis, eccentricity, mu)


__all__ += ["areal_velocity"]
