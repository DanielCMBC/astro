"""Kepler's equation and anomaly conversions (roadmap sections 4.1 and 8.3).

The prototype in the 3D branch used the first-order approximation
``E ~ M + e*sin(M)``.  That is accurate only for small eccentricity and
degrades badly as ``e`` grows.  This module solves

.. math::  M = E - e \\sin E

properly, with a safeguarded Newton-Raphson iteration that is vectorised
over NumPy arrays and provably convergent for every ``0 <= e < 1``.

The renderer must never solve orbital mechanics itself; it calls here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "solve_kepler",
    "true_anomaly_from_eccentric",
    "eccentric_from_true_anomaly",
    "mean_anomaly_from_eccentric",
    "kepler_residual",
]

#: Newton-Raphson stops when |E - e sinE - M| falls below this (radians).
DEFAULT_TOLERANCE = 1e-13
DEFAULT_MAX_ITER = 60


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def _initial_guess(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    """Starting estimate for E.

    Uses the classical ``M + e sin M`` seed for mild orbits and a
    cube-root seed near periapsis of a very eccentric orbit, where the
    classical seed converges slowly.
    """
    guess = mean_anomaly + eccentricity * np.sin(mean_anomaly)

    # Near-parabolic regime close to periapsis: the classical seed converges
    # slowly there, so fall back to the cube-root seed of the series
    # M ~ (1-e)E + e E^3/6.
    steep = (eccentricity > 0.8) & (np.abs(mean_anomaly) < 0.5)
    if np.any(steep):
        alpha = 6.0 * mean_anomaly / np.maximum(eccentricity, 1e-12)
        guess = np.where(steep, np.cbrt(alpha), guess)

    return guess


def solve_kepler(
    mean_anomaly,
    eccentricity,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITER,
):
    """Solve ``M = E - e sin E`` for the eccentric anomaly ``E``.

    Parameters
    ----------
    mean_anomaly:
        Mean anomaly in radians.  Scalar or array; any range is accepted and
        internally wrapped to (-pi, pi].
    eccentricity:
        Orbital eccentricity, ``0 <= e < 1``.  Scalar or array broadcastable
        against ``mean_anomaly``.
    tolerance:
        Absolute convergence tolerance on the Kepler residual, in radians.
    max_iterations:
        Hard cap on Newton steps.  The bisection safeguard guarantees the
        bracket halves every rejected step, so this is never reached in
        practice.

    Returns
    -------
    numpy.ndarray or float
        The eccentric anomaly in radians, wrapped consistently with the
        wrapped mean anomaly.  A scalar input returns a Python float.

    Raises
    ------
    ValueError
        If any eccentricity is negative or >= 1.  Hyperbolic and parabolic
        orbits are a different equation and are not silently coerced.
    """
    mean = np.asarray(mean_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    scalar = mean.ndim == 0 and ecc.ndim == 0

    if np.any(ecc < 0.0):
        raise ValueError("eccentricity must be non-negative")
    if np.any(ecc >= 1.0):
        raise ValueError(
            "solve_kepler handles elliptical orbits only (0 <= e < 1); "
            "got e >= 1 which is parabolic/hyperbolic"
        )

    mean, ecc = np.broadcast_arrays(mean, ecc)
    mean = _wrap_to_pi(np.array(mean, dtype=np.float64, copy=True))
    ecc = np.array(ecc, dtype=np.float64, copy=True)

    # Solve on the positive half and mirror back: the equation is odd in M,
    # which keeps the safeguard bracket simple and symmetric.
    sign = np.where(mean < 0.0, -1.0, 1.0)
    m_abs = np.abs(mean)

    lower = np.zeros_like(m_abs)
    upper = np.full_like(m_abs, np.pi)
    ecc_anom = np.clip(_initial_guess(m_abs, ecc), lower, upper)

    for _ in range(max_iterations):
        residual = ecc_anom - ecc * np.sin(ecc_anom) - m_abs
        if np.all(np.abs(residual) < tolerance):
            break

        # f is monotonically increasing in E, so the sign of the residual
        # tells us which side of the root we are on.
        lower = np.where(residual < 0.0, ecc_anom, lower)
        upper = np.where(residual > 0.0, ecc_anom, upper)

        derivative = 1.0 - ecc * np.cos(ecc_anom)
        # derivative >= 1 - e > 0 for e < 1, but guard against 0/0 anyway.
        derivative = np.where(np.abs(derivative) < 1e-15, 1e-15, derivative)
        candidate = ecc_anom - residual / derivative

        # Reject Newton steps that leave the bracket; bisect instead.
        outside = (candidate <= lower) | (candidate >= upper)
        ecc_anom = np.where(outside, 0.5 * (lower + upper), candidate)

    ecc_anom = sign * ecc_anom
    if scalar:
        return float(ecc_anom)
    return ecc_anom


def kepler_residual(eccentric_anomaly, eccentricity, mean_anomaly):
    """Return ``E - e sin E - M`` wrapped to (-pi, pi]; a solver diagnostic."""
    ecc_anom = np.asarray(eccentric_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    mean = np.asarray(mean_anomaly, dtype=np.float64)
    return _wrap_to_pi(ecc_anom - ecc * np.sin(ecc_anom) - mean)


def true_anomaly_from_eccentric(eccentric_anomaly, eccentricity):
    """Convert eccentric anomaly ``E`` to true anomaly ``nu``.

    Uses the half-angle form, which is numerically stable across the whole
    orbit including near periapsis and apoapsis.
    """
    ecc_anom = np.asarray(eccentric_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    scalar = ecc_anom.ndim == 0 and ecc.ndim == 0

    result = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(0.5 * ecc_anom),
        np.sqrt(1.0 - ecc) * np.cos(0.5 * ecc_anom),
    )
    return float(result) if scalar else result


def eccentric_from_true_anomaly(true_anomaly, eccentricity):
    """Inverse of :func:`true_anomaly_from_eccentric`."""
    nu = np.asarray(true_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    scalar = nu.ndim == 0 and ecc.ndim == 0

    result = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(0.5 * nu),
        np.sqrt(1.0 + ecc) * np.cos(0.5 * nu),
    )
    return float(result) if scalar else result


def mean_anomaly_from_eccentric(eccentric_anomaly, eccentricity):
    """Forward Kepler equation ``M = E - e sin E``."""
    ecc_anom = np.asarray(eccentric_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    scalar = ecc_anom.ndim == 0 and ecc.ndim == 0

    result = ecc_anom - ecc * np.sin(ecc_anom)
    return float(result) if scalar else result
