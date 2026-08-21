"""Blackbody radiation, explicitly labelled as an approximation.

Roadmap sections 3.7 and 14.2.  Planck's law and Wien's displacement law are
kept from the original program; what changes is that the result carries a
label saying it is an *ideal* blackbody, because a real stellar spectrum has
line absorption and atmosphere-dependent structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from .constants import B_WIEN, C_LIGHT, H_PLANCK, K_BOLTZMANN, SIGMA_SB

__all__ = [
    "SpectrumModel",
    "BlackbodyCurve",
    "planck_spectral_radiance",
    "wien_peak_wavelength",
    "stefan_boltzmann_flux",
    "blackbody_curve",
]


class SpectrumModel(str, Enum):
    """Which stellar spectrum model produced a curve (roadmap 3.7)."""

    IDEAL_BLACKBODY = "IDEAL_BLACKBODY"
    OBSERVED_STELLAR_SPECTRUM = "OBSERVED_STELLAR_SPECTRUM"
    SYNTHETIC_STELLAR_ATMOSPHERE = "SYNTHETIC_STELLAR_ATMOSPHERE"

    @property
    def label(self) -> str:
        return {
            SpectrumModel.IDEAL_BLACKBODY: "Ideal blackbody approximation",
            SpectrumModel.OBSERVED_STELLAR_SPECTRUM: "Observed stellar spectrum",
            SpectrumModel.SYNTHETIC_STELLAR_ATMOSPHERE: "Synthetic stellar atmosphere",
        }[self]

    @property
    def caveat(self) -> str:
        if self is SpectrumModel.IDEAL_BLACKBODY:
            return (
                "Ideal blackbody approximation: a real stellar spectrum contains "
                "absorption lines and atmosphere-dependent structure."
            )
        return ""


def planck_spectral_radiance(wavelength, temperature):
    """Planck's law ``B_lambda(T)`` in W m^-3 sr^-1.

    .. math::
        B_\\lambda(T) = \\frac{2hc^2}{\\lambda^5}
                        \\frac{1}{e^{hc/\\lambda k T} - 1}

    Parameters
    ----------
    wavelength:
        A :class:`~astropy.units.Quantity` of length, or a plain array in
        metres.
    temperature:
        A Quantity in kelvin, or a plain value in kelvin.

    ``numpy.expm1`` is used for the denominator so the long-wavelength
    Rayleigh-Jeans tail does not lose precision to cancellation.
    """
    lam = u.Quantity(wavelength, u.m)
    temp = u.Quantity(temperature, u.K)
    if np.any(temp.value <= 0):
        raise ValueError("temperature must be positive")

    exponent = (H_PLANCK * C_LIGHT / (lam * K_BOLTZMANN * temp)).decompose().value
    # The steradian is implicit in the classical form; make it explicit so
    # downstream unit conversions cannot silently mix radiance with flux.
    radiance = (2.0 * H_PLANCK * C_LIGHT**2) / (lam**5 * np.expm1(exponent) * u.sr)
    return radiance.to(u.W / u.m**3 / u.sr)


def wien_peak_wavelength(temperature):
    """Wien's displacement law ``lambda_max = b / T``.

    ``b`` is derived from h, c and k_B in :mod:`.constants` rather than
    quoted as a literal.
    """
    temp = u.Quantity(temperature, u.K)
    if np.any(temp.value <= 0):
        raise ValueError("temperature must be positive")
    return (B_WIEN / temp).to(u.m)


def stefan_boltzmann_flux(temperature):
    """Bolometric surface flux ``sigma T^4`` in W/m^2."""
    temp = u.Quantity(temperature, u.K)
    return (SIGMA_SB * temp**4).to(u.W / u.m**2)


@dataclass(frozen=True)
class BlackbodyCurve:
    """A sampled Planck curve plus the labels the UI must display."""

    wavelength: u.Quantity
    radiance: u.Quantity
    normalized: np.ndarray
    temperature: u.Quantity
    peak_wavelength: u.Quantity
    model: SpectrumModel = SpectrumModel.IDEAL_BLACKBODY

    @property
    def label(self) -> str:
        return self.model.label

    @property
    def caveat(self) -> str:
        return self.model.caveat


def blackbody_curve(
    temperature,
    *,
    wavelength_min=100.0 * u.nm,
    wavelength_max=3000.0 * u.nm,
    samples: int = 1000,
) -> BlackbodyCurve:
    """Sample an ideal Planck curve over a wavelength range."""
    temp = u.Quantity(temperature, u.K)
    lam = np.linspace(
        u.Quantity(wavelength_min, u.m).value,
        u.Quantity(wavelength_max, u.m).value,
        samples,
    ) * u.m

    radiance = planck_spectral_radiance(lam, temp)
    peak = float(np.nanmax(radiance.value))
    normalized = radiance.value / peak if peak > 0 else np.zeros_like(radiance.value)

    return BlackbodyCurve(
        wavelength=lam,
        radiance=radiance,
        normalized=normalized,
        temperature=temp,
        peak_wavelength=wien_peak_wavelength(temp),
        model=SpectrumModel.IDEAL_BLACKBODY,
    )
