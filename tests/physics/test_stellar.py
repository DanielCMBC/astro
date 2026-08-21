"""Stellar physics tests (roadmap section 22, "Stellar physics")."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.physics.constants import (
    AU_PER_PARSEC,
    B_WIEN,
    PARSEC_PER_AU,
    SOLAR_EFFECTIVE_TEMPERATURE,
)
from astro_explorer.physics.ephemeris import (
    kepler_third_law_residual,
    period_from_semimajor_axis,
    semimajor_axis_from_period,
)
from astro_explorer.physics.radiation import (
    SpectrumModel,
    blackbody_curve,
    planck_spectral_radiance,
    stefan_boltzmann_flux,
    wien_peak_wavelength,
)
from astro_explorer.physics.stellar import (
    DiagramKind,
    absolute_magnitude_from_luminosity,
    equilibrium_temperature,
    habitable_zone_au,
    luminosity_from_radius_and_teff,
)
from astro_explorer.provenance import Status, measured, unknown

SOLAR_TEFF = 5772.0


# -- blackbody ---------------------------------------------------------------


def test_wien_peak_for_the_sun():
    """The Sun peaks around 500 nm."""
    peak = wien_peak_wavelength(SOLAR_TEFF * u.K).to_value(u.nm)
    assert 495.0 < peak < 510.0


def test_wien_constant_matches_the_accepted_value():
    """B_WIEN is derived from h, c, k_B, not quoted; check it anyway."""
    assert np.isclose(B_WIEN.to_value(u.m * u.K), 2.897771955e-3, rtol=1e-7)


def test_wien_law_is_inverse_in_temperature():
    assert np.isclose(
        wien_peak_wavelength(2000 * u.K).to_value(u.nm),
        2.0 * wien_peak_wavelength(4000 * u.K).to_value(u.nm),
    )


def test_planck_curve_peaks_at_the_wien_wavelength():
    curve = blackbody_curve(SOLAR_TEFF * u.K, samples=20000)
    peak_index = int(np.argmax(curve.normalized))
    peak_nm = curve.wavelength[peak_index].to_value(u.nm)
    assert np.isclose(peak_nm, curve.peak_wavelength.to_value(u.nm), rtol=2e-3)


def test_planck_integrates_to_stefan_boltzmann():
    """Integrating B_lambda over wavelength and solid angle gives sigma T^4."""
    temperature = 3000.0 * u.K
    lam = np.logspace(-8, -2, 200000) * u.m
    radiance = planck_spectral_radiance(lam, temperature)
    integral = np.trapezoid(radiance.value, lam.value) * np.pi  # over the hemisphere
    assert np.isclose(integral, stefan_boltzmann_flux(temperature).value, rtol=1e-3)


def test_blackbody_is_labelled_an_approximation():
    """Roadmap 3.7: the UI must say this is an ideal blackbody."""
    curve = blackbody_curve(SOLAR_TEFF * u.K)
    assert curve.model is SpectrumModel.IDEAL_BLACKBODY
    assert "blackbody" in curve.label.lower()
    assert "absorption lines" in curve.caveat


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError):
        planck_spectral_radiance(500e-9 * u.m, 0 * u.K)
    with pytest.raises(ValueError):
        wien_peak_wavelength(-10 * u.K)


# -- luminosity --------------------------------------------------------------


def test_solar_luminosity_derivation():
    result = luminosity_from_radius_and_teff(1.0, SOLAR_TEFF)
    assert result.status is Status.DERIVED
    assert np.isclose(result.value, 1.0, rtol=2e-3)


def test_luminosity_scales_as_r2_t4():
    doubled_radius = luminosity_from_radius_and_teff(2.0, SOLAR_TEFF).value
    doubled_teff = luminosity_from_radius_and_teff(1.0, 2 * SOLAR_TEFF).value
    assert np.isclose(doubled_radius, 4.0, rtol=2e-3)
    assert np.isclose(doubled_teff, 16.0, rtol=2e-3)


def test_luminosity_is_unknown_without_inputs():
    assert not luminosity_from_radius_and_teff(None, SOLAR_TEFF).is_known
    assert not luminosity_from_radius_and_teff(1.0, None).is_known
    assert not luminosity_from_radius_and_teff(unknown(u.R_sun), SOLAR_TEFF).is_known


def test_luminosity_uncertainty_propagates():
    radius = measured(1.0, u.R_sun, error_plus=0.05, error_minus=0.05)
    teff = measured(SOLAR_TEFF, u.K, error_plus=50.0, error_minus=50.0)
    result = luminosity_from_radius_and_teff(radius, teff)
    # dL/L = 2 dR/R + 4 dT/T = 0.10 + 0.0347
    assert np.isclose(result.error_plus / result.value, 0.1347, rtol=1e-2)


def test_absolute_bolometric_magnitude_of_the_sun():
    assert np.isclose(absolute_magnitude_from_luminosity(1.0).value, 4.74, atol=1e-6)


# -- derived planetary quantities -------------------------------------------


def test_earth_equilibrium_temperature():
    result = equilibrium_temperature(1.0, 1.0, albedo=0.3)
    assert result.status is Status.DERIVED
    assert 250.0 < result.value < 260.0
    assert "albedo" in result.note


def test_habitable_zone_of_the_sun():
    zone = habitable_zone_au(1.0, SOLAR_TEFF)
    assert 0.9 < zone.inner.value < 1.0
    assert 1.6 < zone.outer.value < 1.8


def test_habitable_zone_refuses_to_extrapolate():
    """The Kopparapu fit is only valid for 2600-7200 K."""
    assert not habitable_zone_au(1.0, 30000.0).inner.is_known
    assert not habitable_zone_au(1.0, 1500.0).outer.is_known


# -- Kepler's third law ------------------------------------------------------


def test_earth_semimajor_axis_from_period():
    result = semimajor_axis_from_period(365.256, 1.0)
    assert result.status is Status.DERIVED
    assert np.isclose(result.value, 1.0, rtol=1e-4)


def test_earth_period_from_semimajor_axis():
    result = period_from_semimajor_axis(1.0, 1.0)
    assert np.isclose(result.value, 365.256, rtol=1e-4)


def test_jupiter_semimajor_axis():
    result = semimajor_axis_from_period(4332.59, 1.0, 9.5458e-4)
    assert np.isclose(result.value, 5.2044, rtol=2e-3)


def test_kepler_third_law_round_trips():
    for period in (0.5, 3.2, 88.0, 365.25, 4332.0):
        axis = semimajor_axis_from_period(period, 0.8)
        recovered = period_from_semimajor_axis(axis, 0.8)
        assert np.isclose(recovered.value, period, rtol=1e-9)


def test_kepler_residual_is_small_for_consistent_values():
    residual = kepler_third_law_residual(365.256, 1.0, 1.0)
    assert abs(residual) < 1e-3


def test_kepler_residual_flags_inconsistent_values():
    residual = kepler_third_law_residual(365.256, 2.0, 1.0)
    assert residual > 0.9


# -- units and constants -----------------------------------------------------


def test_au_to_parsec_is_the_real_conversion_not_0_005():
    """Roadmap 4.2: the prototype used 0.005 pc per AU."""
    assert np.isclose(PARSEC_PER_AU, 4.8481368e-6, rtol=1e-6)
    assert not np.isclose(PARSEC_PER_AU, 0.005, rtol=0.5)
    assert np.isclose(PARSEC_PER_AU * AU_PER_PARSEC, 1.0)


def test_solar_effective_temperature_carries_units():
    assert SOLAR_EFFECTIVE_TEMPERATURE.unit is u.K


def test_diagram_kinds_are_distinct():
    """Roadmap 3.6: the temperature-radius plot is not an HR diagram."""
    assert DiagramKind.HR_DIAGRAM.y_label != DiagramKind.TEMPERATURE_RADIUS.y_label
    assert "Hertzsprung" in DiagramKind.HR_DIAGRAM.title
    assert "radius" in DiagramKind.TEMPERATURE_RADIUS.title.lower()
