"""Orbital metadata semantics.

Review sections 9, 10, 11 and 13. The numerics were settled by the previous
milestone; these tests pin the *meaning* of the elements - whose orbit an
argument of periastron describes, which time system an epoch is quoted in,
and what the published set actually supports.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.physics.epoch import (
    BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS,
    TDB_MINUS_UTC_FALLBACK_SECONDS,
    Epoch,
    EpochKind,
    TimeScale,
)
from astro_explorer.physics.orbital_elements import OrbitalElements
from astro_explorer.physics.orbital_semantics import (
    OrbitValidity,
    PeriastronConvention,
    angular_difference,
    planet_to_stellar_reflex,
    resolve_argument_of_periapsis,
    stellar_reflex_to_planet,
)
from astro_explorer.provenance import Status, measured, unknown

DEG = np.pi / 180.0


# ==========================================================================
# Review section 13: the required 180-degree regression test
# ==========================================================================


@pytest.mark.parametrize(
    "star_omega",
    [0.0, 0.3, np.pi / 2, 2.0, np.pi, 4.0, 2 * np.pi - 0.1, -58.887 * DEG, -3.0],
)
def test_stellar_reflex_converts_to_planet_by_exactly_pi(star_omega):
    """The assertion review section 13 asks for, with angular wrapping."""
    planet_omega = stellar_reflex_to_planet(star_omega)
    assert angular_difference(planet_omega, star_omega) == pytest.approx(np.pi)


def test_the_conversion_is_its_own_inverse():
    """Applying it twice returns the original direction."""
    for omega in (0.0, 1.0, 2.5, 5.9, -2.0):
        there = stellar_reflex_to_planet(omega)
        back = planet_to_stellar_reflex(there)
        assert angular_difference(back, omega) == pytest.approx(0.0, abs=1e-12)


def test_the_conversion_output_is_wrapped_into_zero_to_two_pi():
    for omega in (-3.0, -0.5, 0.0, 7.0, 100.0):
        result = stellar_reflex_to_planet(omega)
        assert 0.0 <= result < 2.0 * np.pi


def test_the_conversion_is_exactly_180_degrees_not_approximately():
    assert stellar_reflex_to_planet(0.0) == pytest.approx(np.pi, abs=1e-15)
    assert stellar_reflex_to_planet(np.pi) == pytest.approx(0.0, abs=1e-15)


def test_conversion_on_a_parameter_returns_a_derived_value():
    star = measured(30.0, u.deg, error_plus=2.0, error_minus=2.0, provenance="ps.pl_orblper")
    planet = stellar_reflex_to_planet(star)

    assert planet.status is Status.DERIVED
    assert planet.to(u.deg).value == pytest.approx(210.0)
    # A rotation does not change how well the angle is known.
    assert planet.error_plus == pytest.approx(2.0)
    assert "180" in planet.note


def test_converting_an_unknown_angle_stays_unknown():
    assert not stellar_reflex_to_planet(unknown(u.rad)).is_known
    assert not planet_to_stellar_reflex(unknown(u.rad)).is_known


def test_angular_difference_handles_the_wrap_point():
    assert angular_difference(0.01, 2 * np.pi - 0.01) == pytest.approx(0.02)
    assert angular_difference(np.pi, -np.pi) == pytest.approx(0.0, abs=1e-12)
    assert angular_difference(0.0, np.pi) == pytest.approx(np.pi)


def test_angular_difference_is_never_negative_or_above_pi():
    rng = np.random.default_rng(0)
    first, second = rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500)
    delta = angular_difference(first, second)
    assert np.all(delta >= 0.0) and np.all(delta <= np.pi + 1e-12)


# ==========================================================================
# Review section 10: the convention model
# ==========================================================================


def test_the_raw_value_is_never_modified():
    """Whatever the convention, the catalogued number survives untouched."""
    raw = measured(-58.887 * DEG, u.rad, provenance="ps.pl_orblper")
    for convention in PeriastronConvention:
        elements = OrbitalElements(
            argument_of_periastron=raw, periastron_convention=convention
        )
        assert elements.argument_of_periastron.value == pytest.approx(raw.value)
        assert elements.argument_of_periastron.provenance == "ps.pl_orblper"


def test_planet_convention_passes_the_value_through_as_measured():
    raw = measured(1.0, u.rad)
    resolved = resolve_argument_of_periapsis(raw, PeriastronConvention.PLANET)
    assert resolved.status is Status.MEASURED
    assert resolved.value == pytest.approx(1.0)


def test_stellar_reflex_convention_rotates_and_marks_derived():
    raw = measured(1.0, u.rad)
    resolved = resolve_argument_of_periapsis(raw, PeriastronConvention.STELLAR_REFLEX)
    assert resolved.status is Status.DERIVED
    assert angular_difference(resolved.value, 1.0) == pytest.approx(np.pi)


def test_as_reported_uses_the_value_but_calls_it_an_assumption():
    """The central point of review section 10."""
    raw = measured(1.0, u.rad)
    resolved = resolve_argument_of_periapsis(raw, PeriastronConvention.AS_REPORTED)

    assert resolved.value == pytest.approx(1.0)
    assert resolved.status is Status.ASSUMED_FOR_VISUALIZATION
    assert not resolved.is_scientific
    assert "stellar reflex" in resolved.note


def test_an_absent_angle_has_no_convention():
    elements = OrbitalElements()
    assert elements.periastron_convention is PeriastronConvention.UNKNOWN
    assert not elements.argument_of_periapsis_planet.is_known


def test_only_stated_conventions_count_as_determinate():
    assert PeriastronConvention.PLANET.is_determinate
    assert PeriastronConvention.STELLAR_REFLEX.is_determinate
    assert not PeriastronConvention.AS_REPORTED.is_determinate
    assert not PeriastronConvention.UNKNOWN.is_determinate


def test_the_as_reported_caveat_names_the_consequence():
    caveat = PeriastronConvention.AS_REPORTED.caveat
    assert "180 degrees" in caveat
    assert PeriastronConvention.PLANET.caveat == ""


def test_the_two_conventions_place_periapsis_on_opposite_sides():
    """End to end: the choice moves the planet to the other side of the star."""
    from astro_explorer.physics.orientation import position_from_eccentric_anomaly

    a, e = 1.0, 0.6
    star_omega = 30.0 * DEG
    planet_omega = stellar_reflex_to_planet(star_omega)

    as_star = position_from_eccentric_anomaly(a, e, 0.0, argument_of_periapsis=star_omega)
    as_planet = position_from_eccentric_anomaly(a, e, 0.0, argument_of_periapsis=planet_omega)

    # Same distance, opposite direction.
    assert np.linalg.norm(as_star) == pytest.approx(np.linalg.norm(as_planet))
    assert np.allclose(as_star, -as_planet, atol=1e-12)


def test_display_normalisation_carries_the_resolved_angle():
    raw = measured(1.0, u.rad)
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        argument_of_periastron=raw,
        periastron_convention=PeriastronConvention.STELLAR_REFLEX,
    )
    display = elements.for_display()

    assert angular_difference(display.argument_of_periastron.value, 1.0) == pytest.approx(np.pi)
    assert display.periastron_convention is PeriastronConvention.PLANET
    # ...and the original is untouched.
    assert elements.argument_of_periastron.value == pytest.approx(1.0)
    assert elements.periastron_convention is PeriastronConvention.STELLAR_REFLEX


def test_a_stated_planet_convention_needs_no_normalisation():
    raw = measured(1.0, u.rad)
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        argument_of_periastron=raw,
        periastron_convention=PeriastronConvention.PLANET,
    )
    assert elements.for_display().argument_of_periastron.status is Status.MEASURED
    assert not elements.periastron_convention_is_assumed


def test_an_unstated_convention_is_flagged_as_assumed():
    elements = OrbitalElements(argument_of_periastron=measured(1.0, u.rad))
    assert elements.periastron_convention is PeriastronConvention.AS_REPORTED
    assert elements.periastron_convention_is_assumed


# ==========================================================================
# Review section 11: orbit validity states
# ==========================================================================


def _elements(**kwargs) -> OrbitalElements:
    base = dict(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.1),
        period=measured(365.0, u.day),
    )
    base.update(kwargs)
    return OrbitalElements(**base)


def test_geometry_valid_needs_only_a_and_e():
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au), eccentricity=measured(0.1)
    )
    assert OrbitValidity.GEOMETRY_VALID in elements.validity
    assert OrbitValidity.PHASE_VALID not in elements.validity


def test_no_shape_means_no_geometry():
    assert OrbitValidity.GEOMETRY_VALID not in OrbitalElements().validity
    assert OrbitalElements().validity is OrbitValidity.NONE


def test_phase_valid_needs_an_epoch_and_a_period():
    without = _elements()
    assert OrbitValidity.PHASE_VALID not in without.validity

    with_epoch = _elements(epoch_periastron=measured(2450000.0, u.day))
    assert OrbitValidity.PHASE_VALID in with_epoch.validity


def test_an_epoch_without_a_period_is_not_phase_valid():
    elements = OrbitalElements(
        semimajor_axis=measured(1.0, u.au),
        eccentricity=measured(0.1),
        epoch_periastron=measured(2450000.0, u.day),
    )
    assert OrbitValidity.PHASE_VALID not in elements.validity


def test_orientation_partial_when_some_angles_are_known():
    elements = _elements(inclination=measured(0.5, u.rad))
    assert OrbitValidity.ORIENTATION_PARTIAL in elements.validity
    assert OrbitValidity.ORIENTATION_FULL not in elements.validity


def test_no_angles_means_no_orientation_flag_at_all():
    validity = _elements().validity
    assert OrbitValidity.ORIENTATION_PARTIAL not in validity
    assert OrbitValidity.ORIENTATION_FULL not in validity


def test_orientation_full_requires_all_three_angles():
    elements = _elements(
        inclination=measured(0.5, u.rad),
        argument_of_periastron=measured(1.0, u.rad),
        longitude_of_ascending_node=measured(2.0, u.rad),
        periastron_convention=PeriastronConvention.PLANET,
    )
    assert OrbitValidity.ORIENTATION_FULL in elements.validity
    assert OrbitValidity.ORIENTATION_PARTIAL not in elements.validity


def test_orientation_full_also_requires_a_stated_convention():
    """Three known angles under an unstated convention is still ambiguous."""
    elements = _elements(
        inclination=measured(0.5, u.rad),
        argument_of_periastron=measured(1.0, u.rad),
        longitude_of_ascending_node=measured(2.0, u.rad),
        periastron_convention=PeriastronConvention.AS_REPORTED,
    )
    assert OrbitValidity.ORIENTATION_FULL not in elements.validity
    assert OrbitValidity.ORIENTATION_PARTIAL in elements.validity


def test_partial_and_full_are_mutually_exclusive():
    for convention in PeriastronConvention:
        elements = _elements(
            inclination=measured(0.5, u.rad),
            argument_of_periastron=measured(1.0, u.rad),
            longitude_of_ascending_node=measured(2.0, u.rad),
            periastron_convention=convention,
        )
        validity = elements.validity
        both = (
            OrbitValidity.ORIENTATION_FULL in validity
            and OrbitValidity.ORIENTATION_PARTIAL in validity
        )
        assert not both


def test_validity_flags_combine():
    elements = _elements(
        inclination=measured(0.5, u.rad),
        epoch_periastron=measured(2450000.0, u.day),
    )
    validity = elements.validity
    assert OrbitValidity.GEOMETRY_VALID in validity
    assert OrbitValidity.PHASE_VALID in validity
    assert OrbitValidity.ORIENTATION_PARTIAL in validity


def test_validity_describes_itself():
    lines = "\n".join(_elements().validity.describe())
    assert "GEOMETRY_VALID" in lines
    assert "NONE" in "\n".join(OrbitalElements().validity.describe())


# ==========================================================================
# Review section 11: epochs and time systems
# ==========================================================================


def test_mission_offsets_convert_to_full_julian_dates():
    assert TimeScale.BKJD.to_bjd(0.0) == pytest.approx(2454833.0)
    assert TimeScale.BTJD.to_bjd(0.0) == pytest.approx(2457000.0)
    assert TimeScale.BJD_TDB.to_bjd(2458882.344) == pytest.approx(2458882.344)


def test_the_offsets_round_trip():
    for scale in TimeScale:
        assert scale.from_bjd(scale.to_bjd(1234.5)) == pytest.approx(1234.5)


def test_ignoring_a_mission_offset_would_be_catastrophic():
    """13 years, not a rounding error - worth having a type for."""
    assert TimeScale.BKJD.offset_to_bjd / 365.25 > 6700.0


def test_only_stated_scales_are_determinate():
    assert TimeScale.BJD_TDB.is_determinate
    assert TimeScale.BKJD.is_determinate
    assert not TimeScale.JD_UNSPECIFIED.is_determinate
    assert not TimeScale.UNKNOWN.is_determinate


def test_a_stated_barycentric_scale_has_no_ambiguity():
    assert TimeScale.BJD_TDB.uncertainty_seconds == 0.0
    assert TimeScale.BKJD.uncertainty_seconds == 0.0


def test_an_unstated_scale_reports_its_worst_case():
    worst = TimeScale.JD_UNSPECIFIED.uncertainty_seconds
    assert worst == pytest.approx(
        BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS + TDB_MINUS_UTC_FALLBACK_SECONDS
    )
    assert 500.0 < worst < 600.0


def test_barycentric_classification():
    assert TimeScale.BJD_TDB.is_barycentric
    assert TimeScale.BTJD.is_barycentric
    assert not TimeScale.HJD_UTC.is_barycentric


def test_an_epoch_reports_its_kind_and_scale():
    epoch = Epoch(measured(2458882.344, u.day), EpochKind.PERIASTRON, TimeScale.BJD_TDB)
    assert epoch.as_bjd() == pytest.approx(2458882.344)
    text = epoch.describe()
    assert "periastron" in text
    assert "TDB" in text


def test_an_unstated_scale_says_so_in_the_description():
    epoch = Epoch(measured(2458882.344, u.day), EpochKind.PERIASTRON, TimeScale.JD_UNSPECIFIED)
    assert "not stated" in epoch.describe()
    assert "worst case" in epoch.describe()


def test_a_missing_epoch_says_so():
    assert "not published" in Epoch.missing(EpochKind.PERIASTRON).describe()
    assert not Epoch.missing().is_known


def test_a_kepler_epoch_is_offset_before_use():
    epoch = Epoch(measured(100.0, u.day), EpochKind.TRANSIT, TimeScale.BKJD)
    assert epoch.as_bjd() == pytest.approx(2454933.0)


def test_a_transit_epoch_depends_on_the_argument_of_periapsis():
    """So it inherits the convention ambiguity; periastron does not."""
    assert EpochKind.TRANSIT.needs_argument_of_periapsis
    assert not EpochKind.PERIASTRON.needs_argument_of_periapsis


def test_phase_uncertainty_is_negligible_for_a_long_period():
    epoch = Epoch(measured(2458882.344, u.day), EpochKind.PERIASTRON, TimeScale.JD_UNSPECIFIED)
    fraction = epoch.phase_uncertainty_fraction(111.436765)
    assert fraction is not None
    assert fraction < 1e-4


def test_phase_uncertainty_needs_a_period():
    epoch = Epoch(measured(2458882.344, u.day), EpochKind.PERIASTRON, TimeScale.BJD_TDB)
    assert epoch.phase_uncertainty_fraction(None) is None
    assert epoch.phase_uncertainty_fraction(0.0) is None


def test_elements_expose_their_epochs_with_metadata():
    elements = _elements(
        epoch_periastron=measured(2450000.0, u.day),
        epoch_transit=measured(2450001.0, u.day),
        epoch_scale=TimeScale.BJD_TDB,
        reference="Someone et al. 2024",
    )
    epochs = elements.epochs
    assert {e.kind for e in epochs} == {EpochKind.PERIASTRON, EpochKind.TRANSIT}
    assert all(e.scale is TimeScale.BJD_TDB for e in epochs)
    assert all(e.reference == "Someone et al. 2024" for e in epochs)


def test_no_epochs_when_none_are_published():
    assert _elements().epochs == []
