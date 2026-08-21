"""Missing-data policy tests (roadmap section 22, "Missing data").

The governing requirement is one sentence long:

    No missing-data test should silently create Earth-like values.

These tests are the regression guard for the P0 corrections 3.3, 3.4, 4.4
and 4.6.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.coordinates.frames import distance_from_parallax
from astro_explorer.data.schema import build_planet_record, parse_float
from astro_explorer.physics.ephemeris import semimajor_axis_from_period
from astro_explorer.physics.orbital_elements import OrbitalElements, PhaseKnowledge
from astro_explorer.provenance import Parameter, Status, assumed, measured, unknown

EARTH_LIKE_DEFAULTS = {"semimajor_axis": 1.0, "eccentricity": 0.0, "period": 365.25}


def _row(**overrides):
    row = {"pl_name": "Test b", "hostname": "Test"}
    row.update(overrides)
    return row


# -- semimajor axis (roadmap 3.3) -------------------------------------------


def test_missing_semimajor_axis_never_becomes_one_au():
    """The original code did pl_orbsmax.fillna(1.0)."""
    record = build_planet_record(_row())
    axis = record.elements.semimajor_axis
    assert not axis.is_known
    assert axis.status is Status.UNKNOWN


def test_missing_semimajor_axis_is_derived_when_possible():
    record = build_planet_record(_row(pl_orbper=365.256, st_mass=1.0))
    axis = record.elements.semimajor_axis
    assert axis.status is Status.DERIVED
    assert np.isclose(axis.value, 1.0, rtol=1e-3)
    assert "kepler3" in axis.provenance


def test_derived_axis_is_distinguishable_from_a_published_one():
    """A derived 1 AU must not look like a measured 1 AU."""
    derived_record = build_planet_record(_row(pl_orbper=365.256, st_mass=1.0))
    published_record = build_planet_record(_row(pl_orbsmax=1.0))

    assert derived_record.elements.semimajor_axis.status is Status.DERIVED
    assert published_record.elements.semimajor_axis.status is Status.MEASURED
    assert "derived" in derived_record.elements.semimajor_axis.format()


def test_axis_stays_unknown_without_a_stellar_mass():
    record = build_planet_record(_row(pl_orbper=3.5))
    assert not record.elements.semimajor_axis.is_known


def test_derivation_refuses_nonsense_inputs():
    assert not semimajor_axis_from_period(0.0, 1.0).is_known
    assert not semimajor_axis_from_period(-5.0, 1.0).is_known
    assert not semimajor_axis_from_period(365.0, 0.0).is_known
    assert not semimajor_axis_from_period(365.0, None).is_known
    assert not semimajor_axis_from_period(np.nan, 1.0).is_known


def test_for_display_does_not_invent_a_semimajor_axis():
    """Unknown a means no orbit at all, not a one-AU orbit."""
    display = OrbitalElements().for_display()
    assert not display.semimajor_axis.is_known


# -- eccentricity (roadmap 3.4) ---------------------------------------------


def test_missing_eccentricity_stays_unknown_in_the_record():
    """The original code did pl_orbeccen.fillna(0.0)."""
    record = build_planet_record(_row(pl_orbsmax=0.05))
    assert not record.elements.eccentricity.is_known
    assert record.elements.eccentricity.status is Status.UNKNOWN


def test_zero_eccentricity_is_only_a_labelled_display_assumption():
    record = build_planet_record(_row(pl_orbsmax=0.05))
    display = record.elements.for_display()

    assert display.eccentricity.value == 0.0
    assert display.eccentricity.status is Status.ASSUMED_FOR_VISUALIZATION
    assert display.eccentricity.is_assumed
    assert not display.eccentricity.is_scientific
    assert "circular" in display.eccentricity.note


def test_a_published_zero_eccentricity_is_not_an_assumption():
    record = build_planet_record(_row(pl_orbsmax=0.05, pl_orbeccen=0.0))
    assert record.elements.eccentricity.status is Status.MEASURED
    assert record.elements.for_display().eccentricity.status is Status.MEASURED


def test_display_normalisation_does_not_mutate_the_original():
    elements = OrbitalElements(semimajor_axis=measured(1.0, u.au))
    elements.for_display()
    assert elements.eccentricity.status is Status.UNKNOWN


# -- orbital orientation (roadmap 4.3) --------------------------------------


def test_ascending_node_is_never_published_for_exoplanets():
    record = build_planet_record(_row(pl_orbsmax=0.05, pl_orbincl=87.0))
    assert not record.elements.longitude_of_ascending_node.is_known
    assert not record.elements.orientation_known


def test_orientation_description_states_what_is_measured():
    record = build_planet_record(_row(pl_orbsmax=0.05, pl_orbincl=87.0))
    lines = "\n".join(record.elements.for_display().describe_orientation())
    assert "Inclination: 87" in lines
    assert "Ascending node: unknown" in lines


# -- phase (roadmap 4.5) -----------------------------------------------------


def test_phase_is_not_computable_without_an_epoch():
    record = build_planet_record(_row(pl_orbsmax=0.05, pl_orbeccen=0.0, pl_orbper=3.5))
    assert record.elements.phase_knowledge is PhaseKnowledge.ORBIT_SHAPE_KNOWN
    assert not record.elements.can_compute_current_position
    assert record.elements.mean_anomaly_at(2460000.5) is None


def test_phase_is_computable_with_a_periastron_epoch():
    record = build_planet_record(
        _row(pl_orbsmax=0.05, pl_orbeccen=0.1, pl_orbper=4.0, pl_orbtper=2450000.0)
    )
    assert record.elements.can_compute_current_position
    # One full period later the mean anomaly returns to zero.
    assert np.isclose(record.elements.mean_anomaly_at(2450004.0), 0.0, atol=1e-9)


def test_transit_epoch_also_constrains_the_phase():
    record = build_planet_record(
        _row(
            pl_orbsmax=0.05,
            pl_orbeccen=0.0,
            pl_orbper=4.0,
            pl_tranmid=2450000.0,
            pl_orblper=90.0,
        )
    )
    assert record.elements.can_compute_current_position
    assert record.elements.mean_anomaly_at(2450000.0) is not None


# -- stellar mass ------------------------------------------------------------


def test_unknown_stellar_mass_leaves_luminosity_unknown():
    record = build_planet_record(_row(pl_orbsmax=0.05))
    assert not record.host.mass.is_known
    assert not record.host.luminosity.is_known
    assert not record.equilibrium_temperature.is_known


# -- parallax (roadmap 4.6) --------------------------------------------------


@pytest.mark.parametrize("parallax", [-5.0, -0.001, 0.0, np.nan, None])
def test_unusable_parallax_never_becomes_a_placeholder_distance(parallax):
    """The prototype mapped bad parallaxes to one billion parsecs."""
    result = distance_from_parallax(parallax)
    assert not result.is_known
    assert result.status is Status.UNKNOWN


def test_unusable_parallax_falls_back_to_a_catalogue_distance():
    result = distance_from_parallax(-5.0, catalog_distance_pc=42.0)
    assert result.is_known
    assert np.isclose(result.value, 42.0)


def test_good_parallax_inverts_correctly():
    result = distance_from_parallax(100.0)  # 100 mas -> 10 pc
    assert np.isclose(result.value_in(u.pc), 10.0, rtol=1e-6)
    assert result.status is Status.DERIVED


# -- the Parameter type itself -----------------------------------------------


def test_nan_degrades_to_unknown_not_to_a_measurement():
    assert Parameter(np.nan, u.au, status=Status.MEASURED).status is Status.UNKNOWN
    assert not measured(None, u.au).is_known


def test_require_raises_rather_than_substituting():
    with pytest.raises(ValueError, match="refusing to substitute"):
        unknown(u.au).require(u.au)


def test_assumed_values_are_never_scientific():
    parameter = assumed(1.0, u.au, note="placeholder")
    assert parameter.is_known
    assert not parameter.is_scientific
    assert parameter.is_assumed


def test_negative_lower_errors_are_normalised():
    """IPAC and NASA write the lower error as a negative number."""
    parameter = measured(1.0, u.au, error_plus=0.1, error_minus=-0.2)
    assert parameter.error_minus == 0.2


def test_parse_float_does_not_invent_a_default():
    assert np.isnan(parse_float("null"))
    assert np.isnan(parse_float(None))
    assert np.isnan(parse_float(""))
    assert parse_float("<0.5") == 0.5


def test_no_record_field_silently_equals_an_earth_like_default():
    """The end-to-end guard for this whole file."""
    record = build_planet_record(_row())
    for name, forbidden in EARTH_LIKE_DEFAULTS.items():
        parameter = getattr(record.elements, name)
        assert not (parameter.is_known and parameter.value == forbidden), (
            "{0} silently became the Earth-like default {1}".format(name, forbidden)
        )
