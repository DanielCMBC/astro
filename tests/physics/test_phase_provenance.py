"""Phase provenance vocabulary (review sections 7, 8, 9, 11).

The distinction this vocabulary exists to make: a temporal anchor can be
*observed* while the orbital orientation used to read it is not. Kepler-11
is that case for six planets at once, and neither "positioned from a
published epoch" nor "assumed" describes it honestly.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.physics.orbital_elements import OrbitalElements
from astro_explorer.physics.orbital_semantics import PeriastronConvention
from astro_explorer.physics.phase import (
    AnomalyMapping,
    PhaseAnchor,
    PhaseProvenance,
    PhaseSolution,
    PhaseStatus,
    conjunction_offset_scale,
)
from astro_explorer.provenance import Status, measured, unknown
from astro_explorer.text import NULL_SPELLINGS, clean_text, display_text

DEG = np.pi / 180.0
EPOCH = 2455590.0


def _elements(**kwargs) -> OrbitalElements:
    base = dict(
        semimajor_axis=measured(0.155, u.au),
        eccentricity=measured(0.004),
        period=measured(22.6845, u.day),
    )
    base.update(kwargs)
    return OrbitalElements(**base)


# ==========================================================================
# The four anchoring cases
# ==========================================================================


def test_a_periastron_epoch_is_fully_constrained():
    """M = 0 at t0; no argument of periastron is involved at all."""
    solution = _elements(epoch_periastron=measured(2450000.0, u.day)).phase_at(EPOCH)

    assert solution.provenance is PhaseProvenance.PERIASTRON_EPOCH
    assert solution.anchor is PhaseAnchor.OBSERVED
    assert solution.mapping is AnomalyMapping.DIRECT
    assert solution.status is PhaseStatus.CONSTRAINED
    assert not solution.is_assumed


def test_a_periastron_epoch_is_unharmed_by_a_missing_omega():
    """It needs no omega, so lacking one must not weaken it."""
    solution = _elements(
        epoch_periastron=measured(2450000.0, u.day),
        argument_of_periastron=unknown(u.rad),
    ).phase_at(EPOCH)
    assert solution.status is PhaseStatus.CONSTRAINED


def test_a_transit_epoch_with_a_stated_omega_is_fully_constrained():
    solution = _elements(
        epoch_transit=measured(2455594.0, u.day),
        argument_of_periastron=measured(40.0 * DEG, u.rad),
        periastron_convention=PeriastronConvention.PLANET,
    ).phase_at(EPOCH)

    assert solution.provenance is PhaseProvenance.TRANSIT_EPOCH
    assert solution.mapping is AnomalyMapping.CONJUNCTION_NORMALIZED
    assert solution.status is PhaseStatus.CONSTRAINED
    assert solution.omega_status is Status.MEASURED


def test_a_transit_epoch_without_omega_is_partially_constrained():
    """The Kepler-11 case, and the whole point of the refinement."""
    solution = _elements(epoch_transit=measured(2455594.0, u.day)).phase_at(EPOCH)

    assert solution.provenance is PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED
    assert solution.anchor is PhaseAnchor.OBSERVED  # the instant is real
    assert solution.mapping is AnomalyMapping.CONJUNCTION_NORMALIZED
    assert solution.status is PhaseStatus.PARTIALLY_CONSTRAINED
    assert solution.status.is_observationally_anchored
    assert "normalised" in solution.note


def test_a_transit_epoch_under_an_unstated_convention_is_also_partial():
    """AS_REPORTED omega leaves periapsis ambiguous by 180 degrees."""
    solution = _elements(
        epoch_transit=measured(2455594.0, u.day),
        argument_of_periastron=measured(40.0 * DEG, u.rad),
        periastron_convention=PeriastronConvention.AS_REPORTED,
    ).phase_at(EPOCH)

    assert solution.omega_status is Status.ASSUMED_FOR_VISUALIZATION
    assert solution.status is PhaseStatus.PARTIALLY_CONSTRAINED


def test_a_derived_omega_still_counts_as_constrained():
    """A stellar-reflex conversion is a documented derivation, not a guess."""
    solution = _elements(
        epoch_transit=measured(2455594.0, u.day),
        argument_of_periastron=measured(40.0 * DEG, u.rad),
        periastron_convention=PeriastronConvention.STELLAR_REFLEX,
    ).phase_at(EPOCH)

    assert solution.omega_status is Status.DERIVED
    assert solution.status is PhaseStatus.CONSTRAINED


def test_no_epoch_yields_no_phase_by_default():
    solution = _elements().phase_at(EPOCH)
    assert solution.mean_anomaly is None
    assert solution.provenance is PhaseProvenance.UNKNOWN
    assert solution.status is PhaseStatus.UNKNOWN
    assert not solution.is_placeable


def test_no_epoch_yields_an_assumed_phase_only_when_asked():
    solution = _elements().phase_at(EPOCH, allow_assumed=True)
    assert solution.is_placeable
    assert solution.provenance is PhaseProvenance.ASSUMED_ZERO_PHASE
    assert solution.anchor is PhaseAnchor.ASSUMED
    assert solution.mapping is AnomalyMapping.ARBITRARY_ZERO
    assert solution.status is PhaseStatus.ASSUMED
    assert not solution.status.is_observationally_anchored


def test_no_period_means_no_phase_at_all():
    solution = OrbitalElements(
        semimajor_axis=measured(1.0, u.au), eccentricity=measured(0.0)
    ).phase_at(EPOCH, allow_assumed=True)
    assert solution.mean_anomaly is None
    assert solution.anchor is PhaseAnchor.NONE
    assert "period" in solution.note


# ==========================================================================
# The mapping is still numerically right
# ==========================================================================


def test_at_the_transit_time_the_planet_is_at_conjunction():
    """nu = pi/2 - omega, so with omega := 0 the planet sits at nu = pi/2."""
    from astro_explorer.physics.kepler import solve_kepler, true_anomaly_from_eccentric

    elements = _elements(epoch_transit=measured(2455594.0, u.day))
    solution = elements.phase_at(2455594.0)

    eccentricity = elements.eccentricity.value
    nu = true_anomaly_from_eccentric(
        solve_kepler(solution.mean_anomaly, eccentricity), eccentricity
    )
    assert np.degrees(nu) == pytest.approx(90.0, abs=1e-9)


def test_a_published_omega_shifts_conjunction_by_exactly_that_angle():
    from astro_explorer.physics.kepler import solve_kepler, true_anomaly_from_eccentric

    omega = 40.0 * DEG
    elements = _elements(
        epoch_transit=measured(2455594.0, u.day),
        argument_of_periastron=measured(omega, u.rad),
        periastron_convention=PeriastronConvention.PLANET,
    )
    solution = elements.phase_at(2455594.0)

    eccentricity = elements.eccentricity.value
    nu = true_anomaly_from_eccentric(
        solve_kepler(solution.mean_anomaly, eccentricity), eccentricity
    )
    assert nu == pytest.approx(0.5 * np.pi - omega, abs=1e-9)


def test_the_phase_advances_at_the_orbital_rate():
    elements = _elements(epoch_periastron=measured(2450000.0, u.day))
    period = elements.period.value_in(u.day)

    first = elements.phase_at(EPOCH).mean_anomaly
    later = elements.phase_at(EPOCH + period).mean_anomaly
    assert np.mod(later - first + np.pi, 2 * np.pi) - np.pi == pytest.approx(0.0, abs=1e-9)


# ==========================================================================
# The conjunction caveat is stated rather than buried (review section 8)
# ==========================================================================


def test_the_conjunction_offset_vanishes_for_a_circular_orbit():
    assert conjunction_offset_scale(0.0, 60 * DEG) == 0.0


def test_the_conjunction_offset_vanishes_for_an_edge_on_orbit():
    assert conjunction_offset_scale(0.5, 90 * DEG) == pytest.approx(0.0, abs=1e-30)


def test_the_conjunction_offset_grows_away_from_edge_on():
    near = conjunction_offset_scale(0.3, 89 * DEG)
    far = conjunction_offset_scale(0.3, 45 * DEG)
    assert far > near


def test_the_offset_is_negligible_for_a_transiting_planet():
    """Kepler-11 d: e = 0.004, i = 89.6 deg."""
    assert conjunction_offset_scale(0.004, 89.6 * DEG) < 1e-6


def test_the_solution_reports_the_offset_when_it_matters():
    solution = _elements(
        eccentricity=measured(0.3),
        inclination=measured(45 * DEG, u.rad),
        epoch_transit=measured(2455594.0, u.day),
    ).phase_at(EPOCH)
    assert solution.conjunction_offset > 0.0
    assert "conjunction" in "\n".join(solution.describe())


# ==========================================================================
# Reporting
# ==========================================================================


def test_the_solution_describes_every_field():
    text = "\n".join(
        _elements(epoch_transit=measured(2455594.0, u.day)).phase_at(EPOCH).describe()
    )
    assert "TRANSIT_CONJUNCTION_NORMALIZED" in text
    assert "OBSERVED" in text
    assert "CONJUNCTION_NORMALIZED" in text
    assert "PARTIALLY_CONSTRAINED" in text
    assert "normalised to 0 deg" in text


def test_the_solution_serialises():
    record = _elements(epoch_periastron=measured(2450000.0, u.day)).phase_at(EPOCH).as_dict()
    assert record["provenance"] == "PERIASTRON_EPOCH"
    assert record["phase_status"] == "CONSTRAINED"
    assert record["anchor"] == "OBSERVED"


def test_every_provenance_value_maps_to_an_anchor_and_a_mapping():
    for provenance in PhaseProvenance:
        assert isinstance(provenance.anchor, PhaseAnchor)
        assert isinstance(provenance.mapping, AnomalyMapping)
        assert provenance.label


def test_an_empty_solution_is_unknown():
    assert PhaseSolution(None).status is PhaseStatus.UNKNOWN
    assert not PhaseSolution(None).is_placeable


# ==========================================================================
# Review section 11: the missing-string boundary utility
# ==========================================================================


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), "", "   ", "nan", "NaN", "NULL", "none", "N/A", "--", "<NA>"],
)
def test_missing_values_collapse_to_empty(value):
    assert clean_text(value) == ""


def test_numpy_and_pandas_sentinels_collapse_too():
    numpy = pytest.importorskip("numpy")
    pandas = pytest.importorskip("pandas")
    assert clean_text(numpy.float64("nan")) == ""
    assert clean_text(pandas.NA) == ""
    assert clean_text(pandas.NaT) == ""


def test_real_text_survives_and_is_stripped():
    assert clean_text("  G8 V  ") == "G8 V"
    assert clean_text("M8V") == "M8V"


def test_a_value_containing_nan_as_a_substring_survives():
    """'Nancy' must not be mistaken for a null."""
    assert clean_text("Nancy et al. 2020") == "Nancy et al. 2020"


def test_display_text_supplies_a_placeholder():
    assert display_text(float("nan")) == "unknown"
    assert display_text(None, fallback="-") == "-"
    assert display_text("G8 V") == "G8 V"


def test_the_null_spellings_are_lowercase():
    """They are compared against a lowercased string."""
    assert all(spelling == spelling.lower() for spelling in NULL_SPELLINGS)


def test_the_word_nan_never_reaches_a_record():
    """The bug this utility exists to prevent."""
    from astro_explorer.data.schema import build_planet_record

    record = build_planet_record(
        {
            "pl_name": "T b",
            "hostname": "T",
            "st_spectype": float("nan"),
            "discoverymethod": float("nan"),
            "disc_facility": None,
        }
    )
    assert record.host.spectral_type == ""
    assert record.discovery_method == ""
    assert "nan" not in "\n".join(record.describe()).lower()
