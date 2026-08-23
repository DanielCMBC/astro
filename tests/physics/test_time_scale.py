"""The time scale must reach the propagator, not stop at the metadata.

Explorer B review section 2 and the P0 items of its action list. The
project already had a good epoch model - ``BJD_TDB``, ``BKJD``, ``BTJD``
and the rest, with ``Epoch.as_bjd()`` knowing every mission offset - but
``TimeControls.for_system()`` read ``parameter.value_in(u.day)`` straight
off the element set, and ``OrbitalElements.phase_at()`` subtracted the raw
epoch parameter. The scale was recorded and then bypassed.

For a full Julian date that is harmless. For Kepler's BKJD it is wrong by
2454833 days, and for TESS's BTJD by 2457000: thirteen and seven years of
silent error in something that renders perfectly happily. These tests pin
the conversion in place at both ends of the path:

    catalogue epoch -> Epoch + TimeScale -> canonical full JD -> clock
                                                              -> phase_at()
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.time_controls import TimeControls
from astro_explorer.physics.epoch import (
    BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS,
    BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS,
    TDB_MINUS_UTC_FALLBACK_SECONDS,
    Epoch,
    EpochKind,
    MeanAnomalyAnchor,
    TimeScale,
    tdb_minus_utc_seconds,
)
from astro_explorer.physics.orbital_elements import PhaseKnowledge
from astro_explorer.physics.phase import PhaseProvenance, PhaseStatus
from astro_explorer.physics.orbital_elements import OrbitalElements
from astro_explorer.provenance import measured

PERIOD_DAYS = 10.0

#: Kepler's and TESS's mission offsets, quoted here rather than imported so
#: a change to the constants has to be argued for against a written number.
BKJD_OFFSET = 2454833.0
BTJD_OFFSET = 2457000.0


def _elements(scale: TimeScale, **overrides) -> OrbitalElements:
    values = dict(
        name="Test b",
        semimajor_axis=measured(0.1, u.au),
        eccentricity=measured(0.3),
        period=measured(PERIOD_DAYS, u.day),
        epoch_scale=scale,
    )
    values.update(overrides)
    return OrbitalElements(**values)


class _Record:
    """The bare shape ``TimeControls.for_system`` consumes."""

    def __init__(self, name: str, elements: OrbitalElements) -> None:
        self.name = name
        self.elements = elements


# ==========================================================================
# The mission offsets themselves
# ==========================================================================


def test_bkjd_epoch_gets_2454833_day_offset():
    """Kepler quotes BJD - 2454833; the clock must add it back."""
    elements = _elements(TimeScale.BKJD, epoch_periastron=measured(1000.0, u.day))
    controls = TimeControls.for_system([_Record("Test b", elements)])

    assert controls.epoch_jd == pytest.approx(1000.0 + BKJD_OFFSET)
    assert controls.source_scale is TimeScale.BKJD
    # And the raw number never reaches the clock.
    assert controls.epoch_jd != pytest.approx(1000.0)


def test_btjd_epoch_gets_2457000_day_offset():
    """TESS quotes BJD - 2457000."""
    elements = _elements(TimeScale.BTJD, epoch_transit=measured(1500.0, u.day))
    controls = TimeControls.for_system([_Record("Test b", elements)])

    assert controls.epoch_jd == pytest.approx(1500.0 + BTJD_OFFSET)
    assert controls.source_scale is TimeScale.BTJD


def test_a_stated_barycentric_epoch_is_not_shifted():
    """The offset is applied because the scale says so, not by default."""
    elements = _elements(TimeScale.BJD_TDB, epoch_periastron=measured(2458882.344, u.day))
    controls = TimeControls.for_system([_Record("Test b", elements)])

    assert controls.epoch_jd == pytest.approx(2458882.344)
    assert controls.scale_uncertainty_days == 0.0


# ==========================================================================
# What the offset cannot absorb travels with the clock
# ==========================================================================


def test_jd_unspecified_keeps_scale_uncertainty():
    """An unstated scale has a zero offset and a non-zero residual."""
    elements = _elements(
        TimeScale.JD_UNSPECIFIED, epoch_periastron=measured(2458882.344, u.day)
    )
    controls = TimeControls.for_system([_Record("Test b", elements)])

    assert controls.epoch_jd == pytest.approx(2458882.344)
    assert controls.source_scale is TimeScale.JD_UNSPECIFIED
    # About 568 s: the geocentric-barycentric light time plus leap seconds.
    assert controls.scale_uncertainty_days > 0.0
    assert controls.scale_uncertainty_days * 86400.0 == pytest.approx(568.184, abs=0.01)

    text = "\n".join(controls.describe())
    assert "Time-system error" in text
    assert "not" in text and "stated" in text


def test_unspecified_jd_is_not_labelled_exact_bjd_tdb():
    """Naming discipline: only a stated BJD_TDB epoch is displayed as one."""
    unstated = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(
                    TimeScale.JD_UNSPECIFIED,
                    epoch_periastron=measured(2458882.344, u.day),
                ),
            )
        ]
    )
    stated = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(TimeScale.BJD_TDB, epoch_periastron=measured(2458882.344, u.day)),
            )
        ]
    )

    assert "BJD_TDB" not in unstated.epoch_label
    assert unstated.epoch_label.startswith("JD ")
    assert stated.epoch_label.startswith("BJD_TDB ")

    # The field carrying the number is named for what it is, too.
    assert hasattr(unstated, "epoch_jd")
    assert not hasattr(unstated, "epoch_bjd")


def test_the_clock_records_which_epoch_it_started_from():
    """Review section 10: the starting instant is explained, not just shown."""
    elements = _elements(TimeScale.BKJD, epoch_periastron=measured(1000.0, u.day))
    controls = TimeControls.for_system([_Record("Test b", elements)])

    assert controls.source_kind is EpochKind.PERIASTRON
    assert controls.source_name == "Test b"
    assert "periastron" in "\n".join(controls.describe())


def test_a_system_with_no_epoch_says_the_date_is_arbitrary():
    controls = TimeControls.for_system([_Record("Test b", _elements(TimeScale.UNKNOWN))])

    assert controls.source_kind is EpochKind.UNKNOWN
    assert "arbitrary fallback date" in "\n".join(controls.describe())


# ==========================================================================
# The propagator must be on the same axis as the clock
# ==========================================================================


def test_phase_at_uses_same_epoch_scale_as_time_controls():
    """The two ends of the path must agree, or the picture lies.

    ``phase_at`` is evaluated at the clock's own date. If the clock applies
    the mission offset and the propagator does not, the planet is drawn
    2454833 days from where the clock says it is - which is exactly the
    bug this pins down.
    """
    elements = _elements(TimeScale.BKJD, epoch_periastron=measured(1000.0, u.day))
    controls = TimeControls.for_system([_Record("Test b", elements)])

    # The clock starts at a published periastron, so M = 0 there exactly.
    at_start = elements.phase_at(controls.epoch_jd).mean_anomaly
    assert at_start == pytest.approx(0.0, abs=1e-9)

    # Half a period later, half way round.
    half = elements.phase_at(controls.epoch_jd + PERIOD_DAYS / 2.0).mean_anomaly
    assert half == pytest.approx(np.pi, abs=1e-9)


def test_phase_at_applies_the_offset_for_a_transit_epoch_too():
    """Both anchors go through Epoch, not only the periastron branch.

    The period here is deliberately not a divisor of 2457000: with a round
    10-day period the mission offset happens to be an exact whole number of
    orbits, and the bug would hide behind the modulo.
    """
    elements = _elements(
        TimeScale.BTJD,
        epoch_transit=measured(1500.0, u.day),
        period=measured(111.436765, u.day),
    )

    canonical = 1500.0 + BTJD_OFFSET
    at_transit = elements.phase_at(canonical).mean_anomaly
    raw = elements.phase_at(1500.0).mean_anomaly

    assert at_transit is not None
    # Reading the raw number would have been a whole different phase.
    assert not np.isclose(at_transit, raw, atol=1e-6)


def test_phase_is_identical_before_and_after_offset_normalisation():
    """The review's equivalence check.

    BKJD 1000 and full JD 2455833 are the same instant. Two element sets
    that differ only in how that instant is written must give the same
    phase at the same physical time.
    """
    offset_scale = _elements(TimeScale.BKJD, epoch_periastron=measured(1000.0, u.day))
    full_jd = _elements(
        TimeScale.BJD_TDB, epoch_periastron=measured(1000.0 + BKJD_OFFSET, u.day)
    )

    assert offset_scale.epoch_of(EpochKind.PERIASTRON).canonical_jd == pytest.approx(
        full_jd.epoch_of(EpochKind.PERIASTRON).canonical_jd
    )

    # Phase at BKJD 1001 == phase at full JD 2455834, one day after each.
    instant = 1001.0 + BKJD_OFFSET
    assert offset_scale.phase_at(instant).mean_anomaly == pytest.approx(
        full_jd.phase_at(instant).mean_anomaly
    )

    # And over a whole period, sampled, not just at one point.
    for day in np.linspace(0.0, 3.0 * PERIOD_DAYS, 17):
        assert offset_scale.phase_at(instant + day).mean_anomaly == pytest.approx(
            full_jd.phase_at(instant + day).mean_anomaly
        )


def test_the_clocks_of_two_writings_of_one_epoch_agree():
    """Same instant, two conventions, one starting date on the clock."""
    bkjd = TimeControls.for_system(
        [_Record("Test b", _elements(TimeScale.BKJD, epoch_periastron=measured(1000.0, u.day)))]
    )
    full = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(
                    TimeScale.BJD_TDB,
                    epoch_periastron=measured(1000.0 + BKJD_OFFSET, u.day),
                ),
            )
        ]
    )
    assert bkjd.epoch_jd == pytest.approx(full.epoch_jd)


# ==========================================================================
# Review section 10: the starting-epoch selection policy
# ==========================================================================


def test_the_selected_planet_anchors_the_clock():
    """Rule 1: the planet the user is looking at wins."""
    inner = _Record(
        "Test b", _elements(TimeScale.BJD_TDB, epoch_periastron=measured(2450000.0, u.day))
    )
    outer = _Record(
        "Test c", _elements(TimeScale.BJD_TDB, epoch_periastron=measured(2455000.0, u.day))
    )

    assert TimeControls.for_system([inner, outer]).epoch_jd == pytest.approx(2450000.0)
    chosen = TimeControls.for_system([inner, outer], selected_name="Test c")
    assert chosen.epoch_jd == pytest.approx(2455000.0)
    assert chosen.source_name == "Test c"


def test_a_periastron_epoch_outranks_a_transit_epoch():
    """Rule 2: M = 0 directly beats reaching M through omega."""
    transiting = _Record(
        "Test b", _elements(TimeScale.BJD_TDB, epoch_transit=measured(2450000.0, u.day))
    )
    periastron = _Record(
        "Test c", _elements(TimeScale.BJD_TDB, epoch_periastron=measured(2455000.0, u.day))
    )

    controls = TimeControls.for_system([transiting, periastron])
    assert controls.source_kind is EpochKind.PERIASTRON
    assert controls.epoch_jd == pytest.approx(2455000.0)


def test_the_policy_is_deterministic_across_record_order():
    """Equal-ranked candidates fall back to record order, not dict order."""
    first = _Record(
        "Test b", _elements(TimeScale.BJD_TDB, epoch_transit=measured(2450000.0, u.day))
    )
    second = _Record(
        "Test c", _elements(TimeScale.BJD_TDB, epoch_transit=measured(2455000.0, u.day))
    )
    assert TimeControls.for_system([first, second]).source_name == "Test b"
    assert TimeControls.for_system([second, first]).source_name == "Test c"


def test_seeking_a_published_epoch_carries_its_scale():
    """The scale-aware counterpart of seek(); no bare BKJD number can land."""
    epoch = Epoch(measured(1000.0, u.day), EpochKind.TRANSIT, TimeScale.BKJD)
    controls = TimeControls(epoch_jd=2450000.0)

    controls.seek_epoch(epoch)
    assert controls.epoch_jd == pytest.approx(1000.0 + BKJD_OFFSET)
    assert controls.source_scale is TimeScale.BKJD
    assert controls.source_kind is EpochKind.TRANSIT
    assert controls.offset_days() == pytest.approx(0.0)


# ==========================================================================
# A mean anomaly at epoch is an angle, not a date
# ==========================================================================


def test_a_mean_anomaly_at_epoch_is_not_treated_as_a_julian_date():
    """It is quoted in radians; asking it for a JD is a category error.

    The angle no longer occupies an epoch slot at all - it lives in
    ``mean_anomaly_anchor`` - but the guard on :attr:`Epoch.is_dated` stays
    tested directly, because it is what makes the separation safe rather
    than merely tidy.
    """
    elements = _elements(
        TimeScale.BJD_TDB, mean_anomaly_at_epoch=measured(1.0, u.rad)
    )

    # An undated M0 contributes no epoch of any kind.
    assert elements.epoch_of(EpochKind.MEAN_ANOMALY_AT_EPOCH) is None
    assert elements.dated_epochs == []
    assert elements.mean_anomaly_anchor.is_known
    assert not elements.mean_anomaly_anchor.is_dated

    # And an angle forced into an Epoch is still refused a Julian date.
    miscast = Epoch(measured(1.0, u.rad), EpochKind.MEAN_ANOMALY_AT_EPOCH, TimeScale.BJD_TDB)
    assert not miscast.is_dated
    assert miscast.canonical_jd is None


def test_a_system_with_only_a_mean_anomaly_falls_back_to_the_fallback_date():
    """It cannot start the clock, so the clock says the date is arbitrary."""
    record = _Record(
        "Test b", _elements(TimeScale.BJD_TDB, mean_anomaly_at_epoch=measured(1.0, u.rad))
    )
    controls = TimeControls.for_system([record], fallback_jd=2451545.0)

    assert controls.epoch_jd == pytest.approx(2451545.0)
    assert controls.source_kind is EpochKind.UNKNOWN


# ==========================================================================
# Reference-frame uncertainty: the heliocentric and geocentric terms are
# not the same size and do not share a constant
# ==========================================================================


def test_hjd_utc_uncertainty_uses_solar_barycentre_scale():
    """An HJD owes the Sun's wobble, not the Earth's orbit.

    Heliocentric dates have already had the Earth's orbital light time
    removed. What remains is the Sun's own motion about the solar-system
    barycentre - up to ~1.6 million km, about five light-seconds - plus the
    leap seconds. Charging an HJD the full 8-minute geocentric term would
    overstate the error by nearly two orders of magnitude.
    """
    seconds = TimeScale.HJD_UTC.uncertainty_seconds

    assert seconds == pytest.approx(
        BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS + TDB_MINUS_UTC_FALLBACK_SECONDS
    )
    # Dominated by the leap seconds, not by light time.
    assert 70.0 < seconds < 90.0
    assert BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS < 20.0


def test_jd_utc_uncertainty_uses_earth_orbit_light_time_scale():
    """A plain JD is geocentric, so it owes the whole AU: ~499 s."""
    seconds = TimeScale.JD_UTC.uncertainty_seconds

    assert seconds == pytest.approx(
        BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS + TDB_MINUS_UTC_FALLBACK_SECONDS
    )
    # The famous "8.3 minutes" plus leap seconds.
    assert 550.0 < seconds < 580.0
    assert BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS == pytest.approx(499.0, abs=5.0)


def test_hjd_uncertainty_is_smaller_than_jd_uncertainty():
    """The ordering is the whole point of splitting the constants.

    Removing the heliocentric correction is most of the work; an HJD is far
    closer to a BJD than an unlabelled geocentric JD is. A single shared
    constant made these two equal, which said the correction bought nothing.
    """
    hjd = TimeScale.HJD_UTC.uncertainty_seconds
    jd = TimeScale.JD_UTC.uncertainty_seconds

    assert hjd < jd
    # Not marginally smaller: the frame terms differ by ~60x.
    assert jd > 5.0 * hjd
    assert TimeScale.BJD_TDB.uncertainty_seconds < hjd


def test_jd_unspecified_bound_covers_plain_jd_utc():
    """An unlabelled date might be a plain JD, so it must be bounded as one.

    The archive publishes no scale column. Assuming the milder heliocentric
    case would understate the error for exactly the reading that is most
    likely to be true of an unlabelled Julian day.
    """
    unspecified = TimeScale.JD_UNSPECIFIED.uncertainty_seconds

    assert unspecified >= TimeScale.JD_UTC.uncertainty_seconds
    assert unspecified >= TimeScale.HJD_UTC.uncertainty_seconds
    assert unspecified == pytest.approx(568.184)


def test_the_clock_carries_the_scale_specific_bound():
    """The split has to survive the trip to the clock, not just the enum."""
    hjd = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(
                    TimeScale.HJD_UTC, epoch_periastron=measured(2458882.344, u.day)
                ),
            )
        ]
    )
    jd = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(
                    TimeScale.JD_UTC, epoch_periastron=measured(2458882.344, u.day)
                ),
            )
        ]
    )

    # The leap-second term is evaluated at the epoch's own date, so these
    # carry the sub-millisecond TDB-TT wobble rather than a round literal.
    assert hjd.scale_uncertainty_days * 86400.0 == pytest.approx(77.184, abs=0.01)
    assert jd.scale_uncertainty_days * 86400.0 == pytest.approx(568.184, abs=0.01)
    assert hjd.scale_uncertainty_days < jd.scale_uncertainty_days


# ==========================================================================
# TDB - UTC is evaluated at the date, not frozen into the source
# ==========================================================================


def test_tdb_minus_utc_is_evaluated_at_the_date_not_hard_coded():
    """Leap seconds accumulate, so the offset is a function of when.

    A 1990 epoch owed twelve fewer leap seconds than a 2020 one. A single
    literal in the source would have charged both the same, and would go
    stale the next time IERS announces one.
    """
    in_1990 = tdb_minus_utc_seconds(2447892.5)
    in_2020 = tdb_minus_utc_seconds(2458882.344)

    assert in_1990 < in_2020
    assert in_2020 - in_1990 == pytest.approx(12.0, abs=0.01)
    assert in_2020 == pytest.approx(TDB_MINUS_UTC_FALLBACK_SECONDS, abs=0.01)


def test_tdb_minus_tt_periodic_term_is_not_assumed_zero():
    """The relativistic term is small, real, and should not be rounded away."""
    offset = tdb_minus_utc_seconds(2458882.344)
    residual = offset - TDB_MINUS_UTC_FALLBACK_SECONDS

    assert residual != 0.0
    # Milliseconds, not seconds: big enough to be real, small enough that
    # rounding it away would have looked harmless.
    assert 0.0 < abs(residual) < 0.005


def test_the_scale_offset_falls_back_when_there_is_no_date():
    """A scale with no epoch still has to state something defensible."""
    assert tdb_minus_utc_seconds(None) == TDB_MINUS_UTC_FALLBACK_SECONDS
    assert tdb_minus_utc_seconds(float("nan")) == TDB_MINUS_UTC_FALLBACK_SECONDS
    assert TimeScale.JD_UTC.uncertainty_seconds == pytest.approx(568.184)


def test_an_epoch_uses_its_own_date_for_the_leap_second_term():
    """The production path: the epoch knows when it is."""
    old = Epoch(measured(2447892.5, u.day), EpochKind.PERIASTRON, TimeScale.JD_UTC)
    recent = Epoch(measured(2458882.344, u.day), EpochKind.PERIASTRON, TimeScale.JD_UTC)

    assert old.scale_uncertainty_seconds < recent.scale_uncertainty_seconds
    assert recent.scale_uncertainty_seconds - old.scale_uncertainty_seconds == pytest.approx(
        12.0, abs=0.01
    )
    # The frame term is unchanged by the date; only the UTC term moves.
    assert old.scale_uncertainty_seconds > BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS


def test_a_stated_barycentric_epoch_owes_nothing_at_any_date():
    """No UTC term to evaluate, so the date cannot change the answer."""
    for jd in (2447892.5, 2458882.344, 2460676.5):
        epoch = Epoch(measured(jd, u.day), EpochKind.PERIASTRON, TimeScale.BJD_TDB)
        assert epoch.scale_uncertainty_seconds == 0.0


# ==========================================================================
# A mean anomaly needs the date it was quoted at
# ==========================================================================


def _anchor(m0_rad, t0_jd, scale=TimeScale.BJD_TDB):
    overrides = {}
    if m0_rad is not None:
        overrides["mean_anomaly_at_epoch"] = measured(m0_rad, u.rad)
    if t0_jd is not None:
        overrides["epoch_mean_anomaly"] = measured(t0_jd, u.day)
    return _elements(scale, **overrides)


def test_mean_anomaly_anchor_requires_a_dated_epoch_for_propagation():
    """M0 alone is a measurement; M0 with t0 is an ephemeris."""
    undated = _anchor(1.14, None).mean_anomaly_anchor
    dated = _anchor(1.14, 2458882.344).mean_anomaly_anchor

    assert undated.is_known
    assert not undated.is_dated
    assert undated.anomaly_rad == pytest.approx(1.14)
    assert undated.reference_jd is None
    assert undated.mean_anomaly_at(2459000.0, 2.0 * np.pi / PERIOD_DAYS) is None

    assert dated.is_dated
    assert dated.reference_jd == pytest.approx(2458882.344)
    assert dated.mean_anomaly_at(2459000.0, 2.0 * np.pi / PERIOD_DAYS) is not None


def test_undated_mean_anomaly_does_not_claim_current_position():
    """The old model returned M0 verbatim for every requested date.

    That is a real position at exactly one unpublished instant and wrong at
    all the others, while reporting itself as anchored by an observation.
    """
    elements = _anchor(1.14, None)

    assert elements.phase_knowledge is PhaseKnowledge.REFERENCE_ANOMALY_UNDATED
    assert not elements.can_compute_current_position

    solution = elements.phase_at(2459000.0)
    assert solution.mean_anomaly is None
    assert not solution.is_placeable
    assert solution.provenance is PhaseProvenance.MEAN_ANOMALY_UNDATED
    assert solution.status is PhaseStatus.UNKNOWN
    assert "epoch it refers to is not" in solution.note

    # The published angle is not thrown away just because it cannot propagate.
    assert elements.mean_anomaly_anchor.anomaly_rad == pytest.approx(1.14)

    # Asking two different dates must not produce the same claimed position.
    assert elements.phase_at(2459500.0).mean_anomaly is None


def test_dated_mean_anomaly_propagates_with_mean_motion():
    """M(t) = M0 + n(t - t0), and it actually moves."""
    m0, t0 = 1.14, 2458882.344
    elements = _anchor(m0, t0)
    n = 2.0 * np.pi / PERIOD_DAYS

    assert elements.can_compute_current_position
    assert elements.phase_knowledge is PhaseKnowledge.CURRENT_POSITION_COMPUTABLE

    at_epoch = elements.phase_at(t0)
    assert at_epoch.mean_anomaly == pytest.approx(m0)
    assert at_epoch.provenance is PhaseProvenance.MEAN_ANOMALY_AT_EPOCH

    quarter = elements.phase_at(t0 + 0.25 * PERIOD_DAYS)
    assert quarter.mean_anomaly == pytest.approx(
        np.mod(m0 + n * 0.25 * PERIOD_DAYS, 2.0 * np.pi)
    )
    assert abs(quarter.mean_anomaly - m0) > 1.0


def test_mean_anomaly_anchor_round_trips_one_period():
    """One full period returns to the same phase, in both directions."""
    m0, t0 = 1.14, 2458882.344
    elements = _anchor(m0, t0)
    reference = elements.phase_at(t0).mean_anomaly

    for turns in (-3, -1, 1, 2, 7):
        later = elements.phase_at(t0 + turns * PERIOD_DAYS).mean_anomaly
        assert later == pytest.approx(reference, abs=1e-9)

    # And a half period is genuinely on the other side of the orbit.
    half = elements.phase_at(t0 + 0.5 * PERIOD_DAYS).mean_anomaly
    assert abs(half - reference) == pytest.approx(np.pi, abs=1e-9)


def test_a_dated_mean_anomaly_epoch_can_anchor_the_clock():
    """It is a real instant, so it is allowed to start the clock - last."""
    anchored = TimeControls.for_system([_Record("Test b", _anchor(1.14, 2458882.344))])
    assert anchored.epoch_jd == pytest.approx(2458882.344)
    assert anchored.source_kind is EpochKind.MEAN_ANOMALY_AT_EPOCH

    # But a periastron epoch still outranks it.
    both = TimeControls.for_system(
        [
            _Record(
                "Test b",
                _elements(
                    TimeScale.BJD_TDB,
                    mean_anomaly_at_epoch=measured(1.14, u.rad),
                    epoch_mean_anomaly=measured(2458882.344, u.day),
                    epoch_periastron=measured(2459000.0, u.day),
                ),
            )
        ]
    )
    assert both.source_kind is EpochKind.PERIASTRON
    assert both.epoch_jd == pytest.approx(2459000.0)


def test_a_dated_mean_anomaly_survives_a_mission_offset():
    """The reference date goes through canonical_jd like any other epoch."""
    anchor = _anchor(1.14, 1000.0, scale=TimeScale.BKJD).mean_anomaly_anchor

    assert anchor.is_dated
    assert anchor.reference_jd == pytest.approx(1000.0 + BKJD_OFFSET)


def test_an_undated_anchor_describes_itself_honestly():
    dated = _anchor(1.14, 2458882.344).mean_anomaly_anchor
    undated = _anchor(1.14, None).mean_anomaly_anchor

    assert "reference epoch is not published" in undated.describe()
    assert "2458882" in dated.describe()
    assert "not published" in MeanAnomalyAnchor.missing().describe()


# ==========================================================================
# canonical_jd is the production name
# ==========================================================================


def test_no_new_production_callsite_uses_as_bjd():
    """``as_bjd()`` is a deprecated alias, not a route into the propagator.

    It overstates what the number is: only a stated ``BJD_TDB`` epoch is
    actually barycentric dynamical. The definition may keep the name; no
    other production module may call it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "astro_explorer"
    allowed = {root / "physics" / "epoch.py"}

    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path in allowed:
            continue
        if "as_bjd" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(root)))

    assert offenders == [], "use Epoch.canonical_jd instead of as_bjd() in {0}".format(
        ", ".join(offenders)
    )
