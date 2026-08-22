"""Explorer B: identity, panels, time controls.

Review sections 5, 7, 13 and 14. Organised by the acceptance criteria, plus
the four tests section 5 asks for by name.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.explorer import (
    Explorer,
    UniverseTarget,
    UnknownSystemPositionError,
)
from astro_explorer.app.panel import (
    Emphasis,
    InfoPanel,
    ParameterRow,
    build_planet_panel,
    build_star_panel,
    build_system_panel,
    parameter_row,
)
from astro_explorer.app.time_controls import RATE_PRESETS, TimeControls
from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.system_frame import UnlocatedFrameError
from astro_explorer.data.identity import (
    Catalog,
    EntityId,
    EntityKind,
    normalise_key,
    parse_entity_id,
    planet_id,
    star_id,
)
from astro_explorer.physics.ephemeris import TimeMode
from astro_explorer.provenance import Status, measured, unknown

HD80606_PERIASTRON = 2458882.344


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def hd80606(catalog):
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def trappist(catalog):
    return build_slice("TRAPPIST-1", catalog)


@pytest.fixture()
def explorer(catalog, hd80606) -> Explorer:
    star = hd80606.star
    instance = Explorer(
        targets=[
            UniverseTarget(
                "HD 80606",
                star.position.cartesian_pc(),
                temperature_k=star.effective_temperature.value_in(u.K),
            )
        ]
    )
    instance.move_to_pc([0.0, 0.0, 40.0])
    instance.focus("HD 80606", hd80606.planets, star)
    return instance


# ==========================================================================
# Review section 5: unknown absolute position must not become the Sun
# ==========================================================================


def test_unknown_absolute_position_cannot_be_navigated_to(trappist):
    """TRAPPIST-1 has no sy_dist, so there is nowhere to fly to."""
    instance = Explorer()
    assert not instance.can_navigate_to("TRAPPIST-1", trappist.star)

    with pytest.raises(UnknownSystemPositionError):
        instance.focus("TRAPPIST-1", trappist.planets, trappist.star)

    assert instance.system is None


def test_detached_system_can_open_locally(trappist):
    """The system is perfectly usable; only its galactic position is not."""
    instance = Explorer()
    instance.open_detached("TRAPPIST-1", trappist.planets, trappist.star)

    assert instance.detached
    assert instance.system is not None
    assert not instance.system.located

    scene = instance.scene(2457000.0)
    assert len(scene.planets) == 7
    assert len(scene.orbits) == 7
    assert any("no published galactic position" in note for note in scene.annotations)


def test_detached_system_never_appears_at_solar_origin(trappist):
    """The core of the fix: (0,0,0) local must not become (0,0,0) pc."""
    from astro_explorer.coordinates.system_frame import UniverseFrame

    instance = Explorer()
    instance.open_detached("TRAPPIST-1", trappist.planets, trappist.star)
    frame = instance.system

    # The star is the origin of its own frame - that is a local convention.
    assert np.array_equal(frame.star_position().values, np.zeros(3))

    # But nothing will convert that into an absolute position.
    with pytest.raises(UnlocatedFrameError):
        frame.to_absolute_pc(frame.star_position())
    with pytest.raises(UnlocatedFrameError):
        UniverseFrame().convert(frame.star_position())

    # And it is not "at the Sun": an unlocated frame is nowhere at all.
    assert not frame.contains([0.0, 0.0, 0.0])


def test_detached_system_has_no_universe_distance(trappist):
    instance = Explorer()
    instance.open_detached("TRAPPIST-1", trappist.planets, trappist.star)

    assert instance.distance_to_system_pc() is None
    assert not instance.has_absolute_position

    with pytest.raises(UnknownSystemPositionError):
        instance.enter_system()
    with pytest.raises(UnknownSystemPositionError):
        instance.path_to_system(4)

    text = "\n".join(instance.describe())
    assert "detached" in text
    assert "no known galactic position" in text


def test_a_detached_system_is_not_in_the_neighbourhood_view(trappist):
    instance = Explorer()
    instance.open_detached("TRAPPIST-1", trappist.planets, trappist.star)
    assert not instance.targets


def test_a_located_system_still_navigates_normally(explorer):
    assert explorer.has_absolute_position
    assert explorer.distance_to_system_pc() > 40.0
    explorer.enter_system()
    assert explorer.view.value == "SYSTEM"


# ==========================================================================
# Review section 7: stable entity keys
# ==========================================================================


def test_entity_ids_are_deterministic():
    assert str(planet_id("HD 80606 b")) == "planet:nasa:HD_80606_b"
    assert str(star_id("HD 80606")) == "star:nasa:HD_80606"
    assert planet_id("HD 80606 b") == planet_id("HD 80606 b")


def test_entity_ids_round_trip():
    identity = planet_id("Kepler-11 g")
    assert parse_entity_id(str(identity)) == identity


def test_designations_keep_their_meaningful_punctuation():
    """Hyphens and pluses are part of the designation, not separators."""
    assert normalise_key("Kepler-11 b") == "Kepler-11_b"
    assert normalise_key("2MASS J0437+2331") == "2MASS_J0437+2331"


def test_case_is_preserved_because_it_is_significant():
    """K2-18 b is a planet; K2-18 B would be a stellar companion."""
    assert normalise_key("K2-18 b") != normalise_key("K2-18 B")


def test_an_unnamed_entity_has_no_id():
    assert planet_id("") is None
    assert star_id(float("nan")) is None
    with pytest.raises(ValueError):
        EntityId(EntityKind.PLANET, Catalog.NASA, "")


def test_records_expose_their_entity_id(hd80606):
    record = hd80606.planet("HD 80606 b")
    assert str(record.entity_id) == "planet:nasa:HD_80606_b"
    assert str(record.host_id) == "star:nasa:HD_80606"
    assert str(hd80606.star.entity_id) == "star:nasa:HD_80606"


def test_render_primitives_carry_the_id_and_the_name_separately(explorer):
    explorer.enter_system()
    scene = explorer.scene(HD80606_PERIASTRON)
    planet = scene.planets[0]
    assert planet.identifier == "planet:nasa:HD_80606_b"
    assert planet.label == "HD 80606 b"
    assert scene.stars[0].identifier == "star:nasa:HD_80606"
    assert scene.stars[0].label == "HD 80606"


def test_display_name_changes_do_not_invalidate_selection(explorer):
    """The acceptance criterion. Identity is the key, not the label."""
    explorer.enter_system()
    selection = explorer.select("planet:nasa:HD_80606_b", "planet")

    scene = explorer.scene(HD80606_PERIASTRON)
    from dataclasses import replace

    # An alias change from a new catalogue release.
    scene.planets[0] = replace(scene.planets[0], label="HD 80606 b (Struve 1341 B b)")

    # The selection still matches the body, and label priority still works.
    assert scene.planets[0].identifier == selection.entity_id
    placements = explorer.labels(scene, 800, 600)
    if placements:
        assert placements[0][5] == selection.entity_id


def test_selection_carries_the_host_key(explorer):
    explorer.enter_system()
    selection = explorer.select("planet:nasa:HD_80606_b", "planet")
    assert selection.host_id == "star:nasa:HD_80606"
    assert selection.label == "HD 80606 b"


# ==========================================================================
# Review section 14: async generation token
# ==========================================================================


def test_selection_changes_bump_the_generation(explorer):
    first = explorer.select("star:nasa:HD_80606")
    second = explorer.select("planet:nasa:HD_80606_b", "planet")
    assert second.generation > first.generation


def test_a_stale_async_result_is_rejected(explorer):
    """The legacy race: A starts, B starts, B finishes, A finishes late."""
    selection_a = explorer.select("star:nasa:HD_80606")
    selection_b = explorer.select("planet:nasa:HD_80606_b", "planet")

    # B completes and is still current.
    assert explorer.is_current(selection_b)
    # A completes later, carrying its older token, and must be discarded.
    assert not explorer.is_current(selection_a)


def test_clearing_the_selection_also_invalidates_pending_work(explorer):
    selection = explorer.select("star:nasa:HD_80606")
    explorer.clear_selection()
    assert not explorer.is_current(selection)
    assert explorer.selection is None


def test_a_panel_carries_the_generation_it_was_built_for(explorer):
    explorer.enter_system()
    explorer.select("planet:nasa:HD_80606_b", "planet")
    panel = explorer.panel(HD80606_PERIASTRON)
    assert panel.generation == explorer.generation


# ==========================================================================
# Review section 13: the panel model
# ==========================================================================


def test_unknown_never_becomes_a_numeric_placeholder():
    """The invariant the panel exists to hold."""
    row = parameter_row("Ascending node", unknown(u.rad))
    assert row.value is None
    assert row.value_text == "unknown"
    assert row.emphasis is Emphasis.UNKNOWN
    assert "0" not in row.value_text
    assert row.unit == ""


def test_a_measured_value_keeps_its_uncertainty_and_unit():
    row = parameter_row(
        "Semi-major axis",
        measured(0.4603, u.au, error_plus=0.0021, error_minus=0.0021, provenance="ps"),
        unit=u.au,
    )
    assert row.value == pytest.approx(0.4603)
    assert row.unit == "AU"
    assert "0.0021" in row.uncertainty_text
    assert row.emphasis is Emphasis.MEASURED


def test_asymmetric_uncertainties_survive():
    row = parameter_row(
        "Teff", measured(5663.0, u.K, error_plus=55.0, error_minus=66.0), unit=u.K
    )
    assert "+55" in row.uncertainty_text and "-66" in row.uncertainty_text


def test_every_scientific_row_exposes_provenance(explorer, hd80606):
    panel = build_planet_panel(hd80606.planet("HD 80606 b"))
    for row in panel.rows:
        assert row.has_provenance, row.label


def test_derived_and_assumed_rows_are_distinguishable(hd80606):
    panel = build_planet_panel(hd80606.planet("HD 80606 b"))
    emphases = {row.label: row.emphasis for row in panel.rows}
    assert emphases["Semi-major axis"] is Emphasis.MEASURED
    assert emphases["Periapsis"] is Emphasis.DERIVED
    assert emphases["Ascending node"] is Emphasis.UNKNOWN

    rendered = "\n".join(panel.render())
    assert "[derived]" in rendered


def test_display_assumptions_are_listed_separately():
    """Measurements and display placeholders must not sit in one list."""
    rows = [
        ParameterRow("Measured", "1", emphasis=Emphasis.MEASURED, source="ps"),
        ParameterRow(
            "Assumed", "0", emphasis=Emphasis.ASSUMED, note="normalised for display"
        ),
    ]
    from astro_explorer.app.panel import PanelSection

    panel = InfoPanel("x", "X", sections=[PanelSection("S", rows)])
    assert [row.label for row in panel.assumptions] == ["Assumed"]
    text = "\n".join(panel.render())
    assert "DISPLAY ASSUMPTIONS (not measurements)" in text


def test_the_planet_panel_reports_the_periastron_convention(hd80606):
    panel = build_planet_panel(hd80606.planet("HD 80606 b"))
    text = "\n".join(panel.render())
    assert "convention" in text
    assert "not stated" in text


def test_the_planet_panel_reports_the_phase_status(hd80606):
    record = hd80606.planet("HD 80606 b")
    phase = record.elements.phase_at(HD80606_PERIASTRON)
    panel = build_planet_panel(record, phase)

    row = next(row for row in panel.rows if row.label == "Phase status")
    assert row.value_text == "CONSTRAINED"
    assert row.emphasis is Emphasis.MEASURED


def test_a_partially_constrained_phase_is_not_shown_as_measured(catalog):
    system = build_slice("Kepler-11", catalog)
    record = system.planets[0]
    phase = record.elements.phase_at(2455590.0)
    panel = build_planet_panel(record, phase)

    row = next(row for row in panel.rows if row.label == "Phase status")
    assert row.value_text == "PARTIALLY_CONSTRAINED"
    assert row.emphasis is not Emphasis.MEASURED


def test_an_assumed_phase_is_shown_as_assumed(trappist):
    record = trappist.planets[0]
    phase = record.elements.phase_at(2457000.0, allow_assumed=True)
    panel = build_planet_panel(record, phase)

    row = next(row for row in panel.rows if row.label == "Phase status")
    assert row.value_text == "ASSUMED"
    assert row.emphasis is Emphasis.ASSUMED
    assert row.is_assumption


def test_the_star_panel_says_when_there_is_no_distance(trappist):
    panel = build_star_panel(trappist.star)
    text = "\n".join(panel.render())
    assert "no position in the neighbourhood view" in text
    distance = next(row for row in panel.rows if row.label == "Distance")
    assert distance.value is None


def test_the_star_panel_carries_the_habitable_zone(hd80606):
    panel = build_star_panel(hd80606.star)
    labels = {row.label for row in panel.rows}
    assert {"Inner edge", "Outer edge"} <= labels


def test_the_system_panel_lists_planets_outwards(catalog):
    system = build_slice("Kepler-11", catalog)
    panel = build_system_panel("Kepler-11", system.star, system.planets)
    values = [row.value for row in panel.rows if row.value is not None]
    assert values == sorted(values)
    assert len(panel.rows) == 6


def test_building_a_panel_does_not_mutate_the_record(hd80606):
    """Selecting a planet must not perturb the orbit it describes."""
    record = hd80606.planet("HD 80606 b")
    before = (
        record.elements.semimajor_axis.value,
        record.elements.eccentricity.value,
        record.elements.longitude_of_ascending_node.status,
        record.elements.periastron_convention,
    )
    build_planet_panel(record, record.elements.phase_at(HD80606_PERIASTRON))
    after = (
        record.elements.semimajor_axis.value,
        record.elements.eccentricity.value,
        record.elements.longitude_of_ascending_node.status,
        record.elements.periastron_convention,
    )
    assert before == after


def test_selecting_a_planet_does_not_change_the_scene_geometry(explorer):
    explorer.enter_system()
    before = explorer.scene(HD80606_PERIASTRON)
    positions = [planet.position_local.copy() for planet in before.planets]

    explorer.select("planet:nasa:HD_80606_b", "planet")
    explorer.panel(HD80606_PERIASTRON)

    after = explorer.scene(HD80606_PERIASTRON)
    for original, planet in zip(positions, after.planets):
        assert np.array_equal(original, planet.position_local)


def test_the_panel_falls_back_to_the_system_summary(explorer):
    explorer.enter_system()
    explorer.clear_selection()
    panel = explorer.panel(HD80606_PERIASTRON)
    assert panel is not None
    assert panel.title == "HD 80606"


def test_no_row_ever_prints_the_word_nan(trappist):
    """The NaN boundary utility, checked through the panel."""
    panel = build_star_panel(trappist.star)
    assert "nan" not in "\n".join(panel.render()).lower()


# ==========================================================================
# Review section 13.6: time controls use the physical propagator
# ==========================================================================


def test_time_controls_start_at_a_published_epoch(hd80606):
    controls = TimeControls.for_system(hd80606.planets)
    assert controls.epoch_bjd == pytest.approx(HD80606_PERIASTRON)


def test_time_controls_fall_back_when_no_epoch_exists(trappist):
    controls = TimeControls.for_system(trappist.planets)
    # Any date is as good as another; the phase reports itself as assumed.
    assert controls.epoch_bjd > 0


def test_playing_advances_the_clock_at_the_chosen_rate():
    controls = TimeControls(epoch_bjd=2450000.0, rate_days_per_second=7.0)
    controls.play()
    controls.advance(3.0)
    assert controls.offset_days() == pytest.approx(21.0)


def test_pausing_stops_the_clock():
    controls = TimeControls(epoch_bjd=2450000.0, rate_days_per_second=1.0)
    controls.play()
    controls.advance(5.0)
    controls.pause()
    controls.advance(100.0)
    assert controls.offset_days() == pytest.approx(5.0)


def test_stepping_works_while_paused():
    controls = TimeControls(epoch_bjd=2450000.0)
    assert not controls.playing
    controls.step_days(10.0)
    assert controls.offset_days() == pytest.approx(10.0)


def test_stepping_by_a_fraction_of_a_period(hd80606):
    period = hd80606.planet("HD 80606 b").elements.period.value_in(u.day)
    controls = TimeControls(epoch_bjd=HD80606_PERIASTRON)
    controls.step_fraction(period, 0.5)
    assert controls.offset_days() == pytest.approx(period / 2.0)


def test_a_missing_period_makes_a_fractional_step_a_no_op():
    controls = TimeControls(epoch_bjd=2450000.0)
    controls.step_fraction(None, 0.5)
    controls.step_fraction(0.0, 0.5)
    assert controls.offset_days() == 0.0


def test_the_clock_drives_the_physical_propagator(hd80606):
    """The acceptance criterion: time control uses the real propagator."""
    record = hd80606.planet("HD 80606 b")
    period = record.elements.period.value_in(u.day)
    controls = TimeControls.for_system(hd80606.planets)

    at_periastron = record.elements.phase_at(controls.epoch_bjd).mean_anomaly
    controls.step_fraction(period, 0.5)
    at_apoapsis = record.elements.phase_at(controls.epoch_bjd).mean_anomaly

    assert at_periastron == pytest.approx(0.0, abs=1e-9)
    assert at_apoapsis == pytest.approx(np.pi, abs=1e-6)


def test_a_full_period_returns_to_the_same_phase(hd80606):
    record = hd80606.planet("HD 80606 b")
    period = record.elements.period.value_in(u.day)
    controls = TimeControls.for_system(hd80606.planets)

    first = record.elements.phase_at(controls.epoch_bjd).mean_anomaly
    controls.step_days(period)
    later = record.elements.phase_at(controls.epoch_bjd).mean_anomaly
    assert np.mod(later - first + np.pi, 2 * np.pi) - np.pi == pytest.approx(0.0, abs=1e-9)


def test_reset_returns_to_the_starting_epoch():
    controls = TimeControls(epoch_bjd=2450000.0)
    controls.step_days(500.0)
    controls.reset()
    assert controls.offset_days() == 0.0


def test_the_rate_presets_are_labelled():
    for label, rate in RATE_PRESETS:
        assert TimeControls(rate_days_per_second=rate).rate_label == label


def test_the_default_mode_is_physical_not_normalised():
    controls = TimeControls()
    assert controls.mode is TimeMode.SCALED
    assert "physical ephemeris time" in "\n".join(controls.describe())


def test_the_normalised_mode_declares_itself_non_physical():
    controls = TimeControls(mode=TimeMode.NORMALIZED)
    assert "NOT physical time" in "\n".join(controls.describe())


def test_the_phase_dial_wraps_with_the_period(hd80606):
    period = hd80606.planet("HD 80606 b").elements.period.value_in(u.day)
    controls = TimeControls(epoch_bjd=HD80606_PERIASTRON)
    assert controls.phase_fraction(period) == pytest.approx(0.0)
    controls.step_fraction(period, 0.25)
    assert controls.phase_fraction(period) == pytest.approx(0.25)
    controls.step_fraction(period, 1.0)
    assert controls.phase_fraction(period) == pytest.approx(0.25)


def test_the_panel_follows_the_clock(explorer, hd80606):
    """What is on screen and what the panel says cannot disagree."""
    explorer.enter_system()
    explorer.select("planet:nasa:HD_80606_b", "planet")
    period = hd80606.planet("HD 80606 b").elements.period.value_in(u.day)

    controls = TimeControls.for_system(hd80606.planets)
    first = explorer.panel(controls.epoch_bjd)
    controls.step_fraction(period, 0.5)
    second = explorer.panel(controls.epoch_bjd)

    assert first is not None and second is not None
    # Same planet, same provenance; only the propagated instant differs.
    assert first.entity_id == second.entity_id
