"""Explorer C4a: the selected host on the Hertzsprung-Russell diagram.

C3.6 finished the coordinate-physics arc. C4a returns to the scientific
explorer experience and asks a question the 3D work has already answered
twice, in a new place:

    can a *plot* show a scientific result without owning it?

The plotting code for both stellar diagrams already existed and was already
correctly named - the original program's temperature-radius scatter is not
an HR diagram, and each has had its own function and title since. What did
not exist was a model between the stellar physics and the figure, and the
absence had three symptoms, all of them silent:

* the marker was drawn from ``luminosity.value_in(u.L_sun)``, so the
  :class:`~astro_explorer.provenance.Status` that says whether anyone
  measured it never reached the figure;
* "is this derived?" was decided by comparing ``status.value`` to the
  string ``"DERIVED"`` - a spelling test standing in for a type test;
* a star with no temperature, no luminosity, or a non-positive one was
  **omitted without a word**. Nothing appeared, and nothing said why, which
  reads as "not interesting" rather than "not measured".

So the tests here are in two halves, the same shape as C1 and C2:

* the placement is *right* - the axes follow the real HR convention, a
  decade of luminosity is a unit of log spacing, hotter is further left;
* the placement is *qualified* - measured and derived stay distinguishable,
  an unknown value produces no coordinate at all, and the plot computes
  none of it for itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from astro_explorer.app.panel import Emphasis, build_star_panel  # noqa: E402
from astro_explorer.app.vertical_slice import (  # noqa: E402
    build_slice,
    load_reference_catalog,
)
from astro_explorer.physics.hr_diagram import (  # noqa: E402
    LUMINOSITY_AXIS,
    LUMINOSITY_NOT_POSITIVE,
    LUMINOSITY_NOT_PUBLISHED,
    MAIN_SEQUENCE_GUIDE,
    MAIN_SEQUENCE_GUIDE_DISCLOSURE,
    MAIN_SEQUENCE_GUIDE_LABEL,
    TEFF_AXIS,
    TEFF_NOT_PUBLISHED,
    HRPlacement,
    hr_placement,
    hr_population,
)
from astro_explorer.physics.stellar import (  # noqa: E402
    DiagramKind,
    luminosity_from_radius_and_teff,
    luminosity_ratio_from_radius_and_teff,
)
from astro_explorer.provenance import (  # noqa: E402
    Status,
    assumed,
    measured,
    unknown,
)
from astro_explorer.ui.plots.hr import (  # noqa: E402
    POPULATION_STYLES,
    draw_hr_diagram,
    placement_for,
    population_arrays,
    population_for,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "astro_explorer"


def _code_only(path: Path) -> str:
    """Source with comments and string literals removed.

    This module *documents* the identity it no longer computes, so scanning
    raw text would flag the explanation as the offence - the same trick the
    architecture tests use.
    """
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def hd80606(catalog):
    """A measured Teff with a luminosity derived from radius and Teff."""
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def trappist1(catalog):
    """The coolest host in the snapshot - the far right of the diagram."""
    return build_slice("TRAPPIST-1", catalog)


@pytest.fixture
def axes():
    figure, ax = plt.subplots()
    yield ax
    plt.close(figure)


def _sun_like(teff=5772.0, luminosity=1.0) -> HRPlacement:
    return hr_placement(
        "probe",
        measured(teff, u.K, provenance="test"),
        measured(luminosity, u.L_sun, provenance="test"),
    )


# ==========================================================================
# The marker is the panel's numbers, not a second reading of them
# ==========================================================================


def test_the_marker_uses_the_same_parameter_objects_as_the_panel(hd80606):
    """Not "the same value" - the same objects.

    Equality of two floats would pass even if the plot had recomputed the
    luminosity by a different route and happened to agree today. Identity
    cannot: there is one parameter, and the panel and the figure both point
    at it.
    """
    record = hd80606.planets[0]
    star = record.host
    placement = placement_for(record)

    assert placement.effective_temperature is star.effective_temperature
    assert placement.luminosity is star.luminosity

    # The panel keeps the raw magnitude beside the rendered text precisely
    # so a plot can use the number behind the words. That number and the
    # axis coordinate have to be the same one.
    rows = {row.label: row for row in build_star_panel(star).rows}
    assert rows["Effective temp."].value == placement.teff_k
    assert rows["Luminosity"].value == placement.luminosity_solar

    # And the provenance agrees too, so the panel cannot call a value
    # derived while the marker presents it as an observation.
    assert rows["Luminosity"].emphasis is Emphasis.DERIVED
    assert placement.luminosity_is_derived


def test_the_plot_never_recalculates_stellar_physics():
    """One Stefan-Boltzmann identity, in one place.

    ``population_arrays`` used to carry its own ``R^2 (T/T_sun)^4``. That is
    the same relation written twice, and the plotting layer recomputing
    stellar physics is exactly what the golden rule forbids - two copies
    agree until one is edited.
    """
    code = _code_only(SRC / "ui" / "plots" / "hr.py")

    # The two ingredients of the identity: a fourth power of a temperature
    # ratio, and the solar reference temperature it is taken against. This
    # module may hold neither, and the docstring that explains why is
    # stripped first so the explanation is not read as the offence.
    assert "SOLAR_EFFECTIVE_TEMPERATURE" not in code
    assert "4.0" not in code

    # It does not even call the physics function any more: the population
    # tiering is a scientific judgement, so it moved to the physics layer
    # whole, and this module asks for a finished HRPopulation.
    assert "luminosity_ratio_from_radius_and_teff" not in code
    assert "hr_population" in code

    physics = _code_only(SRC / "physics" / "hr_diagram.py")
    assert "luminosity_ratio_from_radius_and_teff" in physics

    # It is not enough that the plot stopped computing: the function it now
    # calls has to be the one the records go through, so the figure and the
    # panel cannot report different luminosities for the same star.
    through_parameters = luminosity_from_radius_and_teff(
        measured(2.0, u.R_sun, provenance="test"),
        measured(7000.0, u.K, provenance="test"),
    )
    through_arrays = luminosity_ratio_from_radius_and_teff(2.0, 7000.0)
    assert through_parameters.value_in(u.L_sun) == pytest.approx(
        through_arrays, rel=1e-12
    )

    # And it really is the Stefan-Boltzmann relation, not a fit: L scales as
    # R^2 T^4, so doubling the radius quadruples it and doubling the
    # temperature multiplies it by sixteen.
    base = luminosity_ratio_from_radius_and_teff(1.0, 5772.0)
    assert luminosity_ratio_from_radius_and_teff(2.0, 5772.0) == pytest.approx(
        4.0 * base, rel=1e-12
    )
    assert luminosity_ratio_from_radius_and_teff(1.0, 11544.0) == pytest.approx(
        16.0 * base, rel=1e-12
    )


def test_the_population_derivation_matches_the_physics_function(catalog):
    """The bulk path derives through the same function, exactly.

    Every row the archive left without an ``st_lum`` is filled by the
    Stefan-Boltzmann derivation, and "filled by" has to mean bit-for-bit
    the value that function returns - not a number that happens to be
    close. A second implementation would agree to a few percent and drift.
    """
    teff, luminosity, radius = population_arrays(catalog)

    frame = catalog.drop_duplicates(subset=["hostname"])
    published = frame["st_lum"].to_numpy(dtype=float)
    was_derived = (
        ~np.isfinite(published) & np.isfinite(teff) & np.isfinite(radius)
    )
    assert np.any(was_derived), "the snapshot should contain derivable hosts"

    expected = luminosity_ratio_from_radius_and_teff(radius, teff)
    assert np.array_equal(
        luminosity[was_derived], expected[was_derived]
    ), "the population fill must be the physics function's own output"

    # Where the archive did publish one, it is used unchanged rather than
    # being overwritten by a derivation.
    from_archive = np.isfinite(published)
    if np.any(from_archive):
        assert np.allclose(
            luminosity[from_archive], np.power(10.0, published[from_archive])
        )


# ==========================================================================
# The background population keeps each point's provenance
# ==========================================================================


def test_the_population_keeps_per_point_luminosity_provenance(catalog):
    """Two different kinds of luminosity must not become one cloud.

    The archive publishes ``st_lum`` for some hosts and a radius and a
    temperature for the rest. Flattening both into one numeric array asserts
    that every dot on the diagram is the same kind of thing - which is
    exactly the claim the *selected* marker was making before C4a, made
    again for six thousand other stars.
    """
    population = population_for(catalog)

    assert len(population) > 0
    assert population.luminosity_status.shape == population.luminosity_solar.shape

    counts = population.counts()
    assert counts[Status.MEASURED] > 0, "the snapshot should contain published st_lum"
    assert counts[Status.DERIVED] > 0, "and hosts derived from radius and Teff"

    # The two masks partition the plottable points; nothing is both or neither.
    assert not np.any(population.published & population.derived)
    assert np.array_equal(
        population.published | population.derived, population.plottable
    )


def test_a_published_luminosity_is_marked_measured_and_used_unchanged(catalog):
    """``st_lum`` is ``log10(L/L_sun)``, converted once and not re-derived."""
    frame = catalog.drop_duplicates(subset=["hostname"])
    population = population_for(catalog)
    published = frame["st_lum"].to_numpy(dtype=float)

    for i in np.flatnonzero(np.isfinite(published)):
        assert population.luminosity_status[i] is Status.MEASURED
        assert population.luminosity_solar[i] == pytest.approx(
            10.0 ** published[i], rel=1e-12
        )


def test_a_derived_population_luminosity_is_marked_derived(catalog):
    """And is bit-for-bit the shared Stefan-Boltzmann function's output."""
    frame = catalog.drop_duplicates(subset=["hostname"])
    population = population_for(catalog)
    published = frame["st_lum"].to_numpy(dtype=float)

    derivable = np.flatnonzero(
        ~np.isfinite(published)
        & np.isfinite(population.effective_temperature_k)
        & np.isfinite(population.radius_solar)
    )
    assert derivable.size

    for i in derivable:
        assert population.luminosity_status[i] is Status.DERIVED
        assert population.luminosity_solar[i] == luminosity_ratio_from_radius_and_teff(
            population.radius_solar[i], population.effective_temperature_k[i]
        )


def test_a_host_with_neither_source_is_unknown_and_not_drawn():
    """No luminosity at all is UNKNOWN, and UNKNOWN is not a coordinate."""
    population = hr_population(
        np.array(["published", "derivable", "blind"], dtype=object),
        np.array([5772.0, 5772.0, np.nan]),
        np.array([np.nan, 1.0, np.nan]),
        np.array([0.0, np.nan, np.nan]),
    )

    assert list(population.luminosity_status) == [
        Status.MEASURED,
        Status.DERIVED,
        Status.UNKNOWN,
    ]
    assert list(population.plottable) == [True, True, False]
    assert population.counts() == {Status.MEASURED: 1, Status.DERIVED: 1}


def test_the_two_provenances_are_drawn_with_different_markers(axes, catalog):
    """Not colour alone.

    Colour is the first thing lost to a greyscale print, a projector or a
    colour-blind reader, and the distinction being carried is scientific
    rather than decorative - the same reason the orientation overlay uses a
    dash pattern instead of a hue.
    """
    draw_hr_diagram(axes, catalog)

    markers = {style["marker"] for style in POPULATION_STYLES.values()}
    assert len(markers) == len(POPULATION_STYLES), "each provenance needs its own marker"

    colours = {style.get("color", "gray") for style in POPULATION_STYLES.values()}
    assert len(colours) == 1, "the distinction must not rest on colour"

    labels = axes.get_legend_handles_labels()[1]
    assert any("published luminosity" in text for text in labels)
    assert any("derived luminosity" in text for text in labels)


def test_the_population_arrays_helper_says_it_drops_the_provenance():
    """It is kept for callers that only want numbers, and it says so."""
    source = (SRC / "ui" / "plots" / "hr.py").read_text(encoding="utf-8")
    marker = source.index("def population_arrays")
    docstring = source[marker : marker + 900]
    assert "provenance" in docstring
    assert "population_for" in docstring


# ==========================================================================
# The main-sequence guide is illustrative, and says so
# ==========================================================================


def test_the_main_sequence_guide_does_not_pose_as_catalogue_data():
    """"Main sequence (reference)" invites "reference to what?".

    The points are hand-entered textbook-scale values with no citation,
    which is fine as context and not fine as an unlabelled series next to
    six thousand catalogue points. The label answers the question instead of
    raising it.
    """
    assert "Illustrative" in MAIN_SEQUENCE_GUIDE_LABEL
    assert "not catalogue data" in MAIN_SEQUENCE_GUIDE_LABEL
    assert MAIN_SEQUENCE_GUIDE_LABEL != "Main sequence (reference)"

    assert "no catalogue provenance" in MAIN_SEQUENCE_GUIDE_DISCLOSURE
    assert "no citation" in MAIN_SEQUENCE_GUIDE_DISCLOSURE


def test_the_guide_is_not_fitted_from_the_exoplanet_host_sample(catalog):
    """That sample is selection-biased: it is the stars people surveyed.

    A "main sequence" drawn through it would be a property of the survey
    rather than of stellar structure, so the guide is independent of the
    catalogue entirely - it does not change when the catalogue does.
    """
    assert "selection-biased" in MAIN_SEQUENCE_GUIDE_DISCLOSURE

    # It takes no catalogue input at all: it is a module constant.
    assert isinstance(MAIN_SEQUENCE_GUIDE, tuple)
    assert all(len(point) == 3 for point in MAIN_SEQUENCE_GUIDE)

    # And it is monotonic in temperature, so it reads as a sequence rather
    # than as a scatter of unrelated points.
    temperatures = [point[0] for point in MAIN_SEQUENCE_GUIDE]
    luminosities = [point[1] for point in MAIN_SEQUENCE_GUIDE]
    assert temperatures == sorted(temperatures)
    assert luminosities == sorted(luminosities)


def test_the_guide_is_drawn_dashed_and_labelled(axes, catalog):
    """Dashed for the same reason the overlay dashes a normalisation."""
    draw_hr_diagram(axes, catalog, show_main_sequence=True)

    labels = axes.get_legend_handles_labels()[1]
    assert MAIN_SEQUENCE_GUIDE_LABEL in labels
    assert not any("Main sequence (reference)" == text for text in labels)

    guide = [line for line in axes.lines if line.get_label() == MAIN_SEQUENCE_GUIDE_LABEL]
    assert guide, "the guide should be drawn"
    assert guide[0].get_linestyle() in ("--", "dashed")


def test_the_guide_can_be_turned_off(axes, catalog):
    draw_hr_diagram(axes, catalog, show_main_sequence=False)
    assert MAIN_SEQUENCE_GUIDE_LABEL not in axes.get_legend_handles_labels()[1]


def test_both_diagrams_share_one_guide_table():
    """One table, so the two plots cannot disagree about the same stars."""
    code = _code_only(SRC / "ui" / "plots" / "hr.py")
    assert "_MAIN_SEQUENCE" not in code
    assert code.count("MAIN_SEQUENCE_GUIDE") >= 2


# ==========================================================================
# The HR convention, which is old and backwards on purpose
# ==========================================================================


def test_hotter_stars_plot_further_left(axes, catalog, hd80606, trappist1):
    """The abscissa increases to the *left*: O B A F G K M.

    A plot with hot stars on the right is a temperature-luminosity scatter
    that looks like an HR diagram, which is the quiet substitution this
    pins. The check is on the axis limits, so it holds however the data
    happen to be distributed.
    """
    draw_hr_diagram(axes, catalog, hd80606.planets[0])
    left, right = axes.get_xlim()
    assert left > right, "the temperature axis must be inverted"

    hot = placement_for(hd80606.planets[0])
    cool = placement_for(trappist1.planets[0])
    assert hot.teff_k > cool.teff_k

    # "Further left" on an inverted axis means a larger data coordinate,
    # which is exactly what makes this easy to get backwards.
    def screen_x(placement):
        return axes.transData.transform((placement.teff_k, 1.0))[0]

    assert screen_x(hot) < screen_x(cool)


def test_one_decade_of_luminosity_is_one_unit_of_log_spacing(axes, catalog):
    """The ordinate is logarithmic, and the model exposes the log directly.

    The main sequence covers about eight decades, so a linear axis shows one
    star and a smear along the bottom. Checking the model rather than only
    the figure means the spacing claim survives a change of plotting
    backend.
    """
    draw_hr_diagram(axes, catalog)
    assert axes.get_yscale() == "log"

    decades = [_sun_like(luminosity=10.0**n) for n in (-3, -2, -1, 0, 1, 2)]
    logs = [p.log_luminosity for p in decades]
    steps = np.diff(logs)
    assert np.allclose(steps, 1.0, atol=1e-12)

    # And the figure agrees: equal decades are equal screen distances.
    heights = [axes.transData.transform((5772.0, p.luminosity_solar))[1] for p in decades]
    gaps = np.diff(heights)
    assert np.allclose(gaps, gaps[0], rtol=1e-6)


def test_the_axis_conventions_are_written_down():
    """Both of them, where someone about to change an axis will read it."""
    assert "left" in TEFF_AXIS
    assert "logarithmic" in LUMINOSITY_AXIS
    assert DiagramKind.HR_DIAGRAM.title == "Hertzsprung-Russell diagram"
    assert DiagramKind.HR_DIAGRAM.y_label == "Luminosity (L_sun)"
    # The other diagram is still a different plot with a different name.
    assert DiagramKind.TEMPERATURE_RADIUS.y_label == "Radius (R_sun)"
    assert DiagramKind.HR_DIAGRAM.title != DiagramKind.TEMPERATURE_RADIUS.title


def test_a_colour_magnitude_diagram_is_not_what_this_draws():
    """Magnitudes increase downward; luminosities do not.

    The two plots look alike and mean different things, so the ordinate says
    luminosity in solar units and the axis is not inverted.
    """
    assert "Luminosity" in DiagramKind.HR_DIAGRAM.y_label
    assert "magnitude" not in DiagramKind.HR_DIAGRAM.y_label.lower()

    brighter = _sun_like(luminosity=100.0)
    fainter = _sun_like(luminosity=0.01)
    assert brighter.log_luminosity > fainter.log_luminosity


# ==========================================================================
# An unknown value produces no coordinate at all
# ==========================================================================


def test_an_unknown_temperature_produces_no_x_coordinate():
    """No fake abscissa, and the reason travels."""
    placement = hr_placement(
        "probe", unknown(u.K), measured(1.0, u.L_sun, provenance="test")
    )

    assert not placement.is_plottable
    assert placement.teff_k is None
    assert TEFF_NOT_PUBLISHED in placement.blockers
    assert placement.status is Status.UNKNOWN

    # The luminosity it does have is still reported: one missing axis is not
    # two missing measurements.
    assert placement.luminosity_solar == 1.0
    assert any("effective temperature" in line for line in placement.describe())


def test_an_unknown_luminosity_produces_no_y_coordinate():
    """No fake ordinate either."""
    placement = hr_placement(
        "probe", measured(5772.0, u.K, provenance="test"), unknown(u.L_sun)
    )

    assert not placement.is_plottable
    assert placement.luminosity_solar is None
    assert placement.log_luminosity is None
    assert LUMINOSITY_NOT_PUBLISHED in placement.blockers
    assert placement.teff_k == 5772.0


def test_a_non_positive_luminosity_is_refused_rather_than_clamped():
    """It has no logarithm, and a small positive stand-in would be a star.

    A zero or negative luminosity is a bad catalogue row, not a very faint
    object. Clamping it onto the axis would invent a brightness.
    """
    for bad in (0.0, -0.4):
        placement = hr_placement(
            "probe",
            measured(5772.0, u.K, provenance="test"),
            measured(bad, u.L_sun, provenance="test"),
        )
        assert not placement.is_plottable
        assert LUMINOSITY_NOT_POSITIVE in placement.blockers
        assert placement.log_luminosity is None


def test_both_missing_values_are_reported_together():
    """A star missing two things should say two."""
    placement = hr_placement("probe", unknown(u.K), unknown(u.L_sun))
    assert TEFF_NOT_PUBLISHED in placement.blockers
    assert LUMINOSITY_NOT_PUBLISHED in placement.blockers
    assert len(placement.blockers) == 2


def test_an_unplaceable_star_is_annotated_rather_than_dropped(axes, catalog):
    """Silence reads as "not interesting" instead of "not measured"."""
    from dataclasses import replace

    record = build_slice("HD 80606", catalog).planets[0]
    blind = replace(record.host, effective_temperature=unknown(u.K))
    record = replace(record, host=blind)

    assert not placement_for(record).is_plottable
    draw_hr_diagram(axes, catalog, record)

    texts = " ".join(t.get_text() for t in axes.texts)
    assert "effective temperature" in texts


# ==========================================================================
# Measured, derived and assumed stay distinguishable
# ==========================================================================


def test_a_derived_luminosity_is_labelled_derived(hd80606):
    """Most hosts land here: the archive publishes a radius, not a luminosity."""
    placement = placement_for(hd80606.planets[0])

    assert placement.is_plottable
    assert placement.luminosity.status is Status.DERIVED
    assert placement.luminosity_is_derived
    assert "derived" in placement.label()

    # A typed question, not a string comparison against "DERIVED".
    assert isinstance(placement.luminosity.status, Status)


def test_a_published_luminosity_is_not_labelled_derived():
    placement = _sun_like()
    assert not placement.luminosity_is_derived
    assert "derived" not in placement.label()
    assert placement.is_scientific


def test_an_assumed_input_makes_the_whole_placement_assumed():
    """A marker drawn from an assumption sits at a perfectly definite height.

    Nothing about a dot says it was invented, so the placement carries the
    pessimistic status and the plot draws it hollow.
    """
    placement = hr_placement(
        "probe",
        measured(5772.0, u.K, provenance="test"),
        assumed(1.0, u.L_sun, provenance="display-normalisation"),
    )

    assert placement.is_plottable  # it has coordinates ...
    assert placement.status is Status.ASSUMED_FOR_VISUALIZATION
    assert not placement.is_scientific  # ... and may not be read as science
    assert "assumed" in placement.label()


def test_a_measured_pair_is_still_only_derived_as_a_placement():
    """The placement is computed from two measurements, so it is derived.

    Same rule the coordinate inspector uses, and now literally the same
    function - a second copy would eventually disagree about the same pair.
    """
    from astro_explorer.coordinates.inspector import _combined_status
    from astro_explorer.provenance import combined_status

    assert _combined_status is combined_status
    assert _sun_like().status is Status.DERIVED


# ==========================================================================
# Selection changes the marker and nothing else
# ==========================================================================


def test_selection_changes_the_marker_without_mutating_stellar_data(
    axes, catalog, hd80606, trappist1
):
    """Drawing is read-only, and drawing twice does not accumulate."""
    hot, cool = hd80606.planets[0], trappist1.planets[0]
    before = (
        hot.host.effective_temperature,
        hot.host.luminosity,
        cool.host.effective_temperature,
        cool.host.luminosity,
    )

    draw_hr_diagram(axes, catalog, hot)
    first = placement_for(hot)
    draw_hr_diagram(axes, catalog, cool)
    second = placement_for(cool)
    draw_hr_diagram(axes, catalog, hot)

    assert first.teff_k != second.teff_k
    assert (
        hot.host.effective_temperature,
        hot.host.luminosity,
        cool.host.effective_temperature,
        cool.host.luminosity,
    ) == before

    # The placement is a value, so re-deriving it gives an equal one.
    assert placement_for(hot) == first


def test_the_placement_holds_parameters_not_bare_floats(hd80606):
    """So a caller can ask it anything it could have asked the star."""
    placement = placement_for(hd80606.planets[0])

    assert placement.luminosity.error_plus is not None
    assert placement.luminosity.provenance
    assert placement.effective_temperature.unit == u.K
    assert placement.luminosity.unit == u.L_sun


# ==========================================================================
# The plot owns no science, and the runtime opens no socket
# ==========================================================================


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = path.relative_to(SRC).parts[:-1]
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ["astro_explorer", *package]
                trimmed = base[: len(base) - node.level + 1]
                modules.add(".".join(trimmed + ([node.module] if node.module else [])))
            elif node.module:
                modules.add(node.module)
    return modules


def test_the_plots_never_import_the_renderer():
    """A figure must not read a colour or an exaggerated radius.

    The renderer works in float32, scales planets so they are visible, and
    moves everything relative to the camera. Every one of those is correct
    for drawing and fatal for measuring - the same reason C3 forbade the
    inspector from reading the scene.
    """
    for path in sorted((SRC / "ui" / "plots").rglob("*.py")):
        for module in _imported_modules(path):
            assert "rendering" not in module, path.name


def test_the_hr_model_stays_in_the_physics_layer():
    """It may use provenance and astropy; it may not use data, ui or rendering."""
    modules = _imported_modules(SRC / "physics" / "hr_diagram.py")
    for module in modules:
        assert not module.startswith("astro_explorer.data"), module
        assert not module.startswith("astro_explorer.ui"), module
        assert not module.startswith("astro_explorer.rendering"), module


def test_the_hr_placement_performs_no_network_access(monkeypatch, catalog):
    """Not "does not need" - cannot."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the plot path must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    figure, ax = plt.subplots()
    try:
        for host in ("HD 80606", "TRAPPIST-1", "Kepler-11"):
            record = build_slice(host, catalog).planets[0]
            placement = placement_for(record)
            assert placement is not None
            draw_hr_diagram(ax, catalog, record)
    finally:
        plt.close(figure)


def test_every_snapshot_host_is_placeable_or_says_why(catalog):
    """The whole snapshot, swept, with no silent omissions."""
    for host in ("HD 80606", "HD 219134", "K2-18", "Kepler-11", "TRAPPIST-1", "WASP-39"):
        record = build_slice(host, catalog).planets[0]
        placement = placement_for(record)
        assert placement.is_plottable or placement.blockers, host
        if placement.is_plottable:
            assert placement.teff_k > 0.0, host
            assert placement.luminosity_solar > 0.0, host
            assert placement.log_luminosity is not None, host
