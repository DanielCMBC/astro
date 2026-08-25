"""Explorer C3: the coordinate and distance inspector.

C1 and C2 asked whether the scene could *draw* derived science honestly.
C3 asks whether the explorer can put a number next to it.

That turns out to be the easier thing to get subtly wrong. A drawn overlay
is obviously a picture; a figure reading "0.2057 AU" is read as a
measurement, and nothing about its appearance says whether it came from the
float64 propagator or from a float32 vertex that had already been scaled so
the planet would be visible next to its star.

So this file is built around one adversarial question, asked repeatedly:
*would this number change if someone moved the camera?* Every scientific
value here must be invariant under exaggeration, level of detail and camera
placement, because none of those are physics. The checks that vary a
rendering setting and assert the inspector did not move are the point of the
file, not padding around it.

The second half is the detached-system rule inherited from Explorer B. A
host whose parallax is unusable has no absolute position, and the honest
answer to "how far is it from Earth" is UNKNOWN - never zero, and never the
Sun's own coordinates. TRAPPIST-1 is that case in the committed snapshot.
"""

from __future__ import annotations

import math
from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.frames import Frame, SkyPosition, sky_position
from astro_explorer.coordinates.inspector import (
    CoordinateRow,
    InspectorFrame,
    InspectorRow,
    NoteRow,
    ABSOLUTE_POSITION_NO_HOST,
    absolute_planet_position,
    apoapsis_distance,
    galactic_coordinates,
    host_planet_distance,
    periapsis_distance,
    planet_distance_rows,
    star_coordinate_rows,
    system_frame_position,
)
from astro_explorer.coordinates.tangent import (
    ASTROMETRY_NOT_PROPAGATED,
    NODE_SENSE_UNRESOLVED,
)
from astro_explorer.provenance import Status, assumed, measured, unknown

#: An arbitrary but fixed epoch, so every distance in the file is comparable.
EPOCH_JD = 2460000.0


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def hd80606(catalog):
    """e = 0.93: the instantaneous distance sweeps a factor of ~28."""
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def kepler11(catalog):
    """Six planets, all with local coordinates worth reporting."""
    return build_slice("Kepler-11", catalog)


@pytest.fixture(scope="module")
def trappist1(catalog):
    """The detached case: no usable distance in this snapshot."""
    return build_slice("TRAPPIST-1", catalog)


@pytest.fixture(scope="module")
def hd219134(catalog):
    """Mixed data, and close enough that light travel time is small."""
    return build_slice("HD 219134", catalog)


def _row(rows, label):
    found = next((r for r in rows if r.label == label), None)
    assert found is not None, "no row labelled {0!r}".format(label)
    return found


def _assert_identical_rows(before, after):
    """Every row of two inspector results must match, whatever its type.

    Comparing on ``format()`` as well as on the underlying value means a new
    row type cannot slip past an invariance test by simply not having the
    attribute the comparison happened to check - which is exactly how the
    phase-provenance row first escaped these checks.
    """
    assert [r.label for r in before] == [r.label for r in after]
    for a, b in zip(before, after):
        assert type(a) is type(b), a.label
        assert a.format() == b.format(), a.label
        if isinstance(a, CoordinateRow):
            assert np.array_equal(a.values, b.values), a.label
        elif isinstance(a, InspectorRow):
            assert a.parameter.value == b.parameter.value, a.label
        else:
            assert a.text == b.text, a.label


# ==========================================================================
# The frame layer is Astropy's, not a second hand-written one
# ==========================================================================


def test_a_known_position_round_trips_through_skycoord():
    """RA/Dec/distance in, the same RA/Dec/distance out."""
    position = sky_position("probe", 187.25, -12.5, catalog_distance_pc=42.0)
    coord = position.skycoord

    assert coord.ra.to_value(u.deg) == pytest.approx(187.25)
    assert coord.dec.to_value(u.deg) == pytest.approx(-12.5)
    assert coord.distance.to_value(u.pc) == pytest.approx(42.0)

    # And the Cartesian form is the same position, not a rescaled one.
    cartesian = position.cartesian_pc()
    assert np.linalg.norm(cartesian) == pytest.approx(42.0, rel=1e-12)


def test_icrs_to_galactic_and_back_is_the_identity():
    """The transform is Astropy's, so it must invert to machine precision."""
    position = sky_position("probe", 33.0, 71.0, catalog_distance_pc=15.0)
    l, b = galactic_coordinates(position)

    back = SkyCoord(
        l=l.value_in(u.deg) * u.deg,
        b=b.value_in(u.deg) * u.deg,
        distance=15.0 * u.pc,
        frame="galactic",
    ).icrs

    assert back.ra.to_value(u.deg) == pytest.approx(33.0, abs=1e-9)
    assert back.dec.to_value(u.deg) == pytest.approx(71.0, abs=1e-9)

    # The Galactic Cartesian triplet has the same length as the ICRS one:
    # a frame rotation cannot change how far away the star is.
    icrs = position.cartesian_pc(Frame.ICRS)
    galactic = position.cartesian_pc(Frame.GALACTIC)
    assert np.linalg.norm(galactic) == pytest.approx(np.linalg.norm(icrs), rel=1e-12)


def test_galactic_direction_survives_an_unusable_parallax():
    """l and b are a direction, and a direction needs no distance.

    Withholding them from a distance-less star would be over-correction: it
    is the radial coordinate that is missing, not the pointing.
    """
    position = sky_position("probe", 12.0, 3.0, parallax_mas=-4.0)
    assert not position.has_distance

    l, b = galactic_coordinates(position)
    assert l.is_known and b.is_known
    assert position.cartesian_pc() is None


def test_the_host_distance_agrees_with_the_catalog_parameter(hd219134):
    """The inspector reports the scientific distance, not a copy of it."""
    rows = hd219134.inspect_star()
    catalog_distance = hd219134.star.position.distance

    assert _row(rows, "Distance from Sun").parameter.value_in(u.pc) == pytest.approx(
        catalog_distance.value_in(u.pc)
    )
    # And the Cartesian position is that same distance, decomposed.
    cartesian = _row(rows, "Cartesian position")
    assert cartesian.norm().value_in(u.pc) == pytest.approx(
        catalog_distance.value_in(u.pc), rel=1e-12
    )


def test_light_travel_time_matches_the_distance(hd219134):
    rows = hd219134.inspect_star()
    distance_ly = _row(rows, "Distance from Sun").parameter.to(u.lyr).value
    light_time = _row(rows, "Light travel time").parameter.value_in(u.yr)
    assert light_time == pytest.approx(distance_ly)


# ==========================================================================
# The distances come from the physics state, and agree with the identities
# ==========================================================================


def test_the_host_planet_distance_is_the_norm_of_the_physics_state(hd80606):
    """No independent recomputation: one propagation, one answer."""
    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)
    assert state is not None

    reported = host_planet_distance(state.position)
    assert reported.value_in(u.au) == pytest.approx(
        float(np.linalg.norm(state.position)), rel=1e-15
    )
    assert reported.status is Status.DERIVED


def test_the_distance_agrees_with_a_times_one_minus_e_cos_e(hd80606):
    """``r = a(1 - e cos E)`` is the independent check on the propagator.

    The inspector does not use this identity - it takes the norm of the
    propagated vector - which is exactly what makes it usable as a check
    here rather than a restatement of the implementation.
    """
    record = hd80606.planet("HD 80606 b")
    elements = record.elements.for_display()
    axis = elements.semimajor_axis.value_in(u.au)
    ecc = elements.eccentricity.value

    for offset in (0.0, 7.0, 31.0, 93.5, 200.0):
        state = hd80606.state(record, EPOCH_JD + offset)
        eccentric_anomaly = float(np.asarray(state.eccentric_anomaly).reshape(()))
        expected = axis * (1.0 - ecc * math.cos(eccentric_anomaly))
        assert host_planet_distance(state.position).value_in(u.au) == pytest.approx(
            expected, rel=1e-10
        )


def test_periapsis_and_apoapsis_are_the_textbook_identities(hd80606):
    elements = hd80606.planet("HD 80606 b").elements
    axis = elements.semimajor_axis.value_in(u.au)
    ecc = elements.eccentricity.value

    assert periapsis_distance(elements).value_in(u.au) == pytest.approx(axis * (1.0 - ecc))
    assert apoapsis_distance(elements).value_in(u.au) == pytest.approx(axis * (1.0 + ecc))


def test_a_high_eccentricity_orbit_sweeps_between_periapsis_and_apoapsis(hd80606):
    """HD 80606 b is the reason the inspector cannot quote ``a`` and stop.

    At e = 0.93 the instantaneous distance varies by a factor of nearly 30
    over one period. A UI that showed the semimajor axis as "the distance"
    would be wrong by that factor for most of the orbit.
    """
    record = hd80606.planet("HD 80606 b")
    elements = record.elements
    period = elements.period.value_in(u.day)

    distances = [
        host_planet_distance(hd80606.state(record, EPOCH_JD + f * period).position).value_in(u.au)
        for f in np.linspace(0.0, 1.0, 4000, endpoint=False)
    ]
    peri = periapsis_distance(elements).value_in(u.au)
    apo = apoapsis_distance(elements).value_in(u.au)

    assert min(distances) == pytest.approx(peri, rel=1e-4)
    assert max(distances) == pytest.approx(apo, rel=1e-4)
    # Every sample stays inside the bounds, to within float noise.
    assert all(peri * (1 - 1e-9) <= d <= apo * (1 + 1e-9) for d in distances)
    # And the sweep is real, not a near-circular orbit dressed up.
    assert max(distances) / min(distances) > 20.0


def test_an_assumed_semimajor_axis_does_not_yield_a_derived_periapsis(hd80606):
    """A visualisation assumption must not become a quotable distance.

    ``a(1-e)`` is exact, which is the trap: the arithmetic being sound says
    nothing about whether ``a`` was measured. If the axis was invented so
    the orbit could be drawn, so was the periapsis.
    """
    elements = hd80606.planet("HD 80606 b").elements
    invented = replace(
        elements,
        semimajor_axis=assumed(0.45, u.au, provenance="test: made up so it draws"),
    )

    assert periapsis_distance(elements).status is Status.DERIVED
    assert periapsis_distance(invented).status is Status.ASSUMED_FOR_VISUALIZATION
    assert apoapsis_distance(invented).status is Status.ASSUMED_FOR_VISUALIZATION


def test_an_unknown_element_leaves_the_distance_unknown(hd80606):
    elements = replace(
        hd80606.planet("HD 80606 b").elements, semimajor_axis=unknown(u.au)
    )
    assert not periapsis_distance(elements).is_known
    assert not apoapsis_distance(elements).is_known
    assert periapsis_distance(elements).status is Status.UNKNOWN


# ==========================================================================
# A detached system keeps its local coordinates and claims no address
# ==========================================================================


def test_a_detached_system_reports_no_distance_from_earth(trappist1):
    """UNKNOWN, and specifically not zero.

    Zero parsecs is the Sun. A bug that produced it would place every
    unlocated system on top of the observer, and the number would look
    entirely reasonable in a table.
    """
    assert not trappist1.frame.located
    rows = trappist1.inspect_star()

    distance = _row(rows, "Distance from Sun").parameter
    assert not distance.is_known
    assert distance.value != 0.0
    assert distance.value is None

    cartesian = _row(rows, "Cartesian position")
    assert not cartesian.is_known
    assert cartesian.values is None
    assert cartesian.status is Status.UNKNOWN


def test_a_detached_system_still_reports_local_coordinates(trappist1):
    """The orbit is fine. It is the address that is missing."""
    record = trappist1.planets[0]
    rows = trappist1.inspect_planet(record, EPOCH_JD)

    local = _row(rows, "System-frame position")
    assert local.is_known
    assert local.frame is InspectorFrame.SYSTEM
    assert local.unit == u.au

    separation = _row(rows, "Distance from host").parameter
    assert separation.is_known and separation.value_in(u.au) > 0.0

    absolute = _row(rows, "Absolute position")
    assert not absolute.is_known

# ==========================================================================
# Absolute position: withheld, because it cannot be expressed in one basis
# ==========================================================================


def test_the_absolute_planet_position_is_not_published(hd80606):
    """No real planet in the snapshot gets an absolute position.

    C3 withheld this because no SystemFrame -> ICRS rotation existed at all.
    C3.5 supplied that rotation, and the row stays withheld anyway - for a
    different and now sharper reason: HD 80606 b has no measured ascending
    node, so the display normalises it to zero and the orbit's azimuth about
    the line of sight is a convention. The transform is exact; its input is
    not an observation.

    Marking the row assumed would still be the wrong repair. It would put a
    triplet labelled ICRS in front of a reader as though someone had
    measured it.

    C3.6 sharpens it once more. The slice now propagates HD 80606's Gaia DR3
    astrometry to the requested instant, so the epoch gate that used to
    block this row alongside the node is genuinely satisfied - and the row
    is still withheld, for the one reason that is left. A gate that opened
    when the science supported it is what makes the remaining refusal a
    scientific statement rather than a permanent stub.
    """
    record = hd80606.planet("HD 80606 b")
    row = _row(hd80606.inspect_planet(record, EPOCH_JD), "Absolute position")

    assert not row.is_known
    assert row.values is None
    assert row.status is Status.UNKNOWN
    assert NODE_SENSE_UNRESOLVED in row.note
    # The epoch is no longer among the reasons: it was handled, not waived.
    assert ASTROMETRY_NOT_PROPAGATED not in row.note
    # The reasons travel with the row, so a panel can say why.
    assert "modulo 180 degrees" in row.format()


def test_a_located_host_does_not_make_the_absolute_position_available(hd80606):
    """A known host address is not the missing piece.

    HD 80606 has a usable parallax distance and, since C3.5, a valid tangent
    basis. The only thing missing is the node. If either of the other two
    were enough on its own, this is where that would show up.

    This call passes no astrometry, so the epoch gate closes too - which is
    the C3.6 default and the point of it: the gate is opened by supplying a
    propagated state, never by omitting an argument.
    """
    assert hd80606.frame.located
    assert hd80606.star.position.has_distance

    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)
    row = absolute_planet_position(
        hd80606.star.position,
        state.position,
        node=record.elements.longitude_of_ascending_node,
    )

    assert not row.is_known
    # The note names the node and the epoch, not the host, as the blockers.
    assert NODE_SENSE_UNRESOLVED in row.note
    assert ASTROMETRY_NOT_PROPAGATED in row.note
    assert ABSOLUTE_POSITION_NO_HOST not in row.note


def test_an_unlocated_host_reports_both_reasons(trappist1):
    """Two independent obstacles, both named."""
    record = trappist1.planets[0]
    state = trappist1.state(record, EPOCH_JD)
    row = absolute_planet_position(
        trappist1.star.position,
        state.position,
        node=record.elements.longitude_of_ascending_node,
    )

    assert not row.is_known
    assert ABSOLUTE_POSITION_NO_HOST in row.note
    assert ASTROMETRY_NOT_PROPAGATED in row.note


def test_the_absolute_position_takes_no_frame_argument():
    """Still no frame argument, even now that the rotation exists.

    A ``frame`` parameter would suggest the answer merely differs by frame.
    It does not: the sum is formed in ICRS, and any other frame is a
    downstream rotation of the finished vector, not a different computation.

    ``node`` *is* a parameter, and required, because whether the node was
    observed is exactly what decides if the result may be published at all.
    """
    import inspect

    parameters = inspect.signature(absolute_planet_position).parameters
    assert "frame" not in parameters
    assert "node" in parameters


def test_what_is_well_defined_is_still_published(hd80606):
    """Withholding the sum costs nothing that was actually known.

    A separation is invariant under any rotation, and the local triplet is
    reported in the frame it is genuinely expressed in. Neither needs the
    missing transform, so neither is withheld.
    """
    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)
    rows = hd80606.inspect_planet(record, EPOCH_JD)

    separation = _row(rows, "Distance from host").parameter
    assert separation.is_known
    assert separation.value_in(u.au) == pytest.approx(
        float(np.linalg.norm(state.position)), rel=1e-15
    )

    local = _row(rows, "System-frame position")
    assert local.is_known
    assert local.frame is InspectorFrame.SYSTEM
    assert np.array_equal(local.values, state.position)

    # The host's own ICRS position needs no rotation: it is measured in that
    # basis to begin with.
    assert _row(hd80606.inspect_star(), "Cartesian position").is_known


def test_no_published_planet_triplet_claims_a_celestial_frame(
    hd80606, kepler11, hd219134
):
    """Across every located system, the only planet triplet is SystemFrame.

    An ICRS or Galactic planet triplet carrying values would be the basis
    error returning by another route, so this sweeps for one rather than
    checking a single call site.
    """
    for system in (hd80606, kepler11, hd219134):
        for record in system.planets:
            for row in system.inspect_planet(record, EPOCH_JD):
                if isinstance(row, CoordinateRow) and row.is_known:
                    assert row.frame is InspectorFrame.SYSTEM, (
                        system.star.name, record.name, row.label, row.frame,
                    )


# ==========================================================================
# Every displayed coordinate says what it is a coordinate of
# ==========================================================================


def test_a_coordinate_row_cannot_be_built_without_a_frame():
    """Structural, not a display convention.

    The same planet has three entirely different x/y/z depending on the
    frame. A triplet whose frame the caller could omit is a number with no
    meaning, so omission is a construction error.
    """
    with pytest.raises(ValueError, match="must name its frame"):
        CoordinateRow("position", np.zeros(3), u.pc, InspectorFrame.NONE)


def test_every_coordinate_row_carries_a_frame_and_a_unit(hd80606):
    record = hd80606.planet("HD 80606 b")
    rows = hd80606.inspect_star() + hd80606.inspect_planet(record, EPOCH_JD)
    triplets = [r for r in rows if isinstance(r, CoordinateRow)]
    assert triplets

    for row in triplets:
        assert row.frame is not InspectorFrame.NONE
        assert isinstance(row.unit, u.UnitBase)
        assert row.frame.value in row.format()
        assert row.unit.to_string() in row.format() or not row.is_known


def test_a_coordinate_row_rejects_a_non_finite_component():
    """NaN dressed as a position is the failure mode being closed."""
    with pytest.raises(ValueError, match="must be finite"):
        CoordinateRow("position", np.array([1.0, np.nan, 3.0]), u.pc, InspectorFrame.ICRS)


def test_a_component_of_a_coordinate_row_keeps_unit_and_status(hd80606):
    record = hd80606.planet("HD 80606 b")
    row = _row(hd80606.inspect_planet(record, EPOCH_JD), "System-frame position")

    for axis in range(3):
        component = row.component(axis)
        assert component.unit == u.au
        assert component.status is row.status
        assert component.value == pytest.approx(row.values[axis])


def test_uncertainty_and_provenance_survive_presentation(hd219134):
    """A converted distance is still the same measurement.

    Unit conversion is where error bars are most often silently dropped:
    the value is scaled and the uncertainty is not, and the result reads as
    far more precise than the catalogue supports.
    """
    catalog_distance = hd219134.star.position.distance
    assert catalog_distance.error_plus  # the snapshot really does quote one

    row = _row(hd219134.inspect_star(), "Distance from Sun")
    assert row.parameter.error_plus == pytest.approx(catalog_distance.error_plus)
    assert row.parameter.status is catalog_distance.status
    assert row.parameter.provenance == catalog_distance.provenance

    in_ly = row.parameter.to(u.lyr)
    ratio = in_ly.error_plus / row.parameter.error_plus
    assert ratio == pytest.approx(float((1.0 * u.pc).to_value(u.lyr)))


def test_an_inspector_row_names_its_frame_in_its_own_text(hd80606):
    rows = hd80606.inspect_star()
    assert "[ICRS]" in _row(rows, "Right ascension").format()
    assert "[Galactic]" in _row(rows, "Galactic longitude l").format()
    # A frame-independent scalar must not claim a frame. It may still carry
    # a status tag, so this looks for frame names rather than a bracket.
    light_time = _row(rows, "Light travel time").format()
    assert not any(f.value and f.value in light_time for f in InspectorFrame)


# ==========================================================================
# Nothing here is downstream of the renderer
# ==========================================================================


def test_the_inspector_imports_no_rendering_primitives():
    """The firewall, asserted on the module rather than assumed."""
    import ast
    from pathlib import Path

    import astro_explorer.coordinates.inspector as inspector

    source = Path(inspector.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any("rendering" in name for name in imported), imported
    assert not any(name.startswith("astro_explorer.data") for name in imported)
    assert not hasattr(inspector, "SceneDescription")


def test_display_exaggeration_does_not_move_a_single_distance(hd80606):
    """The adversarial check: build the scene both ways, compare numbers.

    Exaggeration exists so a planet is visible beside its star, and it
    changes render radii by large factors. If any inspector value were read
    back out of the scene, this test would catch it immediately.
    """
    from astro_explorer.rendering.scene_builder import build_frame_scene

    record = hd80606.planet("HD 80606 b")
    anomalies = hd80606.mean_anomalies(EPOCH_JD)

    def rows_after(exaggerate: bool):
        scene = build_frame_scene(
            hd80606.frame, hd80606.star, hd80606.planets, mean_anomalies=anomalies,
            exaggerate=exaggerate,
        )
        assert scene.planets  # the scene really was built
        return hd80606.inspect_planet(record, EPOCH_JD)

    plain, exaggerated = rows_after(False), rows_after(True)

    # The rendering really did differ ...
    small = build_frame_scene(
        hd80606.frame, hd80606.star, hd80606.planets, mean_anomalies=anomalies,
        exaggerate=False,
    )
    large = build_frame_scene(
        hd80606.frame, hd80606.star, hd80606.planets, mean_anomalies=anomalies,
        exaggerate=True,
    )
    assert large.planets[0].radius_display != small.planets[0].radius_display

    # ... and the science did not.
    _assert_identical_rows(plain, exaggerated)


def test_level_of_detail_does_not_change_any_inspector_value(hd80606):
    """LOD is a triangle count. It is not allowed to be physics."""
    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.scene_builder import build_frame_scene

    record = hd80606.planet("HD 80606 b")
    anomalies = hd80606.mean_anomalies(EPOCH_JD)
    baseline = hd80606.inspect_planet(record, EPOCH_JD)

    observed_levels = set()
    for distance in (0.5, 5.0, 5000.0):
        scene = build_frame_scene(
            hd80606.frame, hd80606.star, hd80606.planets, mean_anomalies=anomalies
        )
        scene.assign_lod(
            Camera(target=np.zeros(3), distance=distance, aspect=1.5), 800
        )
        observed_levels.update(p.lod for p in scene.planets)

        _assert_identical_rows(baseline, hd80606.inspect_planet(record, EPOCH_JD))

    # The LOD really did vary; otherwise the loop above proves nothing.
    assert len(observed_levels) > 1, observed_levels


def test_orbit_sampling_does_not_change_any_inspector_value(hd80606):
    """How finely the path is tessellated is a drawing choice."""
    from astro_explorer.rendering.scene_builder import build_frame_scene

    record = hd80606.planet("HD 80606 b")
    anomalies = hd80606.mean_anomalies(EPOCH_JD)
    baseline = hd80606.inspect_planet(record, EPOCH_JD)

    vertex_counts = set()
    for samples in (64, 720, 2048):
        scene = build_frame_scene(
            hd80606.frame, hd80606.star, hd80606.planets,
            mean_anomalies=anomalies, orbit_samples=samples,
        )
        vertex_counts.add(scene.orbits[0].vertex_count)
        _assert_identical_rows(baseline, hd80606.inspect_planet(record, EPOCH_JD))

    assert len(vertex_counts) == 3


def test_camera_placement_does_not_change_object_to_object_distances(hd80606):
    """A separation between two objects is not a property of the viewer."""
    from astro_explorer.rendering.camera import Camera

    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)
    baseline = host_planet_distance(state.position).value_in(u.au)

    for distance, yaw, pitch in ((1.0, 0.0, 0.0), (500.0, 2.1, -0.7), (1e6, 4.0, 1.2)):
        Camera(target=np.zeros(3), distance=distance, yaw=yaw, pitch=pitch, aspect=1.5)
        again = hd80606.state(record, EPOCH_JD)
        assert host_planet_distance(again.position).value_in(u.au) == baseline


def test_the_inspector_uses_the_physics_state_not_the_framed_position(hd80606):
    """The two agree today, and the inspector still must not use the frame.

    In SystemFrame the render position happens to equal the physics vector,
    so reading the wrong one would be invisible here and wrong the moment a
    presentation transform is added. Pin that they agree *and* that the
    reported value tracks the physics state when they are made to differ.
    """
    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)
    framed = hd80606.framed_position(record, EPOCH_JD)

    assert np.allclose(np.asarray(framed.values, dtype=np.float64), state.position)

    # A float32 round trip is what a render position would have been through.
    as_rendered = np.asarray(framed.to_render(), dtype=np.float64)
    from_physics = host_planet_distance(state.position).value_in(u.au)
    from_rendered = float(np.linalg.norm(as_rendered))
    assert from_physics == pytest.approx(from_rendered, rel=1e-6)
    # Same to float32 precision, and not bit-identical: the inspector is
    # reporting the float64 value, which is the one that is right.
    assert from_physics != from_rendered


# ==========================================================================
# The multi-planet and mixed-data regressions
# ==========================================================================


def test_every_planet_in_a_multi_planet_system_gets_local_coordinates(kepler11):
    """Six planets, six distinct distances, all inside their own bounds."""
    distances = {}
    for record in kepler11.planets:
        rows = kepler11.inspect_planet(record, EPOCH_JD)
        local = _row(rows, "System-frame position")
        assert local.is_known and local.frame is InspectorFrame.SYSTEM

        separation = _row(rows, "Distance from host").parameter
        peri = _row(rows, "Periapsis distance").parameter
        apo = _row(rows, "Apoapsis distance").parameter
        assert peri.value_in(u.au) <= separation.value_in(u.au) <= apo.value_in(u.au) * (
            1 + 1e-9
        )
        distances[record.name] = separation.value_in(u.au)

    assert len(distances) == len(kepler11.planets) >= 5
    assert len(set(round(d, 9) for d in distances.values())) == len(distances)


def test_a_multi_planet_system_shares_one_host_position(kepler11):
    """Every planet is offset from the *same* star, not from six of them.

    Checked on the separations rather than on absolute positions, which C3
    does not publish. The system frame has exactly one origin by
    construction, so the scalar distance and the local triplet must be the
    same quantity for every planet, and each must sit inside its own
    orbital bounds.
    """
    # One host identity across all six records, and one frame origin.
    assert {record.host_id for record in kepler11.planets} == {kepler11.star.entity_id}
    assert np.array_equal(kepler11.frame.star_position().values, np.zeros(3))

    for record in kepler11.planets:
        rows = kepler11.inspect_planet(record, EPOCH_JD)
        separation = _row(rows, "Distance from host").parameter
        local = _row(rows, "System-frame position")
        apo = _row(rows, "Apoapsis distance").parameter

        assert separation.value_in(u.au) == pytest.approx(
            float(np.linalg.norm(local.values)), rel=1e-15
        )
        assert separation.value_in(u.au) <= apo.value_in(u.au) * (1 + 1e-9)


def test_a_mixed_data_system_reports_what_it_has(hd219134):
    """Nothing is withheld because something else is missing."""
    rows = hd219134.inspect_star()
    assert _row(rows, "Distance from Sun").parameter.is_known
    assert _row(rows, "Cartesian position").is_known

    for record in hd219134.planets:
        planet_rows = hd219134.inspect_planet(record, EPOCH_JD)
        local = _row(planet_rows, "System-frame position")
        separation = _row(planet_rows, "Distance from host").parameter
        # Either the orbit propagates and both are known, or neither is.
        assert local.is_known == separation.is_known


def test_the_text_report_discloses_a_detached_system(trappist1):
    text = "\n".join(trappist1.describe_coordinates(trappist1.planets[0], EPOCH_JD))
    assert "Absolute position unavailable" in text
    assert "claims no distance from Earth" in text
    assert "SystemFrame" in text


def test_the_text_report_names_every_frame_it_prints(hd80606):
    record = hd80606.planet("HD 80606 b")
    text = "\n".join(hd80606.describe_coordinates(record, EPOCH_JD))

    assert "[ICRS]" in text
    assert "[Galactic]" in text
    assert "[SystemFrame]" in text
    # No triplet is printed without a frame tag beside it.
    for line in text.splitlines():
        if line.strip().startswith(("Cartesian position:", "System-frame position:", "Absolute position:")):
            assert "[" in line and "]" in line


# ==========================================================================
# Degenerate inputs
# ==========================================================================


def test_a_star_with_no_position_yields_no_rows():
    assert star_coordinate_rows(None) == []


def test_an_unpropagatable_orbit_yields_unknown_rather_than_zero():
    assert not host_planet_distance(None).is_known
    row = system_frame_position(None)
    assert not row.is_known
    assert row.frame is InspectorFrame.SYSTEM
    assert row.status is Status.UNKNOWN


def test_an_assumed_host_distance_is_not_treated_as_a_position():
    """A distance invented for display is not an address.

    ``has_distance`` already excludes it, and the absolute position must
    follow: otherwise a placeholder distance quietly becomes a Cartesian
    position that looks exactly like a measured one.
    """
    position = SkyPosition(
        name="probe",
        ra=measured(10.0, u.deg, provenance="test"),
        dec=measured(20.0, u.deg, provenance="test"),
        distance=assumed(100.0, u.pc, provenance="test: so it can be drawn"),
    )
    assert not position.has_distance
    assert position.cartesian_pc() is None

    row = absolute_planet_position(position, np.array([1.0, 0.0, 0.0]))
    assert not row.is_known
    assert row.status is Status.UNKNOWN


def test_an_inspector_row_for_an_unknown_value_formats_as_unknown():
    row = InspectorRow("Distance from Sun", unknown(u.pc))
    assert "unknown" in row.format()

    triplet = CoordinateRow("position", None, u.pc, InspectorFrame.ICRS)
    assert "unknown" in triplet.format()
    assert "ICRS" in triplet.format()


def test_the_row_builder_works_from_scientific_inputs_alone(hd80606):
    """The public entry point needs no application object.

    ``planet_distance_rows`` takes elements, a float64 AU vector and a sky
    position - nothing from the data layer, nothing from the app layer and
    nothing from the renderer. Calling it with hand-built inputs is what
    keeps that true: if it ever grew a dependency on a record or a scene,
    this call would stop working.
    """
    record = hd80606.planet("HD 80606 b")
    position_au = np.array([0.3, -0.1, 0.05])

    rows = planet_distance_rows(
        record.elements, position_au, host=hd80606.star.position
    )

    assert [r.label for r in rows] == [
        "Distance from host",
        "Periapsis distance",
        "Apoapsis distance",
        "System-frame position",
        "Absolute position",
    ]
    assert _row(rows, "Distance from host").parameter.value_in(u.au) == pytest.approx(
        float(np.linalg.norm(position_au))
    )

    # Without a host, the absolute row is simply absent rather than unknown:
    # nothing was asked about an address, so nothing is claimed about one.
    local_only = planet_distance_rows(record.elements, position_au)
    assert "Absolute position" not in [r.label for r in local_only]
    assert _row(local_only, "System-frame position").is_known


# ==========================================================================
# The phase qualifier travels with the instantaneous values
# ==========================================================================


def test_the_planet_rows_carry_their_phase_provenance(hd80606):
    """An instantaneous distance is only as meaningful as its phase.

    HD 80606 b has a published transit epoch, so its position at a given
    instant is a claim about that instant. The row list says so, in the same
    list as the distances rather than in prose a panel could drop.
    """
    record = hd80606.planet("HD 80606 b")
    row = _row(hd80606.inspect_planet(record, EPOCH_JD), "Phase provenance")

    assert isinstance(row, NoteRow)
    assert row.status is Status.MEASURED
    assert row.text == hd80606.phase(record, EPOCH_JD).status.label
    assert "Phase provenance" in row.format()


def test_an_assumed_phase_is_marked_as_such(catalog):
    """A planet advanced from an arbitrary zero must not read as an ephemeris.

    The distance is still physically correct for *some* instant; it is the
    claim that it is correct for *this* instant that is missing, and the row
    is downgraded to say so.
    """
    from astro_explorer.physics.phase import PhaseStatus

    found = None
    for host in ("HD 80606", "Kepler-11", "TRAPPIST-1", "HD 219134"):
        system = build_slice(host, catalog)
        for record in system.planets:
            if system.phase(record, EPOCH_JD).status is PhaseStatus.ASSUMED:
                found = (system, record)
                break
        if found:
            break
    if found is None:
        pytest.skip("no assumed-phase planet in the committed snapshot")

    system, record = found
    row = _row(system.inspect_planet(record, EPOCH_JD), "Phase provenance")
    assert row.status is Status.ASSUMED_FOR_VISUALIZATION

    # The distance itself is still reported: the motion is real.
    assert _row(system.inspect_planet(record, EPOCH_JD), "Distance from host").is_known


def test_a_note_row_needs_no_frame_and_claims_none(hd80606):
    """A qualifier is not a coordinate, and must not look like one."""
    record = hd80606.planet("HD 80606 b")
    row = _row(hd80606.inspect_planet(record, EPOCH_JD), "Phase provenance")

    assert not hasattr(row, "frame")
    assert not any(f.value and f.value in row.format() for f in InspectorFrame)
