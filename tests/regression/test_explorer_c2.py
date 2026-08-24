"""Explorer C2: the orbital-orientation overlay.

C1 asked whether the 3D scene could display a derived *region* without
duplicating the physics that produced it. C2 asks the harder question:
whether it can display an *orientation* without claiming one.

Almost no exoplanet has a measured longitude of the ascending node - it is
not observable from transits or radial velocity - so the display normalises
it to zero. That normalisation is correct and necessary, and it is also the
most dangerous thing in this overlay, because a line of nodes drawn at
``Omega = 0`` looks exactly like a line of nodes drawn at a measured
``Omega = 0``. One is a direction on the sky and the other is a convention,
and a picture cannot tell them apart unless it is made to.

So the checklist here is in two halves:

* the geometry is right - each guide responds to the element that defines
  it, and the whole set agrees with the transform the propagator uses;
* the geometry is *qualified* - measured draws as measured, assumed draws
  dashed and says so, and unknown does not become a scientific-looking line
  at all unless the viewer asked for the normalisation to be shown.
"""

from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.physics.orbital_elements import OrbitalElements
from astro_explorer.physics.node_semantics import resolve_node_azimuth
from astro_explorer.physics.orbital_semantics import OrbitValidity, PeriastronConvention
from astro_explorer.physics.orientation import (
    position_from_eccentric_anomaly,
    rotation_perifocal_to_inertial,
)
from astro_explorer.provenance import Status, derived, measured
from astro_explorer.rendering.renderer import GuideStyle, RenderGuide, SceneDescription
from astro_explorer.rendering.scene_builder import (
    ORIENTATION_DISCLAIMER,
    build_frame_scene,
    orientation_guides,
)
from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.system_frame import SystemFrame


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def hd80606(catalog):
    """A measured inclination, an argument of periastron of unstated convention."""
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def kepler11(catalog):
    """Six transiting planets: i measured, omega and Omega unpublished."""
    return build_slice("Kepler-11", catalog)


@pytest.fixture
def frame():
    return SystemFrame.for_host("test host")


def _elements(
    *,
    inclination_deg=None,
    omega_deg=None,
    node_deg=None,
    eccentricity=0.4,
    axis=1.0,
    convention=PeriastronConvention.PLANET,
) -> OrbitalElements:
    """Elements with exactly the angles a test wants measured."""

    def angle(value):
        return (
            measured(float(value), u.deg, provenance="test").to(u.rad)
            if value is not None
            else None
        )

    return OrbitalElements(
        name="test b",
        semimajor_axis=measured(axis, u.au, provenance="test"),
        eccentricity=measured(eccentricity, provenance="test"),
        period=measured(365.0, u.day, provenance="test"),
        inclination=angle(inclination_deg),
        argument_of_periastron=angle(omega_deg),
        longitude_of_ascending_node=angle(node_deg),
        periastron_convention=convention,
    )


def _points(guide) -> np.ndarray:
    return np.asarray(guide.points_local, dtype=np.float64)


def _named(overlay, name):
    guide = overlay.guide(name)
    assert guide is not None, "{0} was not drawn".format(name)
    return guide


# ==========================================================================
# The geometry answers to the elements that define it
# ==========================================================================


def test_a_coplanar_orbit_plane_matches_the_reference_plane(frame):
    """i = 0: the orbital plane *is* the reference plane."""
    overlay = orientation_guides(_elements(inclination_deg=0.0, node_deg=0.0), frame)

    reference = _points(_named(overlay, "reference-plane"))
    plane = _points(_named(overlay, "orbit-plane"))

    # The same circle, not the same list: the node conversion starts the
    # sampling at a different azimuth, so compare the geometry rather than
    # the point ordering.
    assert np.allclose(plane[:, 2], 0.0, atol=1e-5)
    assert np.ptp(np.linalg.norm(plane, axis=1)) == pytest.approx(0.0, abs=1e-5)
    assert np.linalg.norm(plane, axis=1).mean() == pytest.approx(
        np.linalg.norm(reference, axis=1).mean(), rel=1e-5
    )
    # Every reference point lies on the orbital-plane ring and vice versa:
    # the two are the same circle, traced from a different start angle.
    gaps = np.linalg.norm(reference[:, None, :] - plane[None, :, :], axis=-1)
    assert gaps.min(axis=1).max() < 1e-4
    assert gaps.min(axis=0).max() < 1e-4

    # And there is no arc to draw between two coincident planes.
    assert overlay.guide("inclination") is None
    assert any("no arc to draw" in note for note in overlay.annotations)


def test_a_polar_orbit_normal_lies_in_the_reference_plane(frame):
    """i = 90: the angular momentum is perpendicular to the pole."""
    overlay = orientation_guides(_elements(inclination_deg=90.0, node_deg=0.0), frame)

    normal = _points(_named(overlay, "orbit-normal"))
    tip = normal[1]  # shaft runs origin -> tip
    assert np.isclose(tip[2], 0.0, atol=1e-5)
    # Omega_PA = 0 is North, which is internal +y, so the normal is along +x
    # (East). Before C3.5.1 the raw zero was read as an internal azimuth and
    # put the node on +x, which is East - the annotation said North.
    direction = tip / np.linalg.norm(tip)
    assert np.allclose(direction, [1.0, 0.0, 0.0], atol=1e-5)


def test_the_node_line_rotates_with_the_ascending_node(frame):
    """Omega_PA = 0 is North; 90 is East.

    A catalogued node is a position angle - from North, increasing toward
    East - while the internal frame runs from ``+x`` = East toward ``+y`` =
    North. So the published quarter turn shows up as a quarter turn the
    other way internally, and the guide must land on the *sky* direction the
    catalogue named.
    """
    at_zero = orientation_guides(_elements(inclination_deg=45.0, node_deg=0.0), frame)
    at_ninety = orientation_guides(_elements(inclination_deg=45.0, node_deg=90.0), frame)

    ascending_zero = _points(_named(at_zero, "ascending-node"))[1]
    ascending_ninety = _points(_named(at_ninety, "ascending-node"))[1]

    # PA 0 -> North -> internal +y.
    assert np.allclose(ascending_zero / np.linalg.norm(ascending_zero), [0, 1, 0], atol=1e-5)
    # PA 90 -> East -> internal +x.
    assert np.allclose(
        ascending_ninety / np.linalg.norm(ascending_ninety), [1, 0, 0], atol=1e-5
    )
    # The nodes are in the reference plane whatever the inclination.
    assert np.allclose(_points(_named(at_ninety, "ascending-node"))[:, 2], 0.0, atol=1e-5)


def test_the_periapsis_arrow_rotates_with_the_argument_of_periastron(frame):
    """omega = 90 turns periapsis a quarter turn *within* the orbital plane."""
    at_zero = orientation_guides(
        _elements(inclination_deg=0.0, omega_deg=0.0, node_deg=0.0), frame
    )
    at_ninety = orientation_guides(
        _elements(inclination_deg=0.0, omega_deg=90.0, node_deg=0.0), frame
    )

    tip_zero = _points(_named(at_zero, "periapsis"))[1]
    tip_ninety = _points(_named(at_ninety, "periapsis"))[1]

    # With Omega_PA = 0 the node is North (internal +y), and omega is
    # measured from the node within the orbital plane, so periapsis starts
    # there and a quarter turn carries it to internal -x.
    assert np.allclose(tip_zero / np.linalg.norm(tip_zero), [0, 1, 0], atol=1e-5)
    assert np.allclose(tip_ninety / np.linalg.norm(tip_ninety), [-1, 0, 0], atol=1e-5)
    # Whatever the axes, the two are a quarter turn apart.
    assert np.dot(
        tip_zero / np.linalg.norm(tip_zero), tip_ninety / np.linalg.norm(tip_ninety)
    ) == pytest.approx(0.0, abs=1e-5)


def test_the_guides_use_the_production_orbital_transform(frame):
    """Combined Omega, i and omega must agree with the propagator itself.

    This is the test that makes the overlay trustworthy. A guide built from
    a second, independent reading of the same three angles would look
    perfectly plausible while disagreeing with the orbit it annotates - and
    would disagree silently, because both would be smooth curves in roughly
    the right place.
    """
    i, omega = np.radians([37.0, 64.0])
    elements = _elements(
        inclination_deg=37.0, omega_deg=64.0, node_deg=118.0, eccentricity=0.6
    )
    overlay = orientation_guides(elements, frame)
    # Built through the production node route rather than from the raw
    # catalogue angle. Before C3.5.1 the guides converted the position angle
    # and this expectation did not, so the two silently disagreed by the
    # very ninety degrees the conversion exists to apply.
    rotation = rotation_perifocal_to_inertial(
        i, omega, resolve_node_azimuth(elements.longitude_of_ascending_node)
    )

    # The orbital plane is the perifocal z = 0 plane carried through R.
    plane = _points(_named(overlay, "orbit-plane"))
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    assert np.abs(plane @ normal).max() < 1e-4

    # The periapsis arrow is the perifocal +x axis carried through R.
    tip = _points(_named(overlay, "periapsis"))[1]
    assert np.allclose(tip / np.linalg.norm(tip), rotation @ [1.0, 0.0, 0.0], atol=1e-5)

    # The arc subtends the inclination, about the line of nodes.
    arc = _points(_named(overlay, "inclination"))
    subtended = np.arccos(
        np.clip(
            arc[0] @ arc[-1] / (np.linalg.norm(arc[0]) * np.linalg.norm(arc[-1])), -1, 1
        )
    )
    assert subtended == pytest.approx(i, abs=1e-6)


def test_the_periapsis_arrow_ends_at_the_actual_periapsis(frame):
    """Not merely the right direction: the right point on the orbit."""
    elements = _elements(
        inclination_deg=37.0, omega_deg=64.0, node_deg=118.0, eccentricity=0.6, axis=1.4
    )
    overlay = orientation_guides(elements, frame)
    tip = _points(_named(overlay, "periapsis"))[1]

    # E = 0 is periapsis by definition of the eccentric anomaly.
    expected = position_from_eccentric_anomaly(
        1.4,
        0.6,
        0.0,
        inclination=np.radians(37.0),
        argument_of_periapsis=np.radians(64.0),
        # The production node route, not the raw catalogue angle.
        longitude_of_ascending_node=resolve_node_azimuth(
            elements.longitude_of_ascending_node
        ),
    )
    assert np.allclose(tip, expected, rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(tip) == pytest.approx(1.4 * (1.0 - 0.6), rel=1e-5)


def test_the_guides_enclose_the_orbit_they_annotate(frame):
    """The rings are sized from the orbit, so they frame it rather than hide in it."""
    elements = _elements(inclination_deg=30.0, node_deg=10.0, eccentricity=0.5, axis=2.0)
    overlay = orientation_guides(elements, frame)

    radius = np.linalg.norm(_points(_named(overlay, "orbit-plane")), axis=1)
    assert radius == pytest.approx(2.0 * 1.5, rel=1e-5)  # apoapsis


# ==========================================================================
# Measured, derived, assumed, unknown - and the difference between them
# ==========================================================================


def test_a_measured_node_draws_as_measured(frame):
    overlay = orientation_guides(_elements(inclination_deg=45.0, node_deg=30.0), frame)

    assert _named(overlay, "ascending-node").style is GuideStyle.SOLID
    assert _named(overlay, "orbit-plane").style is GuideStyle.SOLID
    assert any("Ascending node: 30" in note for note in overlay.annotations)
    assert any("measured" in note for note in overlay.annotations)
    # Nothing was assumed, so the dashed-guide disclaimer is not needed.
    assert ORIENTATION_DISCLAIMER not in overlay.annotations


def test_an_unknown_node_is_not_presented_as_observed(kepler11, frame):
    """The default is silence, not a plausible-looking line.

    Kepler-11's planets have measured inclinations and no published node.
    Drawing a solid line of nodes for them would assert a direction on the
    sky that nobody has measured for any of them.
    """
    record = kepler11.planet("Kepler-11 g")
    overlay = orientation_guides(record.elements, kepler11.frame, label=record.name)

    assert overlay.guide("ascending-node") is None
    # The plane's tilt is measured, but the azimuth it is drawn at is not,
    # so it is disclosed as normalised rather than presented as observed.
    assert _named(overlay, "orbit-plane").style is GuideStyle.DASHED

    text = " ".join(overlay.annotations)
    assert "Ascending node: unknown" in text
    assert "Omega = 0 deg" in text
    assert "unconstrained" in text


def test_the_normalised_node_is_available_but_dashed_and_disclosed(kepler11):
    """Asked for explicitly, it is drawn - and it is never drawn solid."""
    record = kepler11.planet("Kepler-11 g")
    overlay = orientation_guides(
        record.elements, kepler11.frame, show_normalised=True, label=record.name
    )

    node = _named(overlay, "ascending-node")
    assert node.style is GuideStyle.DASHED
    assert ORIENTATION_DISCLAIMER in overlay.annotations
    assert any("dashed" in note for note in overlay.annotations)


def test_the_normalised_node_is_tagged_assumed_for_visualization(kepler11):
    """The zero is a display normalisation and the element says so.

    The overlay's dash is downstream of this: the guide is dashed *because*
    the parameter it was drawn from carries ASSUMED_FOR_VISUALIZATION.
    """
    elements = kepler11.planet("Kepler-11 g").elements
    assert not elements.longitude_of_ascending_node.is_known

    display = elements.for_display()
    node = display.longitude_of_ascending_node
    assert node.status is Status.ASSUMED_FOR_VISUALIZATION
    assert node.value_in(u.deg) == pytest.approx(0.0)
    assert "not observable" in (node.note or "") or "normalised" in (node.note or "")


def test_a_stellar_reflex_conversion_stays_derived_and_draws_solid(frame):
    """A converted omega is a real orientation, not a guess.

    +180 degrees from the host star's reflex orbit is a stated transform
    from a stated convention. Dashing it would say the periapsis direction
    was invented; the annotation says it was derived instead.
    """
    elements = _elements(
        inclination_deg=45.0,
        omega_deg=30.0,
        node_deg=10.0,
        convention=PeriastronConvention.STELLAR_REFLEX,
    )
    resolved = elements.argument_of_periapsis_planet
    assert resolved.status is Status.DERIVED
    assert resolved.value_in(u.deg) == pytest.approx(210.0)

    overlay = orientation_guides(elements, frame)
    arrow = _named(overlay, "periapsis")
    assert arrow.style is GuideStyle.SOLID

    # And it points at the converted direction, not the catalogued one.
    tip = _points(arrow)[1]
    expected = rotation_perifocal_to_inertial(
        np.radians(45.0),
        np.radians(210.0),
        resolve_node_azimuth(elements.longitude_of_ascending_node),
    ) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(tip / np.linalg.norm(tip), expected, atol=1e-5)
    assert any("derived" in note for note in overlay.annotations)


def test_an_unstated_convention_draws_dashed_and_says_why(hd80606):
    """HD 80606 b: the number is real, the convention is not stated."""
    record = hd80606.planet("HD 80606 b")
    overlay = orientation_guides(record.elements, hd80606.frame, label=record.name)

    assert record.elements.periastron_convention is PeriastronConvention.AS_REPORTED
    assert _named(overlay, "periapsis").style is GuideStyle.DASHED

    text = " ".join(overlay.annotations)
    assert "assumed" in text
    assert "180 degrees" in text


def test_an_unstated_convention_cannot_claim_full_orientation(frame):
    """Three known angles are not a known orientation.

    ORIENTATION_FULL means the 3D orientation is settled. Under
    AS_REPORTED, periapsis is still ambiguous by half a turn, so the flag
    must stay out of reach however many angles were published.
    """
    as_reported = _elements(
        inclination_deg=45.0,
        omega_deg=30.0,
        node_deg=10.0,
        convention=PeriastronConvention.AS_REPORTED,
    )
    stated = _elements(
        inclination_deg=45.0,
        omega_deg=30.0,
        node_deg=10.0,
        convention=PeriastronConvention.PLANET,
    )

    assert OrbitValidity.ORIENTATION_FULL not in as_reported.validity
    assert OrbitValidity.ORIENTATION_PARTIAL in as_reported.validity
    assert OrbitValidity.ORIENTATION_FULL in stated.validity


def test_provenance_is_never_carried_by_colour_alone(frame):
    """A greyscale print must still distinguish measured from assumed."""
    measured_orbit = orientation_guides(
        _elements(inclination_deg=45.0, omega_deg=30.0, node_deg=10.0), frame
    )
    assumed_orbit = orientation_guides(
        _elements(inclination_deg=45.0, omega_deg=30.0, node_deg=10.0,
                  convention=PeriastronConvention.AS_REPORTED),
        frame,
    )

    solid = _named(measured_orbit, "periapsis")
    dashed = _named(assumed_orbit, "periapsis")
    assert solid.style is not dashed.style
    # Same colour, different stroke: the stroke is what carries the meaning.
    assert solid.color == dashed.color


# ==========================================================================
# The overlay is read-only, and the renderer is told nothing it could reinterpret
# ==========================================================================


def test_drawing_the_guides_does_not_touch_the_orbital_state(hd80606):
    """An overlay is a view of an orbit, never an edit of one."""
    record = hd80606.planet("HD 80606 b")
    elements = record.elements
    before = (
        elements.inclination.value_in(u.rad),
        elements.argument_of_periastron.value_in(u.rad),
        elements.longitude_of_ascending_node.value_in(u.rad),
        elements.semimajor_axis.value_in(u.au),
        elements.eccentricity.value,
        elements.periastron_convention,
    )
    position_before = hd80606.state(record, 2458882.344).position.copy()

    orientation_guides(elements, hd80606.frame, show_normalised=True)

    after = (
        elements.inclination.value_in(u.rad),
        elements.argument_of_periastron.value_in(u.rad),
        elements.longitude_of_ascending_node.value_in(u.rad),
        elements.semimajor_axis.value_in(u.au),
        elements.eccentricity.value,
        elements.periastron_convention,
    )
    assert before == after
    assert np.array_equal(
        hd80606.state(record, 2458882.344).position, position_before
    )


def test_the_renderer_is_given_no_orbital_angles():
    """A guide is finished geometry: vectors, a stroke, a label."""
    forbidden = {
        "inclination",
        "argument_of_periastron",
        "argument_of_periapsis",
        "longitude_of_ascending_node",
        "omega",
        "Omega",
        "eccentricity",
        "semimajor_axis",
        "status",
        "provenance",
        "convention",
    }
    assert not (set(RenderGuide.__dataclass_fields__) & forbidden)

    # And the backend holds no way to reconstruct them.
    import inspect
    import tokenize

    from astro_explorer.rendering import gl_backend

    kept = []
    with open(gl_backend.__file__, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    code = " ".join(kept).lower()
    for token in ("inclination", "periapsis", "ascending_node", "eccentricity",
                  "rotation_perifocal"):
        assert token not in code, token
    assert "orientation_guides" not in inspect.getsource(gl_backend)


def test_a_guide_refuses_a_non_finite_point():
    with pytest.raises(ValueError, match="finite"):
        RenderGuide("g", [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])


def test_a_guide_needs_two_points_to_be_a_line():
    with pytest.raises(ValueError, match="at least 2"):
        RenderGuide("g", [[0.0, 0.0, 0.0]])


def test_guides_count_towards_the_scene_extent(frame):
    """The camera must be able to frame an overlay drawn around an orbit."""
    overlay = orientation_guides(_elements(inclination_deg=30.0, axis=2.0), frame)
    scene = SceneDescription(guides=overlay.guides)

    assert not scene.is_empty()
    assert scene.bounding_radius() == pytest.approx(2.0 * 1.4, rel=1e-3)


# ==========================================================================
# In a scene: selection, and what the guides must not disturb
# ==========================================================================


def test_selection_chooses_whose_guides_are_drawn(kepler11):
    """One planet's orientation at a time, and it is the selected one."""
    anomalies = kepler11.mean_anomalies(2455590.0)

    def scene_for(name):
        return build_frame_scene(
            kepler11.frame,
            kepler11.star,
            kepler11.planets,
            mean_anomalies=anomalies,
            orientation_for=name,
            show_normalised_orientation=True,
        )

    inner = scene_for("Kepler-11 b")
    outer = scene_for("Kepler-11 g")
    none = scene_for(None)

    assert none.guides == []
    assert inner.guides and outer.guides
    assert all("Kepler-11" in g.identifier or ":" in g.identifier for g in inner.guides)

    # Different planets, differently sized overlays - and the same orbits.
    inner_radius = np.linalg.norm(_points(inner.guides[0]), axis=1).max()
    outer_radius = np.linalg.norm(_points(outer.guides[0]), axis=1).max()
    assert inner_radius < outer_radius

    for a, b, c in zip(inner.orbits, outer.orbits, none.orbits):
        assert np.array_equal(a.points_local, b.points_local)
        assert np.array_equal(a.points_local, c.points_local)
    for a, b in zip(inner.planets, none.planets):
        assert np.array_equal(a.position_local, b.position_local)


def test_selecting_a_planet_does_not_rotate_the_habitable_zone(hd80606):
    """The zone is radial: it belongs to the star, not to a planet's plane.

    Rotating the annulus into the selected orbit's plane would be an easy
    and wrong thing to do - it would look tidier, and it would say the
    habitable zone were a property of that plane rather than of distance
    from the star.
    """
    anomalies = hd80606.mean_anomalies(2458882.344)
    plain = build_frame_scene(
        hd80606.frame, hd80606.star, hd80606.planets, mean_anomalies=anomalies
    )
    with_guides = build_frame_scene(
        hd80606.frame,
        hd80606.star,
        hd80606.planets,
        mean_anomalies=anomalies,
        orientation_for="HD 80606 b",
        show_normalised_orientation=True,
    )

    assert len(with_guides.zones) == 1
    zone = with_guides.zones[0]
    assert np.allclose(zone.inner_points_local[:, 2], 0.0, atol=1e-6)
    assert np.allclose(zone.outer_points_local[:, 2], 0.0, atol=1e-6)
    assert np.array_equal(zone.inner_points_local, plain.zones[0].inner_points_local)
    assert np.array_equal(zone.outer_points_local, plain.zones[0].outer_points_local)


def test_the_scene_says_whose_orientation_is_shown(kepler11):
    scene = build_frame_scene(
        kepler11.frame,
        kepler11.star,
        kepler11.planets,
        mean_anomalies=kepler11.mean_anomalies(2455590.0),
        orientation_for="Kepler-11 g",
    )
    text = " ".join(scene.annotations)
    assert "Orientation guides: Kepler-11 g" in text
    assert "Ascending node: unknown" in text


def test_an_orbit_with_no_semimajor_axis_gets_no_guides(frame):
    """Nothing to orient, and it says so rather than drawing a bare plane."""
    elements = OrbitalElements(name="nothing b")
    overlay = orientation_guides(elements, frame, label="nothing b")

    assert overlay.guides == []
    assert not overlay
    assert any("no orbit to orient" in note for note in overlay.annotations)


def test_the_explorer_draws_guides_for_the_selected_planet(kepler11):
    """Selection in the app reaches the scene, without a second code path."""
    from astro_explorer.app.explorer import Explorer, Selection

    explorer = Explorer()
    explorer.open_detached("Kepler-11", kepler11.planets, kepler11.star)
    explorer.show_normalised_orientation = True

    assert explorer.scene(2455590.0).guides == []

    record = kepler11.planets[-1]
    explorer.selection = Selection(
        entity_id=str(record.entity_id or record.name),
        display_name=record.name,
        kind="planet",
    )
    guided = explorer.scene(2455590.0)
    assert guided.guides
    assert any("Orientation guides" in note for note in guided.annotations)


# ==========================================================================
# C2 follow-ups, carried into the C3 working tree
# ==========================================================================


def test_a_known_node_annotation_uses_its_actual_provenance(frame):
    """A known node is not necessarily a *measured* one.

    The annotation used to hard-code the word "measured" for any node that
    was known, which was true only because no path in the catalogue yet
    produced a derived one. The overlay claims to support the full
    provenance ladder, so the sentence has to read the parameter rather
    than assume the top rung of it: a derived node is still drawn solid -
    it is genuinely constrained - but it must not be *described* as an
    observation.
    """
    elements = _elements(inclination_deg=45.0, node_deg=30.0)
    elements = replace(
        elements,
        longitude_of_ascending_node=derived(
            np.deg2rad(30.0), u.rad, provenance="test: from a fitted astrometric arc"
        ),
    )

    overlay = orientation_guides(elements, frame)

    # Constrained, so still solid ...
    assert _named(overlay, "ascending-node").style is GuideStyle.SOLID
    node_note = next(
        note for note in overlay.annotations if note.startswith("Ascending node:")
    )
    # ... but described by what it actually is.
    assert "Ascending node: 30" in node_note
    assert "(derived)" in node_note
    assert "measured" not in node_note


def test_a_measured_node_is_still_annotated_measured(frame):
    """The common case is unchanged by reading provenance properly."""
    overlay = orientation_guides(_elements(inclination_deg=45.0, node_deg=30.0), frame)

    node_note = next(
        note for note in overlay.annotations if note.startswith("Ascending node:")
    )
    assert "(measured)" in node_note


# ==========================================================================
# C3.5.1: the conversion is on the production path, not beside it
# ==========================================================================

_PA_TO_SKY = [
    (0.0, "north"),
    (90.0, "east"),
    (180.0, "south"),
    (270.0, "west"),
]


def _sky_directions():
    """Internal-frame unit vectors for the four cardinal sky directions.

    The canonical frame is ``+x`` = East, ``+y`` = North, so North is +y and
    East is +x. Writing them out here rather than reusing the conversion
    means the expectation is independent of the code under test.
    """
    return {
        "east": np.array([1.0, 0.0, 0.0]),
        "north": np.array([0.0, 1.0, 0.0]),
        "west": np.array([-1.0, 0.0, 0.0]),
        "south": np.array([0.0, -1.0, 0.0]),
    }


@pytest.mark.parametrize("pa_deg,expected", _PA_TO_SKY)
def test_a_raw_position_angle_propagates_to_the_named_sky_direction(pa_deg, expected):
    """The end-to-end check the C3.5 audit asked for.

    A catalogued node is a position angle: 0 is North, 90 is East. Before
    C3.5.1 that number went straight into ``R_z``, whose ``+x`` is East - so
    a published node of 0 was drawn East while every annotation said North.

    This drives the *production* propagator, not the conversion function, so
    it fails if any route stops applying the conversion.
    """
    from astro_explorer.physics.orbital_elements import position_at_eccentric_anomaly

    elements = _elements(
        inclination_deg=0.0, omega_deg=0.0, node_deg=pa_deg, eccentricity=0.0
    )
    # i = 0 and omega = 0 puts periapsis on the line of nodes itself, so the
    # propagated position at E = 0 *is* the node direction.
    node_direction = position_at_eccentric_anomaly(elements.for_display(), 0.0)
    unit = node_direction / np.linalg.norm(node_direction)

    assert np.allclose(unit, _sky_directions()[expected], atol=1e-9), (pa_deg, unit)


@pytest.mark.parametrize("pa_deg,expected", _PA_TO_SKY)
def test_the_orientation_guide_node_matches_the_named_sky_direction(pa_deg, expected):
    """The C2 guide must land on the same sky direction as the orbit."""
    elements = _elements(inclination_deg=30.0, omega_deg=0.0, node_deg=pa_deg)
    overlay = orientation_guides(elements, SystemFrame.for_host("test host"))

    node_tip = _points(_named(overlay, "ascending-node"))[1]
    unit = node_tip / np.linalg.norm(node_tip)

    assert np.allclose(unit, _sky_directions()[expected], atol=1e-5), (pa_deg, unit)


def test_the_guide_node_agrees_with_the_propagated_orbit():
    """Guide and orbit are built from one azimuth, so they cannot diverge."""
    from astro_explorer.physics.orbital_elements import position_at_eccentric_anomaly

    for pa_deg in (0.0, 37.0, 118.0, 264.0):
        elements = _elements(
            inclination_deg=52.0, omega_deg=0.0, node_deg=pa_deg, eccentricity=0.0
        )
        overlay = orientation_guides(elements, SystemFrame.for_host("test host"))

        guide = _points(_named(overlay, "ascending-node"))[1]
        orbit = position_at_eccentric_anomaly(elements.for_display(), 0.0)

        assert np.allclose(
            guide / np.linalg.norm(guide), orbit / np.linalg.norm(orbit), atol=1e-5
        ), pa_deg


def test_the_slice_state_uses_the_canonicalised_node(catalog):
    """The third production route - the propagated state - converts too.

    ``SystemSlice.state`` builds its own call into the propagator rather
    than going through ``_display_angles``, so it is a separate place the
    conversion could have been missed.
    """
    from astro_explorer.physics.node_semantics import resolve_node_azimuth
    from astro_explorer.physics.state_vectors import state_at_mean_anomaly

    system = build_slice("HD 80606", catalog)
    record = system.planet("HD 80606 b")
    display = record.elements.for_display()
    anomaly = system.phase(record, 2460000.0).mean_anomaly

    expected = state_at_mean_anomaly(
        display.semimajor_axis.value_in(u.au),
        display.eccentricity.value_in(u.dimensionless_unscaled, 0.0),
        anomaly,
        inclination=display.inclination.value_in(u.rad, 0.0),
        argument_of_periapsis=display.argument_of_periastron.value_in(u.rad, 0.0),
        longitude_of_ascending_node=resolve_node_azimuth(
            display.longitude_of_ascending_node
        ),
        mu=system.mu,
    )
    assert np.allclose(
        system.state(record, 2460000.0).position, expected.position, rtol=0, atol=0
    )


def test_a_normalised_node_is_displayed_north():
    """``Omega_PA = 0`` means North, and must be *drawn* North.

    This is the case that would have been silently wrong for every planet in
    the catalogue, since almost none publish a node and all of them are
    normalised to zero. The text said North; the geometry said East.
    """
    from astro_explorer.physics.node_semantics import resolve_node_azimuth

    unpublished = _elements(inclination_deg=40.0, omega_deg=0.0, node_deg=None)
    display = unpublished.for_display()
    node = display.longitude_of_ascending_node

    # The display normalisation really is a zero position angle ...
    assert node.value_in(u.deg) == pytest.approx(0.0)
    assert node.status is Status.ASSUMED_FOR_VISUALIZATION
    # ... and that zero means North, which is a quarter turn internally.
    assert resolve_node_azimuth(node) == pytest.approx(np.pi / 2.0)

    overlay = orientation_guides(
        unpublished, SystemFrame.for_host("test host"), show_normalised=True
    )
    tip = _points(_named(overlay, "ascending-node"))[1]
    assert np.allclose(tip / np.linalg.norm(tip), [0.0, 1.0, 0.0], atol=1e-5)


def test_no_production_module_passes_a_raw_node_into_the_rotation():
    """Structural guard: the conversion cannot be bypassed by a new caller.

    The bug this slice fixes was not a wrong formula - the formula existed
    and was tested. It was that three production call sites did not use it.
    So this scans production source for a node angle being read straight out
    of a parameter, which is the shape that mistake takes.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "astro_explorer"
    offenders = []
    pattern = re.compile(r"longitude_of_ascending_node\s*\.\s*value_in")

    for path in root.rglob("*.py"):
        if path.name == "node_semantics.py":
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append("{0}:{1}".format(path.name, number))

    assert not offenders, (
        "a raw node angle is being read for the transform; it must go "
        "through resolve_node_azimuth: {0}".format(offenders)
    )
