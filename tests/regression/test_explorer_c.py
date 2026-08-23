"""Explorer C1: the habitable-zone overlay.

The first scientific overlay, and therefore the first test that the 3D
scene can display a *derived region* without duplicating or contaminating
the physics that produced it. The checklist this file pins:

* the overlay reads the same habitable-zone model the info panel does;
* the renderer never recalculates a boundary;
* unknown luminosity or effective temperature produces no zone at all,
  rather than a default ring;
* the edges keep their measured/derived provenance upstream;
* the zone is disclosed as a stellar-irradiation model, not a claim about
  habitability;
* display thickness and opacity never enter a scientific calculation.

TRAPPIST-1 is the worked counter-example. Its planets are routinely
described as being in the habitable zone, and the Kopparapu coefficients
are fitted only down to 2600 K while the star is cooler than that - so the
honest answer here is no zone, and saying so is the point.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.panel import build_star_panel
from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.physics.stellar import habitable_zone_au
from astro_explorer.provenance import Status
from astro_explorer.rendering.renderer import RenderZone, SceneDescription
from astro_explorer.rendering.scene_builder import (
    HABITABLE_ZONE_DISCLAIMER,
    build_frame_scene,
    habitable_zone_overlay,
)


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
    """A star below the model's fitted temperature range."""
    return build_slice("TRAPPIST-1", catalog)


def _radii(points) -> np.ndarray:
    return np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1)


def _code_only(module) -> str:
    """A module's source with comments and string literals removed."""
    import tokenize

    kept = []
    with open(module.__file__, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


# ==========================================================================
# The overlay and the panel read one model
# ==========================================================================


def test_the_overlay_geometry_matches_the_published_boundaries(hd80606):
    """The drawn rings are the model's own numbers, not a rescaling."""
    zone = hd80606.star.habitable_zone
    scene = build_frame_scene(hd80606.frame, hd80606.star, hd80606.planets)

    assert len(scene.zones) == 1
    overlay = scene.zones[0]

    inner = _radii(overlay.inner_points_local)
    outer = _radii(overlay.outer_points_local)
    assert inner == pytest.approx(zone.inner.value_in(u.au), rel=1e-5)
    assert outer == pytest.approx(zone.outer.value_in(u.au), rel=1e-5)


def test_the_overlay_and_the_panel_agree(hd80606):
    """Two consumers of one model cannot drift apart.

    The panel and the scene both read ``star.habitable_zone``; neither
    evaluates the Kopparapu polynomial itself, so there is no second
    implementation to fall out of step.
    """
    panel = build_star_panel(hd80606.star)
    section = next(s for s in panel.sections if s.heading == "Habitable zone")
    rows = {row.label: row for row in section.rows}

    scene = build_frame_scene(hd80606.frame, hd80606.star, hd80606.planets)
    overlay = scene.zones[0]

    assert rows["Inner edge"].value == pytest.approx(
        float(_radii(overlay.inner_points_local).mean()), rel=1e-5
    )
    assert rows["Outer edge"].value == pytest.approx(
        float(_radii(overlay.outer_points_local).mean()), rel=1e-5
    )


def test_the_renderer_never_recalculates_the_boundaries():
    """The drawing layers hold no way to derive a boundary of their own.

    Scanned as *code*, with comments and docstrings stripped: these modules
    explain in prose that they carry no luminosity, and scanning raw text
    would flag the explanation as the offence.
    """
    from astro_explorer.rendering import gl_backend, renderer

    for module in (gl_backend, renderer):
        code = _code_only(module).lower()
        for token in (
            "kopparapu",
            "habitable_zone_au",
            "_seff",
            "luminosity",
            "effective_temperature",
            "insolation",
        ):
            assert token not in code, "{0}: {1}".format(module.__name__, token)

    # And the primitive it receives carries no way to derive one.
    forbidden = {
        "inner_au",
        "outer_au",
        "luminosity",
        "effective_temperature",
        "temperature_k",
        "model",
        "status",
        "provenance",
    }
    assert not (set(RenderZone.__dataclass_fields__) & forbidden)


def test_the_scene_builder_does_not_re_derive_the_zone():
    """It reads the record's zone; it does not call the physics model."""
    import inspect

    from astro_explorer.rendering import scene_builder

    source = inspect.getsource(scene_builder)
    assert "habitable_zone_au" not in source
    assert "star.habitable_zone" in source


# ==========================================================================
# Unknown inputs produce no zone, not a default one
# ==========================================================================


def test_a_star_outside_the_fitted_range_gets_no_zone(trappist):
    """TRAPPIST-1 is cooler than the coefficients were fitted for."""
    zone = trappist.star.habitable_zone
    assert not zone.is_known

    scene = build_frame_scene(trappist.frame, trappist.star, trappist.planets)
    assert scene.zones == []


def test_a_missing_zone_is_explained_rather_than_silent(trappist):
    scene = build_frame_scene(trappist.frame, trappist.star, trappist.planets)
    text = " ".join(scene.annotations)

    assert "No habitable zone drawn" in text
    assert "unknown rather than defaulted" in text


def test_an_unknown_zone_never_becomes_geometry(hd80606):
    """Every route to a missing boundary returns None, not a ring."""
    frame = hd80606.frame

    assert habitable_zone_overlay(None, frame) is None
    assert habitable_zone_overlay(habitable_zone_au(None, None), frame) is None
    assert habitable_zone_overlay(habitable_zone_au(1.0, None), frame) is None
    assert habitable_zone_overlay(habitable_zone_au(None, 5780.0), frame) is None
    # Below and above the fitted Teff range.
    assert habitable_zone_overlay(habitable_zone_au(1.0, 2000.0), frame) is None
    assert habitable_zone_overlay(habitable_zone_au(1.0, 9000.0), frame) is None
    # A non-positive luminosity is not a small zone; it is no zone.
    assert habitable_zone_overlay(habitable_zone_au(0.0, 5780.0), frame) is None


def test_a_degenerate_band_is_refused(hd80606):
    """Rings that do not bound a region are not drawn as one."""
    from astro_explorer.physics.stellar import HabitableZone
    from astro_explorer.provenance import derived

    inverted = HabitableZone(derived(2.0, u.au, provenance="t"), derived(1.0, u.au, provenance="t"))
    collapsed = HabitableZone(derived(1.0, u.au, provenance="t"), derived(1.0, u.au, provenance="t"))

    assert habitable_zone_overlay(inverted, hd80606.frame) is None
    assert habitable_zone_overlay(collapsed, hd80606.frame) is None


# ==========================================================================
# Provenance stays upstream, and display choices stay downstream
# ==========================================================================


def test_the_edges_keep_their_provenance_upstream(hd80606):
    """Drawing the zone does not launder a derived value into a plain float."""
    zone = hd80606.star.habitable_zone

    assert zone.inner.status is Status.DERIVED
    assert zone.outer.status is Status.DERIVED
    assert "kopparapu" in zone.inner.provenance.lower()

    build_frame_scene(hd80606.frame, hd80606.star, hd80606.planets)

    # The record is untouched by having been drawn.
    after = hd80606.star.habitable_zone
    assert after.inner.status is Status.DERIVED
    assert after.inner.value_in(u.au) == pytest.approx(zone.inner.value_in(u.au))
    assert after.model == zone.model


def test_the_zone_is_disclosed_as_an_irradiation_model(hd80606):
    scene = build_frame_scene(hd80606.frame, hd80606.star, hd80606.planets)
    text = " ".join(scene.annotations)

    assert HABITABLE_ZONE_DISCLAIMER in text
    assert "not a claim about habitability" in text
    # The model is named, so the number can be traced to a paper.
    assert "Kopparapu" in text


def test_display_choices_cannot_move_the_zone(hd80606):
    """Opacity and sample count are presentation, not science."""
    zone = hd80606.star.habitable_zone
    coarse = habitable_zone_overlay(zone, hd80606.frame, samples=12)
    fine = habitable_zone_overlay(zone, hd80606.frame, samples=720)

    assert coarse.vertex_count == 12
    assert fine.vertex_count == 720
    # Same radii regardless of how finely the ring is sampled.
    for overlay in (coarse, fine):
        assert _radii(overlay.inner_points_local) == pytest.approx(
            zone.inner.value_in(u.au), rel=1e-5
        )

    # Recolouring is inert.
    from dataclasses import replace

    recoloured = replace(fine, color=(1.0, 0.0, 0.0, 0.9))
    assert _radii(recoloured.inner_points_local) == pytest.approx(
        _radii(fine.inner_points_local)
    )
    assert hd80606.star.habitable_zone.inner.value_in(u.au) == pytest.approx(
        zone.inner.value_in(u.au)
    )


def test_the_overlay_can_be_switched_off(hd80606):
    scene = build_frame_scene(
        hd80606.frame, hd80606.star, hd80606.planets, draw_habitable_zone=False
    )
    assert scene.zones == []
    assert not any("Habitable-zone cross-section" in note for note in scene.annotations)


def test_the_flat_band_is_disclosed_as_a_cross_section(hd80606):
    """The drawn annulus is a section through a shell, and says so.

    The physical habitable zone is a range of *radial* distances, so the
    region is a spherical shell around the star. Drawing a flat band
    without saying that invites reading the zone as a property of one
    plane - as if a planet on a different plane were outside it.
    """
    scene = build_frame_scene(hd80606.frame, hd80606.star, hd80606.planets)
    text = " ".join(scene.annotations)

    assert "cross-section" in text
    assert "reference plane" in text
    assert "spherical shell" in text


def test_the_zone_edge_colour_is_part_of_the_render_contract():
    """``edge_color`` is documented as drawn, so the backend must draw it.

    Checked here as well as through a real context, because the GL tests
    skip on a machine with no driver and this contract should not go
    unchecked there. The behavioural proof lives in
    ``test_zone_edge_style_is_consumed_by_renderer``.
    """
    import inspect

    from astro_explorer.rendering import gl_backend

    source = inspect.getsource(gl_backend)
    assert "edge_color" in _code_only(gl_backend)
    assert "_batch_zone_edges" in source


def test_kepler11_lies_entirely_starward_of_its_inner_boundary(catalog):
    """The overlay's most useful reading, stated the way it must be worded.

    All six Kepler-11 planets are *starward of* the 1.007 AU inner
    boundary - outside the irradiation-defined habitable zone, on the hot
    side. Saying instead that a planet is "inside the inner edge" invites
    the reading "inside the habitable zone", which is the opposite of what
    it means, so the phrasing is pinned here along with the geometry.
    """
    system = build_slice("Kepler-11", catalog)
    zone = system.star.habitable_zone
    inner = zone.inner.value_in(u.au)

    assert inner == pytest.approx(1.007, rel=1e-3)
    for record in system.planets:
        apoapsis = record.elements.apoapsis.value_in(u.au)
        assert apoapsis < inner, record.name

    scene = build_frame_scene(system.frame, system.star, system.planets)
    assert len(scene.zones) == 1
    zone_radii = _radii(scene.zones[0].inner_points_local)
    orbit_radii = [
        float(np.linalg.norm(orbit.points_local, axis=1).max()) for orbit in scene.orbits
    ]
    assert max(orbit_radii) < zone_radii.min()


# ==========================================================================
# The primitive itself
# ==========================================================================


def test_a_zone_needs_two_rings_of_equal_length():
    """The band is built by pairing the rings index for index."""
    inner = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    outer = 2.0 * inner[:3]
    with pytest.raises(ValueError, match="same vertex count"):
        RenderZone("z", inner, outer)


def test_a_zone_rejects_a_non_finite_boundary():
    ring = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    broken = ring.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RenderZone("z", broken, ring)


def test_a_zone_needs_enough_points_to_bound_a_region():
    with pytest.raises(ValueError, match="at least 3"):
        RenderZone("z", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def test_a_zone_counts_towards_the_scene_extent(hd80606):
    """The camera must be able to frame a zone with no planets drawn."""
    overlay = habitable_zone_overlay(hd80606.star.habitable_zone, hd80606.frame)
    scene = SceneDescription(zones=[overlay])

    assert not scene.is_empty()
    assert scene.bounding_radius() == pytest.approx(
        hd80606.star.habitable_zone.outer.value_in(u.au), rel=1e-4
    )
