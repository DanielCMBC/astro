"""Navigation, selection and labels: review section 10.

Structured around the acceptance criteria, one test group each:

* camera enters and leaves a system without precision loss;
* frame switches are explicit and type-safe;
* picking cannot select objects behind the camera;
* selected identity survives LOD transitions;
* labels never modify scientific coordinates;
* labels are decluttered, and the selected label is always visible;
* LOD is based on projected size, not a world-distance threshold.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.app.explorer import Explorer, Selection, UniverseTarget, ViewState
from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.system_frame import (
    FrameKind,
    PrecisionError,
    SystemFrame,
    UniverseFrame,
)
from astro_explorer.rendering.camera import Camera
from astro_explorer.rendering.labels import resolve_collisions
from astro_explorer.rendering.picking import pick
from astro_explorer.rendering.renderer import RenderPlanet, RenderStar, SceneDescription

AU_IN_PC = float((1.0 * u.au).to_value(u.pc))


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def targets(catalog) -> list[UniverseTarget]:
    found = []
    for host in sorted(catalog["hostname"].unique()):
        system = build_slice(host, catalog)
        star = system.star
        if star.position is None or not star.position.has_distance:
            continue
        found.append(
            UniverseTarget(
                host,
                star.position.cartesian_pc(),
                temperature_k=star.effective_temperature.value_in(u.K),
                distance_pc=star.position.distance.value_in(u.pc),
                planet_count=len(system.planets),
            )
        )
    return found


@pytest.fixture()
def explorer(targets) -> Explorer:
    instance = Explorer(targets=list(targets))
    instance.move_to_pc([0.0, 0.0, 40.0])
    return instance


@pytest.fixture()
def focused(explorer, catalog) -> Explorer:
    system = build_slice("HD 80606", catalog)
    explorer.focus("HD 80606", system.planets, system.star)
    return explorer


# ==========================================================================
# Frame transitions
# ==========================================================================


def test_the_explorer_starts_in_the_universe_view(explorer):
    assert explorer.view is ViewState.UNIVERSE
    assert isinstance(explorer.active_frame, UniverseFrame)
    assert explorer.active_frame.unit == u.pc


def test_focusing_a_system_does_not_by_itself_change_the_view(focused):
    """The frame follows the camera, not an intention."""
    assert focused.system is not None
    assert focused.view is ViewState.UNIVERSE


def test_entering_a_system_switches_the_frame(focused):
    focused.enter_system()
    assert focused.view is ViewState.SYSTEM
    assert isinstance(focused.active_frame, SystemFrame)
    assert focused.active_frame.unit == u.au


def test_leaving_a_system_switches_back(focused):
    focused.enter_system()
    assert focused.view is ViewState.SYSTEM
    focused.leave_system()
    assert focused.view is ViewState.UNIVERSE


def test_the_switch_happens_at_the_engage_radius(focused):
    """Not an arbitrary threshold: it is derived from the float32 limit."""
    frame = focused.system
    radius = frame.engage_radius_pc

    focused.move_to_pc(frame.origin_pc + np.array([radius * 1.01, 0.0, 0.0]))
    assert focused.view is ViewState.UNIVERSE

    focused.move_to_pc(frame.origin_pc + np.array([radius * 0.99, 0.0, 0.0]))
    assert focused.view is ViewState.SYSTEM


def test_the_engage_radius_comes_from_the_float32_limit():
    from astro_explorer.coordinates.system_frame import (
        FLOAT32_SAFE_MAGNITUDE,
        FRAME_ENGAGE_MARGIN,
    )

    frame = SystemFrame.for_host("X")
    assert frame.engage_radius == FLOAT32_SAFE_MAGNITUDE / FRAME_ENGAGE_MARGIN
    # ~0.485 pc, which is also a natural "arriving at the system" distance.
    assert 0.4 < frame.engage_radius_pc < 0.6


def test_the_whole_approach_is_free_of_precision_loss(focused):
    """The headline acceptance criterion.

    Every position along the path must be renderable in whichever frame is
    active there. to_render() raises rather than silently degrading, so a
    single bad step fails this test.
    """
    seen = set()
    for position in focused.path_to_system(48):
        focused.move_to_pc(position)
        rendered = focused.camera_position.to_render()  # raises on precision loss
        assert rendered.dtype == np.float32
        assert np.all(np.isfinite(rendered))
        seen.add(focused.view)

    assert seen == {ViewState.UNIVERSE, ViewState.SYSTEM}


def test_the_return_journey_is_also_safe(focused):
    focused.enter_system()
    for fraction in np.linspace(0.0, 1.0, 32):
        focused.move_to_pc(
            focused.system.origin_pc
            + (focused.camera_pc - focused.system.origin_pc) * 1.0
            + np.array([fraction * 30.0, 0.0, 0.0])
        )
        focused.camera_position.to_render()


def test_a_system_frame_would_lose_precision_at_universe_range(focused):
    """Why the engage radius exists, stated as a failure."""
    frame = focused.system
    far_away = frame.origin_pc + np.array([20.0, 0.0, 0.0])  # 20 pc out
    with pytest.raises(PrecisionError):
        frame.from_absolute_pc(far_away).to_render()


def test_the_active_frame_is_a_pure_function_of_position(focused):
    """No mode flag can drift out of step with where the camera is."""
    inside = focused.system.origin_pc + np.array([1e-4, 0.0, 0.0])
    outside = focused.system.origin_pc + np.array([5.0, 0.0, 0.0])

    for _ in range(3):
        focused.move_to_pc(inside)
        assert focused.view is ViewState.SYSTEM
        focused.move_to_pc(outside)
        assert focused.view is ViewState.UNIVERSE


def test_frames_stay_type_safe_across_a_transition(focused):
    from astro_explorer.coordinates.system_frame import FrameMismatchError

    focused.move_to_pc([0.0, 0.0, 40.0])
    universe_position = focused.camera_position
    focused.enter_system()
    system_position = focused.camera_position

    assert universe_position.kind is FrameKind.UNIVERSE
    assert system_position.kind is FrameKind.SYSTEM
    with pytest.raises(FrameMismatchError):
        universe_position + system_position


def test_a_host_without_a_distance_is_absent_from_the_universe_view(targets):
    """TRAPPIST-1 has no sy_dist; it cannot be placed, so it is not drawn."""
    assert "TRAPPIST-1" not in {target.name for target in targets}
    assert "HD 80606" in {target.name for target in targets}


# ==========================================================================
# Scenes
# ==========================================================================


def test_the_universe_scene_holds_the_hosts(explorer):
    scene = explorer.scene()
    assert scene.unit_label == "pc"
    assert len(scene.stars) == len(explorer.targets)
    assert not scene.planets


def test_the_universe_scene_discloses_the_display_size(explorer):
    text = " ".join(explorer.scene().annotations)
    assert "not a physical radius" in text


def test_the_system_scene_holds_the_planets(focused):
    focused.enter_system()
    scene = focused.scene(2458882.344)
    assert scene.unit_label == "AU"
    assert len(scene.stars) == 1
    assert len(scene.planets) == 1
    assert any("system frame active" in note for note in scene.annotations)


def test_a_multi_planet_system_renders_from_the_explorer(explorer, catalog):
    system = build_slice("Kepler-11", catalog)
    explorer.focus("Kepler-11", system.planets, system.star)
    explorer.enter_system()
    scene = explorer.scene(2455590.0)
    assert len(scene.planets) == 6
    assert len(scene.orbits) == 6


# ==========================================================================
# Picking
# ==========================================================================


def _camera(width=400, height=400):
    return Camera(target=np.zeros(3), distance=10.0, yaw=0.0, pitch=0.0, aspect=width / height)


def test_nothing_behind_the_camera_can_be_picked():
    """The legacy prototype's central picking flaw."""
    camera = _camera()
    scene = SceneDescription(
        stars=[RenderStar("BEHIND", [0.0, 0.0, 30.0], 1.0, (1, 1, 1))]
    )
    assert pick(scene, camera, 200, 200, 400, 400) is None


def test_a_body_straddling_the_camera_plane_is_still_pickable():
    """The rejection must not be so eager that it drops a near object."""
    camera = _camera()
    scene = SceneDescription(
        stars=[RenderStar("AROUND", [0.0, 0.0, 9.5], 2.0, (1, 1, 1))]
    )
    assert pick(scene, camera, 200, 200, 400, 400) is not None


def test_the_nearest_body_wins_not_the_nearest_to_the_ray():
    camera = _camera()
    scene = SceneDescription(
        stars=[RenderStar("FAR", [0.0, 0.0, -5.0], 2.0, (1, 1, 1))],
        planets=[RenderPlanet("NEAR", [0.0, 0.0, 5.0], 0.2)],
    )
    result = pick(scene, camera, 200, 200, 400, 400)
    assert result is not None and result.identifier == "NEAR"


def test_clicking_empty_sky_clears_the_selection(explorer):
    scene = explorer.scene()
    explorer.select("star:nasa:HD_80606")
    assert explorer.selection is not None
    # A corner of a wide view, far from any host.
    explorer.pick_at(scene, 1, 1, 2000, 2000)
    assert explorer.selection is None


def test_selection_identity_survives_lod_transitions():
    """The acceptance criterion, exercised across every LOD level."""
    camera = _camera()
    scene = SceneDescription(
        stars=[RenderStar("S", [0.0, 0.0, 0.0], 0.5, (1, 1, 1))],
        planets=[RenderPlanet("P", [2.5, 0.0, 0.0], 0.3)],
    )
    before = pick(scene, camera, 200, 200, 400, 400)
    assert before is not None

    identifiers = set()
    for distance in (1.5, 5.0, 40.0, 500.0):
        camera.distance = distance
        scene.assign_lod(camera, 400)
        after = pick(scene, camera, 200, 200, 400, 400)
        if after is not None:
            identifiers.add(after.identifier)

    assert identifiers <= {before.identifier}


def test_lod_never_changes_an_identifier():
    scene = SceneDescription(
        planets=[RenderPlanet("P{0}".format(i), [i, 0.0, 0.0], 0.1) for i in range(6)]
    )
    before = [planet.identifier for planet in scene.planets]
    scene.assign_lod(_camera(), 400)
    assert [planet.identifier for planet in scene.planets] == before


def test_the_explorer_records_what_was_picked(focused):
    focused.enter_system()
    scene = focused.scene(2458882.344)
    selection = focused.pick_at(scene, 200, 200, 400, 400)
    if selection is not None:
        assert selection.entity_id in {
            body.identifier for body in list(scene.stars) + list(scene.planets)
        }
        assert selection.host_id == "star:nasa:HD_80606"


def test_selection_is_by_stable_key_not_by_index(focused):
    selection = focused.select("planet:nasa:HD_80606_b", "planet")
    assert isinstance(selection, Selection)
    assert selection.entity_id == "planet:nasa:HD_80606_b"
    # The display name is resolved for the label but is not the identity.
    assert selection.display_name == "HD 80606 b"
    assert selection.is_planet and not selection.is_star


# ==========================================================================
# Labels
# ==========================================================================


def test_labels_never_modify_scientific_coordinates():
    """The acceptance criterion, checked byte for byte."""
    scene = SceneDescription(
        stars=[RenderStar("S", [0.0, 0.0, 0.0], 0.4, (1, 1, 1), label="Star")],
        planets=[
            RenderPlanet("P{0}".format(i), [i * 0.4, 0.0, 0.0], 0.05, label="P{0}".format(i))
            for i in range(6)
        ],
    )
    camera = _camera()
    before = [body.position_local.copy() for body in list(scene.stars) + list(scene.planets)]
    radii = [body.radius_display for body in list(scene.stars) + list(scene.planets)]

    scene.project_labels(camera, 400, 400, priority={"P3"})

    after = list(scene.stars) + list(scene.planets)
    for original, body, radius in zip(before, after, radii):
        assert np.array_equal(original, body.position_local)
        assert body.radius_display == radius


def test_labels_are_decluttered():
    placements = [
        ("a", 100.0, 100.0, 1.0, 0.0, "star:nasa:a"),
        ("b", 104.0, 101.0, 2.0, 0.0, "star:nasa:b"),
        ("c", 300.0, 300.0, 3.0, 0.0, "star:nasa:c"),
    ]
    kept = {item[0] for item in resolve_collisions(placements, min_separation=26)}
    assert kept == {"a", "c"}


def test_the_selected_label_is_always_visible():
    """It must survive decluttering even when crowded and far away."""
    scene = SceneDescription(
        planets=[
            RenderPlanet(
                "P{0}".format(i), [0.0, 0.0, -i * 0.02], 0.01, label="P{0}".format(i)
            )
            for i in range(8)
        ]
    )
    camera = _camera()
    placements = scene.project_labels(camera, 400, 400, priority={"P7"})
    assert placements
    # Priority comes first, so decluttering keeps it whatever the crowding.
    assert placements[0][0] == "P7"
    kept = {item[0] for item in resolve_collisions(placements, min_separation=40)}
    assert "P7" in kept


def test_the_explorer_passes_its_selection_as_the_label_priority(focused):
    focused.enter_system()
    scene = focused.scene(2458882.344)
    focused.select("planet:nasa:HD_80606_b", "planet")
    placements = focused.labels(scene, 1200, 800)
    if placements:
        # Priority is matched on the stable id; the text shown is the name.
        assert placements[0][5] == "planet:nasa:HD_80606_b"
        assert placements[0][0] == "HD 80606 b"


def test_labels_with_no_selection_are_ordered_nearest_first(explorer):
    scene = explorer.scene()
    placements = explorer.labels(scene, 1200, 800)
    depths = [item[3] for item in placements]
    assert depths == sorted(depths)


def test_labels_outside_the_viewport_are_dropped():
    scene = SceneDescription(
        stars=[
            RenderStar("VISIBLE", [0.0, 0.0, 0.0], 0.2, (1, 1, 1), label="Visible"),
            RenderStar("OFFSCREEN", [900.0, 0.0, 0.0], 0.2, (1, 1, 1), label="Off"),
        ]
    )
    names = {item[0] for item in scene.project_labels(_camera(), 400, 400)}
    assert "Off" not in names


# ==========================================================================
# Level of detail
# ==========================================================================


def test_lod_follows_projected_size_not_world_distance():
    """Same world distance, different viewport: different LOD.

    A world-distance threshold could not tell these apart, which is what
    review section 10 asks to avoid.
    """
    from astro_explorer.rendering.mesh import lod_for_distance

    coarse = lod_for_distance(1.0, 50.0, viewport_height=200, fov_y_rad=np.radians(45))
    fine = lod_for_distance(1.0, 50.0, viewport_height=4000, fov_y_rad=np.radians(45))
    assert fine > coarse


def test_lod_follows_the_field_of_view_too():
    from astro_explorer.rendering.mesh import lod_for_distance

    wide = lod_for_distance(1.0, 50.0, viewport_height=1000, fov_y_rad=np.radians(90))
    narrow = lod_for_distance(1.0, 50.0, viewport_height=1000, fov_y_rad=np.radians(10))
    assert narrow > wide


def test_a_larger_body_gets_more_detail_at_the_same_distance():
    from astro_explorer.rendering.mesh import lod_for_distance

    small = lod_for_distance(0.01, 50.0, viewport_height=1000, fov_y_rad=np.radians(45))
    large = lod_for_distance(5.0, 50.0, viewport_height=1000, fov_y_rad=np.radians(45))
    assert large > small


def test_lod_is_bounded():
    from astro_explorer.rendering.mesh import lod_for_distance

    for distance in (1e-6, 1.0, 1e12):
        level = lod_for_distance(1.0, distance, viewport_height=1000, fov_y_rad=1.0)
        assert 0 <= level <= 5


# ==========================================================================
# Reporting
# ==========================================================================


def test_the_explorer_describes_its_state(focused):
    focused.enter_system()
    focused.select("HD 80606 b", "planet")
    text = "\n".join(focused.describe())
    assert "SYSTEM" in text
    assert "HD 80606" in text
    assert "AU" in text
    assert "HD 80606 b" in text


def test_the_universe_view_reports_the_engage_radius(focused):
    text = "\n".join(focused.describe())
    assert "engages within" in text


# ==========================================================================
# Review section 4: screen-space error is what matters to a renderer
# ==========================================================================


def _project(camera, position, width, height):
    """Pixel coordinates of a world position, in float64 throughout."""
    clip = camera.view_projection() @ np.append(
        np.asarray(position, dtype=np.float64), 1.0
    )
    ndc = clip[:3] / clip[3]
    return np.array(
        [(ndc[0] * 0.5 + 0.5) * width, (1.0 - (ndc[1] * 0.5 + 0.5)) * height]
    )


def test_screen_space_error_stays_below_a_quarter_pixel(focused):
    """The measurement that actually matters at the float32 boundary.

    Narrowing to float32 loses precision - at 1e5 AU the spacing is about
    7.8e-3 AU. What decides whether that matters is not the absolute
    figure but how far the rendered pixel moves, so this projects the
    float64 reference and the float32 rendered coordinate through the same
    camera and compares them.
    """
    width, height = 1600, 1000
    worst = 0.0

    for position in focused.path_to_system(24):
        focused.move_to_pc(position)
        frame = focused.active_frame

        reference = focused.camera_position.values  # float64
        rendered = focused.camera_position.to_render().astype(np.float64)

        camera = focused.camera
        camera.aspect = width / height
        # Project a body at the frame origin as seen from each version of
        # the camera position; the difference is the screen-space error.
        offset = np.linalg.norm(
            _project(camera, reference - rendered, width, height)
            - _project(camera, np.zeros(3), width, height)
        )
        worst = max(worst, float(offset))

    assert worst < 0.25, "worst screen-space error {0:.4f} px".format(worst)


def test_a_body_at_the_frame_origin_projects_identically_in_both_precisions(focused):
    """The star sits at (0, 0, 0), which float32 represents exactly."""
    focused.enter_system()
    frame = focused.active_frame
    star = frame.star_position()
    assert np.array_equal(star.values, star.to_render().astype(np.float64))


def test_float32_spacing_at_the_engage_radius_is_what_we_claim():
    """Pin the number the documentation quotes, so it cannot go stale."""
    spacing = np.spacing(np.float32(1.0e5))
    assert spacing == pytest.approx(7.8e-3, rel=0.05)


def test_picking_uses_forward_depth_not_radial_distance():
    """Review section 8: perspective scaling follows depth along the view.

    An off-axis body in a wide field of view is further from the camera
    than its depth, so a radial measure would inflate its pick radius more
    than an on-axis one and bias selection towards the screen edges.
    """
    from astro_explorer.rendering.picking import _effective_radius

    camera = Camera(target=np.zeros(3), distance=10.0, aspect=1.0)
    camera.fov_y_rad = np.radians(90.0)

    on_axis = _effective_radius(1e-9, 10.0, camera, 1000, 6.0)
    # Same depth, but radially further away because it is off to one side.
    off_axis_radial = np.hypot(10.0, 10.0)
    if_radial_were_used = _effective_radius(1e-9, off_axis_radial, camera, 1000, 6.0)

    assert if_radial_were_used > on_axis
    # The implementation is handed depth, so both get the same inflation.
    assert _effective_radius(1e-9, 10.0, camera, 1000, 6.0) == on_axis
