# The explorer: navigation, selection and labels

Review section 10. The interaction model is the one the legacy prototype was
right about:

```
fly through hosts -> select a star -> enter its system -> inspect planets
```

None of its implementation is reused. See
[`legacy-3d-prototype.md`](legacy-3d-prototype.md) for the audited defect
list and the assertions that keep each one from returning.

```bash
python -m astro_explorer.app.explorer_demo
python -m astro_explorer.app.explorer_demo --host Kepler-11 --frames 8
python -m astro_explorer.app.explorer_demo --no-render
```

## Which frame is active is not a mode

The obvious design is a view-mode flag the user toggles, with a camera that
follows it. That design has a window in which the mode has changed and the
camera has not, or the reverse - and that window is exactly where a
coordinate gets expressed in a frame that cannot represent it.

So the active frame is a **pure function of where the camera is**: the
finest frame whose `engage_radius` contains it.

```python
@property
def active_frame(self):
    if self.system is not None and self.system.contains(self._camera_pc):
        return self.system
    return self.universe
```

There is no state to drift. `focus()` names a target system without moving
anything; the frame becomes active when, and only when, the camera arrives.

### The engage radius follows from the precision policy

```
engage_radius = FLOAT32_SAFE_MAGNITUDE / FRAME_ENGAGE_MARGIN
```

For a `SystemFrame` that is `1e6 / 10 = 1e5 AU`, which is **0.485 pc**.

Both constants are engineering policy: `1e6` is the coordinate magnitude
beyond which float32 can no longer represent a unit step, and `10` is the
headroom kept for geometry drawn around the camera rather than the camera
alone. **0.485 pc is derived from that chosen precision budget** - it is not
a physical constant, and changing either number moves it.

What the derivation buys is that "close enough to enter the system" and
"close enough for AU coordinates to survive being narrowed to float32"
remain the same statement, instead of two thresholds that can drift apart.

### What the precision actually is at the boundary

At `1e5 AU`, float32 spacing is about `7.8e-3 AU`, or roughly
`1.2e6 km`. That is visually harmless there - inner-system geometry is far
below a pixel at that range - but it should not be described as high
scientific spatial precision. It is a *rendering* budget.

The measurement that matters to a renderer is screen-space error, and
`test_screen_space_error_stays_below_a_quarter_pixel` checks it directly:
project the float64 reference position and the float32 rendered coordinate
through the same camera, and require the difference to stay under a quarter
of a pixel for visible bodies.

### The approach

Camera position is held in absolute parsecs, float64, and re-expressed in
whichever frame is active. Waypoints are spaced evenly in *log distance*,
because the journey spans five orders of magnitude and the interesting part
is the last thousandth of it:

```
  step   distance (pc)        view    camera (local)  render
     0       43.796374    UNIVERSE          40 pc     ok
     1        2.216612    UNIVERSE       64.81 pc     ok
     2        0.112187      SYSTEM   2.314e+04 AU     ok
     3        0.005678      SYSTEM        1171 AU     ok
     4        0.000287      SYSTEM       59.27 AU     ok
     5        0.000015      SYSTEM           3 AU     ok
```

`to_render()` raises rather than degrading, so
`test_the_whole_approach_is_free_of_precision_loss` walking 48 waypoints is
a real proof, not a smoke test. The complementary test shows what the
engage radius is *for*: asking a `SystemFrame` to render a camera 20 pc away
raises `PrecisionError`.

## Selection

`rendering/picking.py` does an analytic ray-sphere intersection with a
minimum screen-space pick radius, so a two-pixel body is still clickable.

Two properties the legacy version lacked:

* **nothing behind the camera can be picked.** The old code took the
  perpendicular distance to an *infinite* line, which is symmetric about the
  viewer, so a star directly behind scored as well as one in front. The
  offset is now projected onto the view direction first.
* **the nearest body wins, not the nearest to the ray.** The old code took
  `argmin` over ray distance, so a large distant star that happened to lie
  nearer the ray centre beat the small planet actually in front of it.

`Selection` holds the **catalogue name**, never an array index or a render
handle. That is what makes it survive a level-of-detail change, a scene
rebuild or a frame transition - all of which replace the render primitives
but none of which change what the object is called.

## Labels

`SceneDescription.project_labels()` returns
`(label, x, y, depth, screen_radius)` for every labelled body in front of the
camera and inside the viewport.

It is strictly read-only. Label placement must never touch the numbers that
positioned the bodies, and
`test_labels_never_modify_scientific_coordinates` checks positions and radii
byte for byte after a projection pass.

Decluttering keeps the nearest of any overlapping pair - a crowded system
reads better with three legible labels than seven illegible ones - with one
exception: the **current selection is passed as a priority** and sorted
first, so it can never be the label that gets dropped.

`screen_radius` pushes text clear of the body, so a host star is not
labelled across its own disc.

## Level of detail

LOD is chosen from **projected screen size**, not a world-distance
threshold, so the same body at the same distance gets more detail in a
taller viewport or a narrower field of view:

```python
projected = (radius / distance) / tan(fov_y / 2) * viewport_height
```

The backend caches a mesh per subdivision level and groups planets by
`(material, LOD)`, so a six-planet system costs two instanced draws rather
than six. LOD changes geometry only: positions, display radii and
identifiers are asserted unchanged.

## What is not placed

TRAPPIST-1 has no published system distance, so it has no position in the
neighbourhood view and is left out of it rather than placed somewhere
convenient. The demo says so:

```
Hosts placed:      5
Hosts without a usable distance, therefore not placed: TRAPPIST-1
```

Its system view still works perfectly - the frame origin is the star, and
where the system sits in the galaxy never enters the local geometry.

## Acceptance criteria

| Criterion | Test |
|---|---|
| camera enters/leaves a system without precision loss | `test_the_whole_approach_is_free_of_precision_loss` |
| frame switches are explicit and type-safe | `test_frames_stay_type_safe_across_a_transition` |
| picking cannot select objects behind the camera | `test_nothing_behind_the_camera_can_be_picked` |
| selected identity survives LOD transitions | `test_selection_identity_survives_lod_transitions` |
| labels never modify scientific coordinates | `test_labels_never_modify_scientific_coordinates` |
| labels are decluttered | `test_labels_are_decluttered` |
| selected object label is always visible | `test_the_selected_label_is_always_visible` |
| LOD is based on projected size | `test_lod_follows_projected_size_not_world_distance` |
| production code does not import the legacy script | `test_no_production_module_imports_the_legacy_script` |
| all tests stay green | the full pytest suite |
| headless GL CI produces verified artifacts | the `opengl` job renders the approach and asserts the frames exist |
