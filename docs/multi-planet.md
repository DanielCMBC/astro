# Multi-planet `SystemFrame` rendering

Review section 15, the milestone approved once the orbital-semantics pass
was complete. Scaling from one planet to a whole system is where per-object
draw overhead, level of detail and label crowding first bite - and where a
uniform code path starts quietly lying about systems whose planets are not
uniformly well characterised.

```bash
python -m astro_explorer.app.system_demo --host Kepler-11
python -m astro_explorer.app.system_demo --host TRAPPIST-1 --frames 8
python -m astro_explorer.app.system_demo --host "HD 219134" --no-render
```

## The three test systems

They were chosen because they fail differently.

| System | Planets | What it exercises |
|---|---|---|
| **Kepler-11** | 6 | Complete: `a`, `e` and a transit epoch for every planet. All six are positioned from a published ephemeris. |
| **TRAPPIST-1** | 7 | Inclinations only. No eccentricities, no epochs, **no system distance**. Everything must degrade gracefully. |
| **HD 219134** | 6 | *Mixed* completeness - 4 planets with epochs, 2 without. The case a uniform path gets wrong. |

## Phase provenance is per planet, not per system

A planet with a published epoch is where the ephemeris says it is. One
without is being advanced at the correct *rate* from an arbitrary zero: a
picture of the motion, not a claim about tonight's sky. `placements()`
returns `(mean_anomaly, phase_is_assumed)` per planet, and the panel
separates them:

```
  6 planet(s): 6 positioned from a published epoch, 0 with an assumed phase, 0 not placed
```
```
  7 planet(s): 0 positioned from a published epoch, 7 with an assumed phase, 0 not placed
  An assumed phase advances the planet at the correct rate from an arbitrary
  zero: the motion is physical, the current position is not.
```

### Transit epochs without an argument of periastron

Kepler-11 publishes a mid-transit time for every planet but no `omega`.
Converting a transit epoch to a mean anomaly needs `omega`, because the true
anomaly at mid-transit is `pi/2 - omega`.

Rather than discard six usable epochs, the display normalisation `omega := 0`
is used - and that is not a fudge. It places the planet at **inferior
conjunction at the transit time**, which is exactly what was observed. What
stays unknown is the orbit's orientation *within* its plane, which is
already flagged as `ORIENTATION_PARTIAL`.

`test_at_its_own_transit_time_each_planet_is_at_inferior_conjunction` checks
all six land at `nu = 90 deg` to one part in 1e9.

## Batched orbit geometry

Six orbits are **one draw call**. All paths go into a single vertex buffer
and are drawn as indexed `LINES`:

* `LINE_STRIP` cannot do this - the strips would join end to end, drawing a
  spurious segment from one orbit's last point to the next orbit's first;
* GL 3.3 primitive restart is not exposed portably by ModernGL.

So segments are indexed explicitly, and
`test_batched_indices_never_join_two_orbits` asserts no index pair ever
straddles an orbit boundary.

Per-orbit style therefore has to travel **per vertex** rather than as a
uniform: colour as `vec4`, and a dash period where zero means solid. One
buffer carries dashed and solid orbits together.

## Per-system level of detail

`SceneDescription.assign_lod(camera, viewport_height)` picks an icosphere
subdivision from each body's projected screen size, and the backend caches a
mesh per level and groups planets by `(material, LOD)`.

| Level | Triangles |
|---|---|
| 0 | 20 |
| 1 | 80 |
| 2 | 320 |
| 4 | 5120 |

A six-planet system therefore costs **one instanced draw per (material,
LOD) group** - two for Kepler-11, whose planets span `rocky` and
`gas_giant` at a single LOD - never one per planet. LOD changes geometry
only: `test_lod_does_not_move_anything` asserts positions and display radii
are untouched.

## Labels

`SceneDescription.project_labels(camera, width, height)` returns
`(label, x, y, depth, screen_radius)` for every labelled body that is in
front of the camera and inside the viewport, sorted nearest-first.

Placement is computed there rather than in a shader, because a text atlas
would tie the renderer to a font and the same placements have to serve a Qt
overlay later. `rendering/labels.py` composites them with Pillow.

* `screen_radius` pushes text clear of the body, so a host star is not
  labelled across its own disc;
* collisions are resolved nearest-wins - a crowded system reads better with
  three legible labels than seven illegible ones.

## Physical time controls

`--frames N --periods K` steps forward over `K` periods of the **outermost**
planet, so the inner planets visibly lap the outer ones. Over one Kepler-11 g
period:

| Planet | Period (d) | Revolutions |
|---|---|---|
| Kepler-11 b | 10.304 | 11.49 |
| Kepler-11 c | 13.024 | 9.09 |
| Kepler-11 d | 22.684 | 5.22 |
| Kepler-11 e | 32.000 | 3.70 |
| Kepler-11 f | 46.689 | 2.54 |
| Kepler-11 g | 118.381 | 1.00 |

The camera is fixed across a sequence so the motion is the planets', not the
viewpoint's.

## Display-scale disclosure

The header states the effective factors, which are computed per system:
Kepler-11's star is drawn 6.14x and its planets 85.1x; TRAPPIST-1's star
12x and planets 69.1x. The factors differ because the star may never be
drawn wider than a fraction of the tightest periapsis, and a planet may
never be drawn larger than its star.

Review section 6 is enforced: the science packages cannot even name
`display_radius`, and changing the exaggeration leaves every orbital
position bit-for-bit identical.

## Graceful degradation

TRAPPIST-1 has no `sy_dist`, so its `SystemFrame` has no galactic origin.
It renders anyway - the origin is the star, and where the system sits in the
galaxy never enters the local geometry. All seven orbits are dashed, because
both the eccentricity and the node behind them are assumed, and the
star renders in its true M8-dwarf orange from a 2566 K effective
temperature.
