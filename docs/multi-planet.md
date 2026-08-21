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
| **Kepler-11** | 6 | `a`, `e` and a transit epoch for every planet, but no `omega`. All six are *partially* constrained: observed timing, normalised orientation. |
| **TRAPPIST-1** | 7 | Inclinations only. No eccentricities, no epochs, **no system distance**. Everything must degrade gracefully. |
| **HD 219134** | 6 | *Mixed* completeness - 4 planets with epochs, 2 without. The case a uniform path gets wrong. |

## Phase provenance is per planet, and three-valued

A binary observed/assumed split is not enough. The interesting case is
neither: **the temporal anchor can be observed while the orbital
orientation used to read it is not.**

Kepler-11 is that case six times over. Every planet has a published
mid-transit time - a directly observed instant - but no published `omega`,
and converting a transit epoch to a mean anomaly goes through
`nu = pi/2 - omega`. Calling those "positioned from a published epoch"
overstates them; calling them "assumed" understates them.

`OrbitalElements.phase_at()` returns a `PhaseSolution` carrying the whole
story (`physics/phase.py`):

| Field | Values |
|---|---|
| `provenance` | `PERIASTRON_EPOCH`, `TRANSIT_EPOCH`, `TRANSIT_CONJUNCTION_NORMALIZED`, `MEAN_ANOMALY_AT_EPOCH`, `ASSUMED_ZERO_PHASE`, `UNKNOWN` |
| `anchor` | `OBSERVED`, `ASSUMED`, `NONE` |
| `mapping` | `DIRECT`, `CONJUNCTION_NORMALIZED`, `ARBITRARY_ZERO` |
| `omega_status` | the `Status` of the angle actually used |
| `status` | `CONSTRAINED`, `PARTIALLY_CONSTRAINED`, `ASSUMED`, `UNKNOWN` |

`status` is derived, not stored, so the rules stay in one place: a
periastron epoch needs no `omega` at all and cannot be weakened by a
missing one; a transit epoch read through a normalised `omega` is
`PARTIALLY_CONSTRAINED`; a stellar-reflex conversion counts as `DERIVED`
and is therefore still `CONSTRAINED`.

The three systems land in three different places, and HD 219134 shows all
three at once - which a uniform code path would have flattened:

```
  6 planet(s) by phase provenance:
     6  PARTIALLY_CONSTRAINED  timing observed, orbital orientation normalised for display
    via TRANSIT_CONJUNCTION_NORMALIZED: published transit time, argument of
                                        periastron normalised to 0 deg
```
```
  6 planet(s) by phase provenance:
     2  CONSTRAINED            constrained by a published epoch and orientation
     2  PARTIALLY_CONSTRAINED  timing observed, orbital orientation normalised for display
     2  ASSUMED                assumed: correct rate, arbitrary starting point
```

### The conjunction caveat

Inferior conjunction is *defined* by the argument of latitude
`u = omega + nu = pi/2`, so `nu_transit = pi/2 - omega` is exact for
conjunction - not an approximation.

What is approximate is equating conjunction with the instant of minimum
sky-projected separation. For an eccentric, non-edge-on orbit the two
differ by a term of order `e cos(omega) cos^2(i)`, which vanishes as
`i -> 90 deg`. `conjunction_offset_scale()` returns an upper bound, and the
solution reports it when non-zero. For Kepler-11 d (`e = 0.004`,
`i = 89.6 deg`) it is below 1e-6 radians - negligible, but named rather
than buried.

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
