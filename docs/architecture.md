# Architecture

The program is layered so that **the renderer never owns scientific truth**
(roadmap section 6). Data flows in one direction only:

```
NASA / Gaia / SIMBAD / local snapshot
                |
                v
         validated data layer          src/astro_explorer/data
                |
                v
          scientific models            physics / coordinates / spectroscopy
                |
                v
         application state             app/state.py
          +-----+-----+
          v           v
       UI/plots    OpenGL renderer     ui/ and rendering/
```

## Package map

| Package | Responsibility | May import |
|---|---|---|
| `provenance` | `Parameter`, `Status`; the vocabulary everything else speaks | astropy only |
| `physics` | constants, Kepler solver, orbital elements, ephemeris, stellar physics, radiation | `provenance` |
| `coordinates` | ICRS/Galactic frames, unit bridges, floating origin | `provenance` |
| `spectroscopy` | IPAC parsing, `Spectrum`, molecular evidence | `provenance` |
| `classification` | conventional scheme, draft physical vector | `provenance` |
| `data` | catalogue queries, schema, validation, local store | everything above |
| `assets` | resource resolution, provenance manifest, procedural materials | `provenance` |
| `rendering` | meshes, camera, shaders, picking, scene contract, GL backend | `physics` (scene_builder only) |
| `app` | state and controller | everything |
| `ui` | Tkinter shell and matplotlib plots | `app` |

These rules are enforced by tests in `tests/regression/test_architecture.py`:

* `rendering` never imports `data`;
* only `rendering/scene_builder.py` may import `physics` or `provenance`.
  Every other rendering module works on plain numbers;
* `rendering/gl_backend.py` additionally may not import `app`, and contains
  no fixed-function GL call;
* `physics/orientation.py` imports neither astropy nor `provenance`, so the
  3D transform stays testable as pure numerics;
* `FramedPosition` never leaks past `scene_builder`.

## The renderer's input contract

`rendering/renderer.py` defines the only types a GL backend accepts:
`RenderStar`, `RenderPlanet`, `RenderOrbit`, `RenderZone`, `RenderGuide`,
`SceneDescription`. They carry positions in display units, a radius, a
material id, a colour and - for a guide - a stroke style.

The two overlay primitives are where the rule earns its keep. `RenderZone`
is two rings and two colours, with no luminosity or temperature to
recompute a habitable-zone boundary from. `RenderGuide` is a polyline with
a `SOLID` or `DASHED` stroke, with no inclination, `omega` or `Omega` to
re-derive an orientation from - which would be a second reading of the
catalogue's conventions, free to disagree silently with the orbit it is
drawn against.

They deliberately have **no** field for eccentricity, semimajor axis,
anomaly, physical distance, parameter status or molecular detection. The
renderer cannot decide any of those, because it is never told them. A test
asserts that those field names stay absent.

`RenderPlanet` and `RenderStar` reject non-finite positions at construction,
so an unknown parameter can never reach the GPU as a NaN.

## Coordinate frames

A single float32 space cannot hold both parsec- and kilometre-scale
geometry, so geometry lives in one of three frames (roadmap section 9).
`coordinates/system_frame.py` makes these first-class objects:

| Frame | Origin | Unit | CPU | GPU |
|---|---|---|---|---|
| `UniverseFrame` | Sun, or a floating origin | pc | float64 | float32 |
| `SystemFrame` | the host star | AU | float64 | float32 |
| `PlanetFrame` | the planet | km | float64 | float32 |

Mixing is prevented by **type**, not by convention. A `FramedPosition`
carries its frame, and combining two frames raises `FrameMismatchError`
rather than producing a plausible-looking wrong number:

```python
system.at([0.05, 0, 0]) + universe.at([66.47, 0, 0])   # FrameMismatchError
```

Conversions go one way only, through absolute parsecs in float64
(`ReferenceFrame.convert`), so no factor is ever applied twice or to the
wrong quantity. Entering a system is a frame *transition*
(`UniverseFrame.enter_system`), not a scale factor.

`ReferenceFrame.to_render` is the single boundary at which float32 appears,
and it raises `PrecisionError` rather than silently emitting a degenerate
coordinate.

In `SystemFrame` the star is exactly `(0, 0, 0)` and a planet's position is
the orbital vector from `physics/orientation.py` **unchanged** - no
conversion is applied at all. That absence is the structural fix for the
prototype's `0.005` AU-to-parsec factor.

`coordinates/floating_origin.py` keeps the earlier `Scale` / `SceneGraph`
API for the galaxy-scale rebasing story; `system_frame.py` is what the
vertical slice uses.

## Time

`TimeController` offers three models that the UI must never conflate:

* `TimeMode.REAL` - the ephemeris says where the planet is;
* `TimeMode.SCALED` - simulated time at N days per second;
* `TimeMode.NORMALIZED` - the educational mode where every orbit takes the
  same wall-clock time. It is labelled "not physical" wherever it appears
  and it is not the default.

`mean_anomaly` returns `(anomaly, phase_is_assumed)`. A `None` anomaly means
the orbit may be drawn but the planet's position along it is not defined;
callers must not quietly place it at periapsis.

## Offline-first data

Startup reads `CatalogRepository.load()`. Synchronisation is an explicit,
optional action: it downloads into staging, validates columns, units, row
counts and identifiers, hashes the result, and only then atomically replaces
the active snapshot. Any failure leaves the previous snapshot in place and
the program fully usable with the network disabled.

## Adding a new scientific quantity

1. Return a `Parameter` from the physics or data layer, with a `provenance`
   string and the right `Status`.
2. If it can be derived, derive it and mark it `DERIVED`; if it cannot,
   return `unknown(...)`. Never substitute a plausible value.
3. If a renderer needs a number and the parameter is unknown, either exclude
   the object or use `for_display()`, which tags substitutions
   `ASSUMED_FOR_VISUALIZATION`.
4. Surface it through `describe()` so the UI shows its status.
