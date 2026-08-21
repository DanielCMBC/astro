# The one-star-one-planet vertical slice

The milestone is not breadth. It is one host system that is correct at every
stage, from the local snapshot to the pixels.

```
local snapshot  (data/reference_systems/vertical_slice_ps_default.csv)
    |
    v  data/schema.py
PlanetRecord / StarRecord          units + provenance on every value
    |
    v  physics/orbital_elements.py
OrbitalElements                    MEASURED / DERIVED / ASSUMED / UNKNOWN
    |
    v  physics/ephemeris.py, physics/kepler.py
M(t) = M0 + n(t - t0)  ->  M = E - e sin E  ->  E
    |
    v  physics/orientation.py
r_perifocal  ->  R_z(Omega) R_x(i) R_z(omega)  ->  r
    |
    v  coordinates/system_frame.py
FramedPosition in AU, float64, origin = the host star
    |
    v  rendering/scene_builder.py
RenderPlanet(position_local, radius_display, material_id)   float32
    |
    v  rendering/gl_backend.py
OpenGL 3.3 core: VAO / VBO / EBO / GLSL / instanced draws
```

## Why HD 80606 b

`e = 0.93183`. Apoapsis is 28 times periapsis, and the planet moves 28 times
faster at one end than the other. Every class of error this milestone is
meant to exclude produces a large, obvious discrepancy on this orbit and a
negligible one on a circular orbit:

| Mistake | Effect at e = 0.93 | Effect at e = 0.02 |
|---|---|---|
| `E ~ M + e sin M` instead of solving Kepler | position off by ~0.3 rad | invisible |
| advancing true anomaly instead of mean anomaly | speed wrong by 10x near periapsis | ~2% |
| wrong rotation order | orbit tilted into the wrong plane | often looks fine |
| chord instead of sector for swept area | 40% under-read at periapsis | negligible |

`WASP-39 b` and `K2-18 b` ride along as ordinary cases, and because their
orientation is *incomplete* they exercise the unknown-versus-zero policy from
the other direction.

## The data

`data/reference_systems/vertical_slice_ps_default.csv` was fetched once with
the documented default-solution query (`ps` filtered to `default_flag = 1`)
and committed. It is read offline; nothing in the slice touches the network.

HD 80606 b has **eight** rows in `ps`. That is exactly the situation roadmap
section 3.2 describes: eight published parameter sets, and
`drop_duplicates` would have picked whichever sorted first. The snapshot
carries the one the archive marks as default, from Pearson et al. (2022).

## Unknown Omega is not zero Omega

The longitude of the ascending node is not observable from transits or
radial velocity, and the archive publishes none for HD 80606 b. So:

```
elements.longitude_of_ascending_node.status  ->  Status.UNKNOWN
elements.longitude_of_ascending_node.value   ->  None
elements.orientation_known                   ->  False
```

`for_display()` returns a *separate* element set in which the node is `0.0`
with status `ASSUMED_FOR_VISUALIZATION`. The original record is untouched.
The report prints both facts:

```
  Inclination i:     89.24 +/- 0.01 deg
  Arg. periapsis w:  -58.89 deg
  Asc. node O:       UNKNOWN
                     display normalisation 0 deg [assumed for visualisation]
```

and the renderer draws that orbit **dashed**, because at least one element
behind it was substituted. The renderer is told only `dashed=True`; the
reason stays in the UI layer.

## Coordinate scale

The slice renders entirely inside a `SystemFrame`:

| | |
|---|---|
| origin | the host star, so the star is exactly `(0, 0, 0)` |
| unit | AU |
| CPU | float64 |
| GPU | float32, narrowed once, at `to_render()` |

A planet's position is the orbital vector from `physics/orientation.py`
**with no conversion applied at all**. That is the structural fix for the
prototype's `0.005` AU-to-parsec factor: there is no conversion left in
which to be wrong.

Mixing is prevented by type, not by convention. A `FramedPosition` knows its
frame, and combining two frames raises:

```python
system.at([0.05, 0, 0]) + universe.at([66.47, 0, 0])
# FrameMismatchError: refusing to combine a position in HD 80606 frame [AU]
# (AU) with one in Sun frame [pc] (pc); convert explicitly first
```

Conversions go through absolute parsecs in float64, so no factor is ever
applied twice. `to_render()` raises rather than silently emitting a
degenerate float32 coordinate.

The system's galactic position (66.47 pc) is recorded on the frame but never
enters the local geometry, so a host with an unusable parallax still renders.

## Verified numbers

Every figure below is produced by the pipeline and asserted in the tests.

| Quantity | Value | Check |
|---|---|---|
| `\|r\|` at `T_peri` | 0.03137865 AU | `= a(1-e)` to 1e-9 |
| `\|r\|` at `T_peri + P/2` | 0.8892 AU | `= a(1+e)` |
| apoapsis / periapsis | 28.4 | `= (1+e)/(1-e)` |
| speed at periapsis | 239.925 km/s | `= sqrt(mu(1+e)/(a(1-e)))` |
| speed ratio | 28.4 | `= (1+e)/(1-e)` |
| specific energy | -3.387834861e-04 | `= -mu/2a`, rel. error **1.9e-15** |
| orbit normal tilt | 89.24 deg | equals the published inclination |
| Kepler III residual | -0.197% | published `a` vs `a` from `P` and `M*` |

### Kepler's second law, from the real pipeline

Twelve equal time steps of 9.286 d:

```
  step        r (AU)        arc (AU)     area (AU^2)
     0      0.031379        0.463816      0.02012659
     3      0.739209        0.099809      0.02012943
     6      0.889221        0.048984      0.02012944
     9      0.739209        0.144222      0.02012943
  arc length varies by 9.5x; swept area varies by 5.26e-05 relative
```

The planet covers nearly ten times as much ground per unit time near
periapsis, and sweeps the same area doing it.

## Running it

```bash
python -m astro_explorer.app.slice_demo
python -m astro_explorer.app.slice_demo --host WASP-39 --out renders/
python -m astro_explorer.app.slice_demo --no-render      # text only
```

The renderer needs `pip install -e ".[render]"`. Without it the demo prints
the full scientific report and says why it skipped the picture.

## What the tests cover

| File | Subject |
|---|---|
| `tests/physics/test_orientation.py` | the transform, case by case (43 tests) |
| `tests/physics/test_conservation.py` | second law, energy, vis-viva, angular momentum (45) |
| `tests/coordinates/test_system_frame.py` | frames, mixing, precision (27) |
| `tests/regression/test_vertical_slice.py` | the chain end to end (36) |
| `tests/regression/test_gl_backend.py` | real GL 3.3, pixels (12) |
| `tests/regression/test_architecture.py` | the golden rule, mechanically (27) |

The orientation tests cross-check the matrix composition against the
**expanded scalar equations** of the formula reference, section 11, written
out independently in the test file so a mistake in the matrices cannot hide
behind the same mistake in the test.

## What this milestone did not do

Deliberately, per the instruction to stop broadening: no Gaia, no SIMBAD, no
texture catalogue, no N-body, no stellar-neighbourhood view. `UniverseFrame`
and `PlanetFrame` exist and are tested, but only `SystemFrame` is used, so
that entering a system will later be a frame *transition* rather than a
scale factor bolted on.
