# Exoplanet Scientific Suite

[![CI](https://github.com/DanielCMBC/astro/actions/workflows/ci.yml/badge.svg?branch=3D-test)](https://github.com/DanielCMBC/astro/actions/workflows/ci.yml)

An offline-first exoplanet scientific explorer: catalogue-accurate data,
orbital mechanics, stellar physics, spectroscopy and explicit provenance.

This branch contains the corrected scientific baseline, the reusable
physics/data core, and the **actively developed modern 3D/OpenGL engine** -
a GL 3.3 core-profile renderer with a validated one-planet vertical slice
and multi-planet system rendering. Development follows
`EXOPLANET_2D_FIXES_AND_3D_OPENGL_ROADMAP.md`.

## The one rule

Every scientific value carries its status:

```
MEASURED                    published in a catalogue or paper
DERIVED                     computed here by a documented relation
ASSUMED_FOR_VISUALIZATION   a placeholder so something could be drawn
UNKNOWN                     not known, and not substituted
```

A missing semimajor axis is derived from Kepler's third law or left unknown;
it never becomes 1 AU. A missing eccentricity stays unknown; the circular
orbit you see is labelled as an assumption. An unusable parallax stays
unknown; it never becomes a placeholder distance.

## The 3D prototype

`stellar_navigator_3d.py` is the original Pygame + PyOpenGL prototype that
gave this branch its name: a flyable 3D map of exoplanet host stars built on
Gaia, SIMBAD and the NASA archive.

It still runs, and its own documentation is preserved at
[`docs/legacy-3d-prototype.md`](docs/legacy-3d-prototype.md).

**It is not scientifically authoritative and nothing under `src/` may
import it** - a test enforces that. Sixteen audited defects are catalogued
in [`docs/legacy-3d-prototype.md`](docs/legacy-3d-prototype.md), each
verified at a line number and each with an assertion proving the current
code does not repeat it: a first-order Kepler approximation, a `0.005`
AU-to-parsec scale factor, coplanar `[x, y, 0]` orbits, fixed-function
OpenGL, invented missing-data defaults, an unusable parallax becoming a
billion parsecs, seconds mixed with days in the same anomaly, and gas-giant
classification by orbital distance.

What it got right was the interaction model, and that is carried forward.

## Setup

```bash
conda create --name astrodata python=3.11
conda activate astrodata
pip install -e .
```

Optional extras:

```bash
pip install -e ".[render]"     # ModernGL + pygame, for the 3D engine
pip install -e ".[dynamics]"   # REBOUND, for the optional N-body mode
pip install -e ".[ui]"         # PySide6 + pyqtgraph
pip install -e ".[dev]"        # pytest
```

Windows users can run `dependencies.bat`, which wraps the same install.

## Run

```bash
python exoplanet_analyzer.py
```

or, after installing, `astro-explorer-2d`.

The application reads from a validated local snapshot and refreshes in the
background. With no network it stays fully usable; a failed synchronisation
never disturbs the working snapshot.

## Tabs

| Tab | Shows |
|---|---|
| Overview | parameters with their status, classification, Kepler-3 residual, distance |
| Orbit | physical-time animation; an assumed orbit is dashed and explained |
| HR Diagram | luminosity against effective temperature |
| Temperature-Radius | the original program's plot, under its correct name |
| Black Body | Planck curve and Wien peak, labelled an ideal approximation |
| Atmospheric Spectra | one series per measurement, named by instrument and paper |
| Molecular Evidence | detections grouped by how strong the evidence actually is |
| Data & Provenance | source, solution policy, snapshot version, resource paths |

## The 3D vertical slice

One host system, correct at every stage from the local snapshot to the
pixels. The reference target is **HD 80606 b** (`e = 0.93183`), chosen
because an extreme eccentricity exposes orbital errors that a near-circular
orbit hides.

```bash
python -m astro_explorer.app.slice_demo            # report + render
python -m astro_explorer.app.slice_demo --no-render # report only
```

Verified against independently derivable values: `|r|` at periastron equals
`a(1-e)` to 1e-9, periapsis speed 239.925 km/s matches
`sqrt(mu(1+e)/(a(1-e)))`, and the specific orbital energy matches `-mu/2a`
to a relative 1.9e-15. See [`docs/vertical-slice.md`](docs/vertical-slice.md).

## Exploring

```bash
python -m astro_explorer.app.explorer_demo            # neighbourhood -> system
python -m astro_explorer.app.explorer_demo --no-render
```

Fly through the host stars, select one, enter its system. Which reference
frame is active is not a mode the user toggles - it is the finest frame
whose engage radius contains the camera, and that radius is derived from the
float32 limit rather than chosen. So "close enough to enter the system" and
"close enough for AU coordinates to survive the GPU" are the same statement.
See [`docs/explorer.md`](docs/explorer.md).

## Multi-planet systems

```bash
python -m astro_explorer.app.system_demo --host Kepler-11
python -m astro_explorer.app.system_demo --host TRAPPIST-1 --frames 8
```

Six orbits cost one draw call; planets cost one instanced draw per
(material, LOD) group. Each planet's phase provenance is tracked separately,
so a system with published epochs and one without are never reported the
same way - and a transit epoch read through a normalised `omega` is
reported as neither. See [`docs/multi-planet.md`](docs/multi-planet.md).

## Layout

```
exoplanet_analyzer.py          launcher
src/astro_explorer/
    provenance.py              Parameter and Status
    physics/                   constants, Kepler, orientation, state vectors,
                               elements, ephemeris, stellar, radiation
    coordinates/               ICRS/Galactic frames, unit bridges,
                               Universe/System/Planet frames, floating origin
    spectroscopy/              IPAC reader, Spectrum, molecular evidence
    classification/            conventional scheme + draft physical vector
    data/                      catalogue queries, schema, validation, local store
    assets/                    resource manager, provenance manifest, procedural materials
    rendering/                 meshes, camera, GLSL shaders, picking,
                               scene contract, OpenGL 3.3 backend
    app/                       state and controller
    ui/                        Tkinter shell and matplotlib plots
tests/                         540 tests
docs/                          architecture, physics, provenance, assets, roadmap status
legacy/                        the original single-file program, preserved
tables/                        733 NASA IPAC atmospheric spectra
molecular_evidence.csv         detection evidence with provenance
```

## Data notes

* Planet and system parameters come from the NASA Exoplanet Archive. The
  record-selection policy is explicit: `pscomppars` is maximally populated
  but mixes references across columns, while `ps` filtered to
  `default_flag = 1` gives one self-consistent published solution. The
  active policy is shown in the Data tab.
* Atmospheric spectra are read from `tables/*.tbl` with
  `astropy.table.Table.read(..., format="ascii.ipac")`, addressing columns by
  name: `CENTRALWAVELNG` is the wavelength, `PL_TRANDEP` the transit depth,
  `BANDWIDTH` the band width (an x error bar, not the signal), and
  `PL_TRANDEPERR1`/`PL_TRANDEPERR2` the asymmetric uncertainties.
* Spectra from different instruments, facilities, epochs or publications are
  never merged.
* Textures and colours generated from physical parameters are labelled
  "actual appearance unknown". No exoplanet surface has been imaged.

## Tests and CI

```bash
pytest
```

Every push runs the suite on GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), in two jobs:

| Job | What it does |
|---|---|
| **Tests** | Python 3.11 and 3.12: architecture/golden-rule tests first, then the full suite, then an offline check that the app builds every reference system from the committed snapshot with no network. |
| **OpenGL 3.3 core** | Mesa software rasteriser under Xvfb. Verifies a real GL 3.3 core context, compiles all five shader programs, runs the GL and multi-planet tests, and renders both demos - uploading the frames as build artifacts. |

The OpenGL job runs `scripts/verify_gl.py` *before* the GL tests, and that
script fails hard rather than skipping. Without it a runner with no driver
would skip every OpenGL test and still report green.

Beyond the physics, the suite enforces the architecture: the renderer may
not import the data layer, render primitives may not carry scientific
fields, no module outside `constants.py` may hard-code a physical constant,
and the incorrect AU-to-parsec factor may not reappear. It also checks the
CI definition itself - that the workflow triggers on every branch, that the
architecture tests run before the rest, and that the OpenGL job cannot pass
by skipping.

## Build

```bash
pyinstaller exoplanet_analyzer.spec
```

The spec generates its data files from `DECLARED_RESOURCES`, so declaring a
resource in `assets/manager.py` is enough to get it bundled.

## Further reading

* [`docs/vertical-slice.md`](docs/vertical-slice.md)
* [`docs/orbital-semantics.md`](docs/orbital-semantics.md)
* [`docs/multi-planet.md`](docs/multi-planet.md)
* [`docs/explorer.md`](docs/explorer.md)
* [`docs/legacy-3d-prototype.md`](docs/legacy-3d-prototype.md)
* [`docs/architecture.md`](docs/architecture.md)
* [`docs/physics.md`](docs/physics.md)
* [`docs/data-provenance.md`](docs/data-provenance.md)
* [`docs/assets.md`](docs/assets.md)
* [`docs/original-2d-behaviour.md`](docs/original-2d-behaviour.md)
* [`docs/roadmap-status.md`](docs/roadmap-status.md)
