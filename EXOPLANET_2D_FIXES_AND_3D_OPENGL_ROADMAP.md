# Exoplanet Scientific Suite — 2D Correction & 3D/OpenGL Development Roadmap

**Repository:** https://github.com/DanielCMBC/astro  
**2D branch:** `main`  
**3D branch:** `3D-test`  
**Recommended active development branch:** `3D-test`  
**Primary goal:** transform the original educational 2D exoplanet analyzer into a scientifically defensible, offline-first, Python desktop application inspired by NASA Eyes on Exoplanets, with real 3D rendering, accurate orbital mechanics, astronomical coordinate systems, spectra, provenance, and optional N-body dynamics.

---

## 1. Executive decision: which branch should be used?

The repository currently has exactly two branches:

- `main`
- `3D-test`

### `main` — preserve as the corrected 2D scientific baseline

`main` contains the current 2D application centered on:

- `exoplanet_analyzer.py`
- NASA Exoplanet Archive data
- Tkinter
- Matplotlib
- local atmospheric `.tbl` files
- `planet_molecules.csv`
- `atmospheric_signatures.json`
- local CSV/Feather data
- PyInstaller configuration

This branch should **not become the long-term 3D engine**.

Its purpose should be:

1. correct the scientific mistakes in the original prototype;
2. document what the original program did;
3. add validation tests for physics/data parsing;
4. serve as a reference implementation for plots and calculations;
5. remain a stable historical/educational baseline.

### `3D-test` — active development branch

`3D-test` already contains:

- `stellar_navigator_3d.py`
- Pygame
- PyOpenGL
- Gaia/Astroquery experiments
- 3D stellar positions
- star selection
- a camera
- rudimentary planetary-system rendering
- billboards and GLU spheres
- HR diagram generation
- blackbody spectrum generation
- 2D orbit display inside the information panel

Therefore, **future development should happen on `3D-test`**, not `main`.

However, the present `3D-test` code should be regarded as a **prototype**, not the foundation to extend indefinitely. The next major task should be refactoring its useful concepts into a modular architecture.

### Suggested long-term Git strategy

A clean development strategy would be:

```text
main
│
├── corrected 2D reference
│
└── tagged releases such as:
    v0.2-2d-final
    v0.2.1-science-fixes

3D-test
│
└── active experimental 3D/OpenGL development
```

Once the 3D architecture becomes stable, consider renaming or superseding `3D-test` with something clearer such as:

```text
develop
```

or:

```text
3d-engine
```

Do **not** merge the 3D prototype into `main` until the scientific core and architecture are stable.

---

# 2. What the original 2D application already gets right

The 2D program is not something that should be discarded. It established several useful concepts.

Current features include:

- NASA Exoplanet Archive retrieval;
- host-star selection;
- exoplanet selection;
- 2D orbital visualization;
- eccentric Keplerian orbit geometry;
- Hertzsprung-Russell-style stellar plotting;
- stellar blackbody spectrum using Planck's law;
- Wien peak calculation;
- local atmospheric spectroscopy files;
- molecule/evidence lookup;
- cached data;
- PyInstaller experimentation.

These should all survive conceptually.

The main change is that they need to be moved from a single GUI-oriented class into reusable scientific modules.

---

# 3. Critical corrections required in the 2D version (`main`)

These corrections should be completed even if the 2D application is no longer developed visually, because the fixed calculations can become regression tests for the 3D program.

---

## 3.1 CRITICAL — atmospheric spectrum parser uses the wrong columns

### Current behavior

`exoplanet_analyzer.py` manually reads `.tbl` rows and assumes:

```python
parts[0] -> wavelength
parts[1] -> transit depth
parts[2] -> error
```

However, the IPAC atmospheric table included in the repository begins with columns similar to:

```text
CENTRALWAVELNG
BANDWIDTH
PL_TRANDEP
PL_TRANDEPERR1
PL_TRANDEPERR2
...
```

Example:

```text
tables/55_Cnc_e_3.10924_3673_1.tbl
```

Therefore the current program can plot:

```text
CENTRALWAVELNG vs BANDWIDTH
```

instead of:

```text
CENTRALWAVELNG vs PL_TRANDEP
```

and may incorrectly treat `PL_TRANDEP` as the error bar.

### Required fix

Do not parse IPAC tables using positional whitespace splitting.

Use Astropy:

```python
from astropy.table import Table

table = Table.read(path, format="ascii.ipac")

wavelength = table["CENTRALWAVELNG"]
bandwidth = table["BANDWIDTH"]
transit_depth = table["PL_TRANDEP"]
err_plus = table["PL_TRANDEPERR1"]
err_minus = table["PL_TRANDEPERR2"]
```

### Requirements

- retain asymmetric uncertainty;
- preserve table metadata;
- preserve instrument;
- preserve facility;
- preserve spectrum type;
- preserve publication/reference;
- never merge different spectra simply because they belong to the same planet.

### Priority

**P0 — scientific correctness**

---

## 3.2 CRITICAL — arbitrary duplicate removal from NASA `ps`

The 2D program queries the NASA Planetary Systems table:

```sql
select ... from ps
```

and later performs:

```python
drop_duplicates(subset=["pl_name"])
```

The `ps` table may contain multiple rows representing different published parameter sets for the same planet.

Taking one arbitrary row after sorting by hostname can mix the application's scientific state with whichever reference happens to survive.

### Required design

Create two explicit data concepts:

#### Default/reference solution

Use a NASA-designated default solution or deliberately selected internally consistent reference.

#### Composite solution

Use `pscomppars` when the goal is a maximally populated composite record.

Never silently treat both concepts as equivalent.

### Future model

```python
class Parameter:
    value: float | None
    unit: str
    error_plus: float | None
    error_minus: float | None
    provenance: str
    reference: str | None
    status: str
```

Possible statuses:

```text
MEASURED
DERIVED
ASSUMED_FOR_VISUALIZATION
UNKNOWN
```

### Priority

**P0**

---

## 3.3 CRITICAL — missing semimajor axis must not become 1 AU

Current logic includes the equivalent of:

```python
pl_orbsmax.fillna(1.0)
```

This creates false orbital data.

### Required behavior

If semimajor axis is unavailable:

1. try to derive it from period and stellar mass using Kepler's third law;
2. label the value as `DERIVED`;
3. propagate uncertainty when possible;
4. if it cannot be derived, retain `None/NaN`;
5. display an unavailable state rather than inventing 1 AU.

### Priority

**P0**

---

## 3.4 CRITICAL — missing eccentricity must not automatically mean e = 0

The current 2D implementation maps unknown eccentricity to zero.

This is acceptable only as a clearly labeled rendering assumption.

### Correct behavior

```text
Published e available
    -> MEASURED

No published e, but visualization requires an orbit
    -> optionally assume e = 0
    -> mark ASSUMED_FOR_VISUALIZATION

No need to visualize
    -> UNKNOWN
```

The UI should distinguish:

```text
Eccentricity: unknown
Visual orbit assumption: circular
```

### Priority

**P0**

---

## 3.5 HIGH — orbit animation time is normalized, not physical

The 2D animation advances mean anomaly by frame number:

```text
100 frames
```

so every planet completes an orbit in approximately the same visual duration.

That is useful for demonstration but not a physical clock.

### Required implementation

Use:

\[
n = \frac{2\pi}{P}
\]

\[
M(t)=M_0+n(t-t_0)
\]

then solve:

\[
M = E - e\sin E
\]

and transform to true anomaly or directly to orbital-plane coordinates.

Provide two explicit time modes:

```text
REAL / PHYSICAL TIME
SIMULATION TIME SCALE
NORMALIZED EDUCATIONAL ORBIT
```

Never confuse them.

### Priority

**P1**

---

## 3.6 HIGH — the current HR diagram is not a classical HR diagram

The 2D application plots:

```text
effective temperature
vs
stellar radius
```

That is useful, but it is better described as a **stellar temperature-radius diagram**.

A classical HR diagram uses:

```text
luminosity or absolute magnitude
vs
effective temperature / spectral type / color
```

### Required fix

Retain both.

#### HR Diagram

\[
T_{\rm eff} \quad vs \quad L_\star/L_\odot
\]

#### Stellar Physical Diagram

\[
T_{\rm eff} \quad vs \quad R_\star/R_\odot
\]

If luminosity must be derived:

\[
L = 4\pi R^2 \sigma T^4
\]

mark it as `DERIVED`.

### Priority

**P1**

---

## 3.7 MEDIUM — blackbody should be explicitly labeled as an approximation

The current Planck-law implementation is conceptually good.

Keep:

\[
B_\lambda(T)=
\frac{2hc^2}{\lambda^5}
\frac{1}{e^{hc/\lambda kT}-1}
\]

and Wien's displacement law.

But the UI should say:

```text
Ideal blackbody approximation
```

because a real stellar spectrum contains line absorption and atmosphere-dependent structure.

Future modes could include:

```text
Ideal blackbody
Observed stellar spectrum
Synthetic stellar atmosphere
```

### Priority

**P2**

---

## 3.8 HIGH — scientific constants should come from trusted libraries

The program currently defines constants directly.

For the production scientific core, prefer:

```python
from astropy.constants import h, c, k_B, G, sigma_sb
```

and Astropy units.

This prevents unit mistakes and provides traceability.

### Priority

**P1**

---

## 3.9 HIGH — units must become explicit

Current calculations frequently depend on implicit knowledge such as:

- AU;
- parsecs;
- days;
- solar radii;
- Jupiter masses;
- Earth masses.

Production calculations should use `astropy.units`.

Example:

```python
a = 0.05 * u.au
period = 3.2 * u.day
mass = 1.0 * u.M_sun
```

Convert to unitless GPU values only at the rendering boundary.

### Priority

**P1**

---

## 3.10 MEDIUM — dependency installation is inconsistent

`dependencies.bat` installs:

```text
pandas
numpy
requests
matplotlib
tenacity
```

but the application also requires SciPy, and the repository documentation mentions additional packages.

### Required fix

Replace ad-hoc installation scripts as the authoritative dependency source with:

```text
pyproject.toml
```

Optionally retain the `.bat` file only as a convenience wrapper.

### Priority

**P2**

---

## 3.11 MEDIUM — PyInstaller does not package scientific assets

`exoplanet_analyzer.spec` currently uses:

```python
datas=[]
```

but the application expects local files such as:

- `planet_molecules.csv`
- atmospheric tables
- future shaders
- textures
- databases

### Required fix

Implement a resource manager and bundle declared resources properly.

Do not rely on:

```python
os.walk(".")
```

or the process working directory.

### Priority

**P2**

---

## 3.12 MEDIUM — molecule detections need stronger provenance

`planet_molecules.csv` already contains useful reference URLs.

The future system should expand this into a structured evidence table:

```text
planet
molecule
detection_status
instrument
facility
publication
DOI/ADS URL
retrieval_method
confidence
date
notes
```

Avoid presenting a molecule as definitively present if the scientific literature only reports tentative evidence.

### Priority

**P2**

---

# 4. Critical corrections already required in `3D-test`

The existing 3D branch is a useful experiment, but several parts must not be promoted into the final engine unchanged.

---

## 4.1 CRITICAL — Kepler equation is not actually solved in the 3D branch

Current `SystemRenderer.calculate_planet_position()` uses approximately:

```python
eccentric_anomaly = mean_anomaly + ecc * np.sin(mean_anomaly)
```

This is only a low-order approximation.

For small eccentricity it can look plausible.

For larger eccentricities it becomes increasingly inaccurate.

### Required implementation

Solve:

\[
E-e\sin E=M
\]

using:

- Newton-Raphson;
- a safeguarded iterative method;
- or a vectorized Kepler solver.

The physics layer should own this calculation.

The renderer must not solve orbital mechanics itself.

### Priority

**P0**

---

## 4.2 CRITICAL — AU-to-parsec scaling in `3D-test` is incorrect

The current code scales orbital coordinates with:

```python
* 0.005
```

and comments:

```text
Scale AU to parsecs
```

But:

\[
1\,AU \approx 4.8481368\times10^{-6}\,pc
\]

Therefore `0.005` pc per AU is wrong by roughly three orders of magnitude.

This makes planetary systems enormously oversized compared with their stellar positions.

### Correct solution

Do **not** express planet-local and galaxy-scale geometry directly in one global coordinate system.

Use hierarchical/floating origins.

#### Galaxy frame

```text
origin: Sun or Galactocentric frame
unit: pc
CPU: float64
```

#### Stellar system frame

```text
origin: selected host star
unit: AU
CPU physics: float64
GPU local rendering: float32
```

#### Planet frame

```text
origin: selected planet
unit: km or planetary radii
```

### Priority

**P0**

---

## 4.3 CRITICAL — 3D planetary orbits are currently still 2D

Current orbital coordinates are:

```python
[x, y, 0.0]
```

So every system is coplanar in the XY plane.

The final system needs orbital-element transformations.

Start in the perifocal/orbital plane:

\[
\mathbf{r}_{pf}
\]

then transform:

\[
\mathbf r =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf r_{pf}
\]

where:

- \(i\) = orbital inclination;
- \(\omega\) = argument of periapsis;
- \(\Omega\) = longitude of ascending node.

### Important exoplanet limitation

For many exoplanets, \(\Omega\) is unknown.

Do not invent observational certainty.

The UI should state:

```text
Inclination: measured / unknown
Argument of periapsis: measured / unknown
Ascending node: unknown
Display normalization: Ω = 0°
```

### Priority

**P0**

---

## 4.4 CRITICAL — missing data defaults create fictional systems

The 3D prototype currently falls back to values equivalent to:

```text
a = 1 AU
e = 0
period = 365.25 days
```

when orbital values are missing.

This creates Earth-like fictional orbits.

Use the same `MEASURED / DERIVED / ASSUMED / UNKNOWN` policy described for the 2D correction.

### Priority

**P0**

---

## 4.5 HIGH — time uses Unix wall clock without orbital epoch

The prototype derives orbital motion from a scaled `time.time()` value.

A physically meaningful position needs an epoch such as:

```text
time of periastron
transit epoch
mean longitude / mean anomaly at epoch
```

Without phase information, the shape and orbital speed can be shown, but the planet's absolute current location cannot necessarily be claimed.

### Required state

Distinguish:

```text
ORBIT SHAPE KNOWN
ORBIT PHASE CONSTRAINED
CURRENT POSITION COMPUTABLE
DISPLAY PHASE ASSUMED
```

### Priority

**P1**

---

## 4.6 HIGH — parallax failure should never become one billion parsecs

The 3D prototype currently maps non-positive parallax to a huge placeholder distance.

That produces fake coordinates.

### Correct behavior

If astrometric distance is unreliable:

- use a trusted catalog distance when available;
- retain missing/uncertain values;
- exclude the object from geometry that requires distance;
- never push the object to an arbitrary artificial location.

Astropy `Distance`/`SkyCoord` should be preferred.

### Priority

**P1**

---

## 4.7 HIGH — raw spherical coordinate conversion should be replaced by Astropy frames

The prototype manually converts RA/Dec/distance to Cartesian coordinates.

The math is not inherently wrong, but production astronomy should rely on explicit frames.

Use:

```python
SkyCoord(..., frame="icrs")
```

and transformations among:

```text
ICRS
Galactic
Galactocentric
custom host-system frame
```

This also simplifies future inclusion of:

- proper motion;
- radial velocity;
- epoch transformations.

### Priority

**P1**

---

## 4.8 HIGH — current OpenGL code is legacy/fixed-function OpenGL

The current branch uses constructs including:

```text
glBegin / glEnd
glMatrixMode
glTranslatef
glRotatef
GLU quadrics
client-state vertex arrays
```

These are useful for learning and proof-of-concept work, but they are not the architecture to build the final renderer around.

### Production direction

Use a modern core-profile pipeline:

```text
VAO
VBO
EBO
vertex shaders
fragment shaders
uniform buffers
texture objects
framebuffers
instancing
```

Recommended Python interface:

```text
ModernGL
```

PyOpenGL can remain useful if the objective is learning the raw API, but the shipping renderer should preferably use ModernGL or a similarly clean modern wrapper.

### Priority

**P1**

---

## 4.9 HIGH — current "textures" are generated colored circles, not physical planet materials

The present `TextureManager` creates simple colored Pygame circles and applies them to spheres/billboards.

That is completely appropriate for the prototype.

The final application needs a provenance-aware material system.

Possible asset classes:

```text
OBSERVED
NASA_CONCEPT
SCIENTIFIC_PROCEDURAL
GENERIC_CLASS
```

Never imply that an artist concept is a photographed exoplanet surface.

### Priority

**P1**

---

## 4.10 MEDIUM — star color mapping is currently only a visual approximation

The existing branch maps Gaia `BP-RP` to a simple RGB heuristic.

For production rendering, distinguish:

```text
scientifically-derived display color
stylized visibility-enhanced color
```

A star's perceived/display RGB can be estimated from effective temperature or calibrated color transformations, but tone mapping and display gamut mean it is still a visualization.

### Priority

**P2**

---

## 4.11 MEDIUM — information plots are rendered to PNG surfaces every time

The 3D prototype renders Matplotlib figures to an in-memory PNG and converts them into Pygame surfaces.

This works, but it is not ideal for responsive interactive plots.

Potential future alternatives:

- PyQtGraph;
- embedded Matplotlib inside a PySide6 UI;
- Dear ImGui + ImPlot;
- retained plot textures with cache invalidation.

### Priority

**P2**

---

# 5. Recommended production architecture

The final program should no longer be a single Python file.

Recommended structure:

```text
astro/
│
├── pyproject.toml
│
├── README.md
│
├── docs/
│   ├── architecture.md
│   ├── data-provenance.md
│   ├── physics.md
│   └── assets.md
│
├── src/
│   └── astro_explorer/
│       │
│       ├── app/
│       │   ├── main.py
│       │   ├── controller.py
│       │   └── state.py
│       │
│       ├── data/
│       │   ├── nasa_archive.py
│       │   ├── gaia.py
│       │   ├── simbad.py
│       │   ├── synchronizer.py
│       │   ├── repository.py
│       │   ├── schema.py
│       │   └── provenance.py
│       │
│       ├── physics/
│       │   ├── constants.py
│       │   ├── kepler.py
│       │   ├── orbital_elements.py
│       │   ├── ephemeris.py
│       │   ├── stellar.py
│       │   ├── radiation.py
│       │   └── gravity.py
│       │
│       ├── coordinates/
│       │   ├── frames.py
│       │   ├── transforms.py
│       │   └── floating_origin.py
│       │
│       ├── spectroscopy/
│       │   ├── ipac.py
│       │   ├── models.py
│       │   ├── normalization.py
│       │   └── molecular_evidence.py
│       │
│       ├── classification/
│       │   ├── traditional.py
│       │   └── physical_vector.py
│       │
│       ├── rendering/
│       │   ├── renderer.py
│       │   ├── camera.py
│       │   ├── mesh.py
│       │   ├── materials.py
│       │   ├── star_renderer.py
│       │   ├── planet_renderer.py
│       │   ├── orbit_renderer.py
│       │   ├── picking.py
│       │   └── shaders/
│       │       ├── star.vert
│       │       ├── star.frag
│       │       ├── planet.vert
│       │       ├── rocky.frag
│       │       ├── gas_giant.frag
│       │       ├── atmosphere.frag
│       │       └── orbit.frag
│       │
│       ├── assets/
│       │   ├── manager.py
│       │   ├── manifest.py
│       │   └── procedural.py
│       │
│       └── ui/
│           ├── main_window.py
│           ├── object_browser.py
│           ├── information_panel.py
│           ├── timeline.py
│           └── plots/
│               ├── hr.py
│               ├── blackbody.py
│               └── atmosphere.py
│
└── tests/
    ├── physics/
    ├── data/
    ├── coordinates/
    ├── spectroscopy/
    └── regression/
```

---

# 6. Golden architecture rule

The renderer must never own scientific truth.

Use this flow:

```text
NASA / Gaia / SIMBAD / local snapshot
                │
                ▼
         validated data layer
                │
                ▼
          scientific models
                │
                ▼
         application state
          ┌─────┴─────┐
          ▼           ▼
       UI/plots    OpenGL renderer
```

OpenGL receives already-computed positions and display parameters.

For example:

```python
RenderPlanet(
    position_local=np.array([x, y, z], dtype=np.float32),
    radius_display=...,
    material_id=...,
)
```

The renderer should **not** decide:

- orbital eccentricity;
- semimajor axis;
- current anomaly;
- physical distance;
- whether a parameter is measured;
- whether a molecule has been detected.

---

# 7. Recommended Python technology stack

## Scientific/data layer

```text
NumPy
SciPy
Astropy
Pandas
PyArrow
SQLite
Astroquery
Requests/httpx
```

## 3D rendering

Recommended:

```text
ModernGL
GLSL
```

Window/input options:

```text
PySide6 + OpenGL widget/context
or
Pygame/SDL during engine prototyping
```

## Scientific UI

Recommended long term:

```text
PySide6
```

Plots:

```text
PyQtGraph
Matplotlib where appropriate
```

## Gravity

```text
REBOUND
```

## Packaging

```text
pyproject.toml
PyInstaller or Nuitka
```

---

# 8. Orbital physics specification

---

## 8.1 Kepler's first law

Planet orbits should be represented as ellipses with the star at a focus.

\[
r =
\frac{a(1-e^2)}
{1+e\cos\nu}
\]

---

## 8.2 Kepler's second law

Do not move a planet through equal angular steps per frame.

Use time and solve Kepler's equation so that the planet naturally moves faster near periapsis and slower near apoapsis.

---

## 8.3 Kepler's equation

\[
M=E-e\sin E
\]

Use a robust numerical solver.

---

## 8.4 Mean anomaly

\[
M(t)=M_0+n(t-t_0)
\]

with:

\[
n=\frac{2\pi}{P}
\]

---

## 8.5 Orbital-plane position

\[
x=a(\cos E-e)
\]

\[
y=a\sqrt{1-e^2}\sin E
\]

---

## 8.6 3D orbital orientation

\[
\mathbf r =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf r_{pf}
\]

Handle missing orbital orientation parameters explicitly.

---

## 8.7 Kepler's third law

Use as both a derivation and consistency check:

\[
P^2=
\frac{4\pi^2a^3}
{G(M_\star+M_p)}
\]

The application could show:

```text
Published period
Published semimajor axis
Derived semimajor axis
Kepler consistency residual
```

This would be an excellent educational feature.

---

# 9. Coordinate-system architecture

A single global float32 coordinate space is not sufficient.

You need hierarchical coordinate frames.

---

## 9.1 Galaxy / neighborhood frame

Use:

```text
ICRS / Galactic / Galactocentric
unit: parsecs
CPU precision: float64
```

Data may include:

- RA;
- Dec;
- distance;
- parallax;
- proper motion;
- radial velocity.

Use Astropy coordinate objects.

---

## 9.2 Stellar-system frame

When entering a host system:

```text
origin = host star
unit = AU
```

The host's galaxy position no longer needs to be added to each planet in GPU float32 coordinates.

---

## 9.3 Planetary frame

When approaching an individual planet:

```text
origin = planet
unit = km or planet radii
```

---

## 9.4 Floating origin

Rebase coordinates around the active camera/object.

This avoids precision collapse when moving from parsecs to AU to kilometers.

---

# 10. Distance features

---

## 10.1 Earth to exoplanet / host star

Use the star's 3D astrometric coordinate.

Display, when available:

```text
parsecs
light-years
kilometers
light-travel time
```

Avoid fake precision beyond the catalog uncertainty.

---

## 10.2 Exoplanet to its host star

Compute instantaneous orbital distance:

\[
r =
\frac{a(1-e^2)}
{1+e\cos\nu}
\]

Also display:

```text
periapsis
apoapsis
current/assumed orbital distance
```

---

## 10.3 Selected exoplanet to another selected star

Represent both systems in the same high-precision astronomical frame.

Then:

\[
D=
\left|
\mathbf r_{\rm selected\,star}
-
(\mathbf r_{\rm host}+\mathbf r_{\rm planet})
\right|
\]

For interstellar separations, the small planet-host offset is usually negligible compared with parsecs, but retain it when scientifically meaningful.

---

# 11. Offline-first synchronized data architecture

The application should **always read from a local validated scientific store**.

Do not let the GUI directly depend on a NASA request.

Recommended flow:

```text
NASA Exoplanet Archive
Gaia
SIMBAD
other approved catalogs
        │
        ▼
synchronization layer
        │
        ▼
schema validation
        │
        ▼
staging database
        │
        ▼
integrity tests
        │
        ▼
atomic snapshot replacement
        │
        ▼
local scientific store
        │
        ▼
application
```

---

## 11.1 Suggested storage

### SQLite

Use for:

- planet records;
- host stars;
- aliases;
- discovery data;
- references;
- provenance;
- data versions;
- asset metadata;
- synchronization state.

### Parquet / Arrow

Use for:

- large spectra;
- dense numerical arrays;
- large tabular datasets.

### Asset directories

Use for:

- textures;
- NASA concepts;
- shader files;
- thumbnails;
- optional observation imagery.

---

## 11.2 Synchronization requirements

Each synchronization should:

1. detect available connectivity;
2. query source metadata/version;
3. download into staging;
4. validate expected columns;
5. validate units;
6. validate row counts and unique identifiers;
7. reject malformed updates;
8. record source timestamps;
9. compute hashes where appropriate;
10. atomically replace the active dataset only after validation.

If synchronization fails:

```text
keep the last valid local dataset
```

The application must remain fully usable offline.

---

# 12. Data provenance requirements

Every important scientific parameter should know:

```text
value
unit
uncertainty +
uncertainty -
source catalog
source table
reference
retrieval date
status
```

Status:

```text
MEASURED
DERIVED
ASSUMED_FOR_VISUALIZATION
UNKNOWN
```

The UI should make these visible.

This is especially important for research-oriented use.

---

# 13. Spectroscopy architecture

Atmospheric spectra should not simply be "all points for a planet."

Create a `Spectrum` object.

Example:

```python
Spectrum(
    planet="WASP-39 b",
    spectrum_type="Transmission",
    facility="JWST",
    instrument="NIRSpec",
    wavelength=...,
    wavelength_unit=...,
    value=...,
    value_unit=...,
    error_plus=...,
    error_minus=...,
    reference=...,
)
```

Do not combine unrelated:

- instruments;
- facilities;
- epochs;
- publications;
- reductions.

Allow the user to overlay spectra deliberately.

---

# 14. Stellar science modules

The new program should preserve and improve the original features.

---

## 14.1 True HR diagram

Primary:

```text
Teff vs luminosity
```

Optional:

```text
absolute magnitude vs color
```

Also retain the original:

```text
Teff vs radius
```

but label it correctly.

---

## 14.2 Blackbody spectrum

Keep Planck and Wien calculations.

Clearly label:

```text
ideal blackbody
```

Possible future comparison:

```text
ideal Planck curve
vs
observed/model stellar spectrum
```

---

## 14.3 Stellar luminosity

If necessary derive:

\[
L = 4\pi R^2\sigma T^4
\]

and mark as derived.

---

# 15. Rendering and shader design

Do not use one universal planet shader.

Recommended material categories:

---

## Rocky body

Potential channels:

```text
albedo
normal
roughness
height
emissive
```

---

## Gas giant

Features:

```text
procedural banding
turbulent cloud layers
limb haze
optional storm features
```

---

## Hot / ultra-hot giant

Features:

```text
temperature-dependent emission
strong day/night gradient
optional cloud suppression
```

---

## Atmosphere

Eventually support:

```text
Rayleigh scattering
Mie scattering
optical depth
scale height
limb appearance
```

Only enable physically specific atmospheric behavior when scientifically justified.

---

## Star

Possible shader features:

```text
temperature-derived visual color
limb darkening
surface granulation/noise
HDR emission
bloom/post-processing
```

---

# 16. Texture and visual provenance

For exoplanets, a texture must not automatically be interpreted as an observed surface.

Use categories:

```text
OBSERVED
NASA_CONCEPT
SCIENTIFIC_PROCEDURAL
GENERIC_CLASS
```

Every asset should carry:

```text
asset_id
planet_name
source_url
creator
credit
asset_type
license/provenance status
retrieval date
SHA-256
```

The UI should visibly indicate examples such as:

```text
NASA Artist Concept
```

or:

```text
Procedural visualization — actual appearance unknown
```

---

# 17. N-body gravity simulation

Gravity should be an **optional dynamics mode**, not a replacement for the observational Keplerian model.

Provide:

```text
OBSERVATIONAL / KEPLER MODE
```

and:

```text
N-BODY DYNAMICS MODE
```

Recommended library:

```text
REBOUND
```

Suggested integrators:

```text
WHFast
    hierarchical, long-term planetary systems

IAS15
    high precision / difficult configurations

MERCURIUS
    hybrid cases / close encounters
```

Important:

Many exoplanet systems do not have complete enough masses, inclinations, phases, and orbital elements for a uniquely determined N-body realization.

The UI must disclose when assumptions are required.

---

# 18. New multidimensional exoplanet classification system

A new classification system is promising, but avoid reducing all planet physics to one arbitrary scalar.

A better research direction is a multidimensional physical classification.

Possible dimensions:

```text
STRUCTURE
THERMAL REGIME
ORBITAL REGIME
ATMOSPHERIC EVIDENCE
IRRADIATION
CONFIDENCE
```

Conceptual example only:

```text
STRUCTURE
R = rocky
S = super-Earth regime
N = Neptunian
G = giant

THERMAL
C = cold
T = temperate
H = hot
U = ultra-hot

ATMOSPHERE
? = unknown
D = detected
C = composition constrained
```

A planet might eventually receive a vector such as:

```text
G-H-D
```

This is only a design concept.

Do **not** fix scientific boundaries until a literature review and statistical analysis have justified them.

---

# 19. Features from the original application that must survive

The final program should preserve:

- host-star search;
- planet search;
- orbit viewer;
- HR diagram;
- blackbody spectrum;
- atmospheric spectrum;
- molecule/evidence information.

And add:

- modern OpenGL 3D scene;
- galaxy/stellar neighborhood view;
- host-system view;
- planet close-up view;
- real 3D orbital elements;
- physical time controls;
- Earth distance;
- host-star distance;
- selected-star distance;
- coordinate-system display;
- data provenance;
- uncertainty display;
- NASA/procedural texture provenance;
- offline snapshot management;
- optional N-body dynamics;
- new classification framework.

---

# 20. Recommended UI modes

A useful top-level structure could be:

```text
UNIVERSE
SYSTEM
PLANET
SCIENCE
DYNAMICS
DATA
```

---

## UNIVERSE

Show:

- exoplanet host stars;
- galactic coordinates;
- Earth;
- distance measurements;
- discovery filtering;
- search;
- selectable systems.

---

## SYSTEM

Show:

- host star;
- all known planets;
- 3D orbit paths;
- habitable zone;
- orbital time controls;
- orbital-element display;
- barycentric options later.

---

## PLANET

Show:

- 3D sphere;
- provenance badge;
- radius;
- mass;
- density;
- gravity;
- equilibrium temperature;
- atmospheric evidence;
- discovery information.

---

## SCIENCE

Tabs:

```text
HR Diagram
Blackbody
Atmospheric Spectra
Orbital Elements
Distance
Coordinates
```

---

## DYNAMICS

Show:

```text
Kepler mode
N-body mode
integrator
simulation speed
assumptions
energy error
```

---

## DATA

Show:

```text
source
reference
uncertainties
measured/derived/assumed
last synchronized
offline snapshot version
```

---

# 21. Recommended development phases

---

## Phase 0 — freeze and document

**Branch:** `main`

Tasks:

- [ ] stop visual feature development in the old Tkinter application;
- [ ] document current 2D behavior;
- [ ] tag the last historical state if desired;
- [ ] create reproducible dependencies.

---

## Phase 1 — correct the 2D scientific baseline

**Branch:** `main`

Tasks:

- [ ] fix atmospheric IPAC parsing;
- [ ] remove fake `a = 1 AU`;
- [ ] remove silent `e = 0`;
- [ ] establish NASA record-selection policy;
- [ ] fix HR naming/definition;
- [ ] use Astropy constants and units;
- [ ] add Kepler solver tests;
- [ ] add spectrum parser tests;
- [ ] add data provenance model.

Once complete, tag something similar to:

```text
v0.2.1-science-corrected
```

---

## Phase 2 — extract reusable scientific core

**Primary branch:** `3D-test`

Do not copy the old GUI code.

Port/refactor:

```text
NASA data model
Kepler solver
stellar calculations
spectroscopy parser
coordinate logic
provenance
```

into independent modules.

---

## Phase 3 — replace legacy OpenGL prototype internals

**Branch:** `3D-test`

Tasks:

- [ ] create modern OpenGL context;
- [ ] move to VAO/VBO/EBO;
- [ ] write first vertex/fragment shaders;
- [ ] implement perspective camera;
- [ ] implement object picking;
- [ ] create reusable sphere mesh;
- [ ] implement floating origin;
- [ ] stop using GLU spheres;
- [ ] stop using `glBegin/glEnd`.

---

## Phase 4 — one scientifically correct star + one planet

**Branch:** `3D-test`

Acceptance criteria:

- [ ] one star renders;
- [ ] one planet renders;
- [ ] orbital period is physical;
- [ ] Kepler equation is solved;
- [ ] eccentric orbit is correct;
- [ ] inclination is applied;
- [ ] orbital orientation status is visible;
- [ ] no fake missing-data defaults;
- [ ] distance units are correct.

Do not expand to thousands of systems before this passes.

---

## Phase 5 — complete host systems

Tasks:

- [ ] multiple planets;
- [ ] per-system local frame;
- [ ] LOD;
- [ ] orbit-line batching;
- [ ] labels;
- [ ] system information panel;
- [ ] habitable-zone visualization;
- [ ] search/filter.

---

## Phase 6 — offline scientific database

Tasks:

- [ ] SQLite schema;
- [ ] Parquet spectra;
- [ ] data version table;
- [ ] synchronization staging;
- [ ] validation;
- [ ] atomic update;
- [ ] offline startup;
- [ ] provenance queries.

---

## Phase 7 — textures and shaders

Tasks:

- [ ] NASA asset manifest;
- [ ] credits;
- [ ] hash validation;
- [ ] generic/procedural material classes;
- [ ] rocky shader;
- [ ] giant shader;
- [ ] atmosphere shader;
- [ ] star shader;
- [ ] provenance badges.

---

## Phase 8 — scientific workstation UI

Tasks:

- [ ] HR diagram;
- [ ] stellar radius-temperature diagram;
- [ ] blackbody spectrum;
- [ ] atmospheric spectra;
- [ ] molecule evidence;
- [ ] coordinate inspector;
- [ ] distance calculator;
- [ ] uncertainty display;
- [ ] reference links.

---

## Phase 9 — stellar neighborhood / universe view

Tasks:

- [ ] ICRS/Galactic coordinates;
- [ ] host-star positions;
- [ ] floating origin;
- [ ] high-performance star rendering;
- [ ] Earth marker;
- [ ] selected-star distance;
- [ ] proper-motion support later.

---

## Phase 10 — gravity sandbox

Tasks:

- [ ] REBOUND integration;
- [ ] WHFast;
- [ ] IAS15;
- [ ] assumptions panel;
- [ ] simulation state copy;
- [ ] energy/angular-momentum diagnostics;
- [ ] never overwrite catalog orbit data with simulation output.

---

## Phase 11 — classification research

Tasks:

- [ ] literature review;
- [ ] candidate dimensions;
- [ ] statistical clustering;
- [ ] uncertainty treatment;
- [ ] classification versioning;
- [ ] validation against known populations.

---

# 22. Minimum scientific test suite

Before calling the application scientifically reliable, test at least:

---

## Orbital mechanics

- [ ] circular orbit;
- [ ] low eccentricity;
- [ ] Mercury-like eccentricity;
- [ ] very high eccentricity;
- [ ] period consistency;
- [ ] semimajor-axis derivation;
- [ ] inclination transform;
- [ ] unknown node handling;
- [ ] periapsis/apoapsis distances.

---

## Stellar physics

- [ ] Sun blackbody reference;
- [ ] Wien peak;
- [ ] luminosity derivation;
- [ ] unit consistency.

---

## Coordinates

- [ ] known RA/Dec/distance object;
- [ ] ICRS Cartesian round-trip;
- [ ] pc ↔ ly;
- [ ] AU ↔ pc;
- [ ] floating-origin transforms.

---

## Spectroscopy

- [ ] IPAC table column mapping;
- [ ] asymmetric uncertainties;
- [ ] metadata extraction;
- [ ] multiple spectra per planet remain separate.

---

## Missing data

- [ ] unknown eccentricity;
- [ ] unknown semimajor axis;
- [ ] unknown stellar mass;
- [ ] unreliable parallax;
- [ ] unknown orbital node;
- [ ] incomplete N-body initial conditions.

No missing-data test should silently create Earth-like values.

---

# 23. Immediate repository action list

## On `main`

Fix these first:

```text
P0  atmospheric .tbl parser
P0  NASA `ps` duplicate/reference handling
P0  fake semimajor-axis fallback
P0  fake eccentricity fallback
P1  physical-time orbital propagation
P1  correct HR terminology/physics
P1  Astropy units/constants
P2  packaging/assets
P2  pyproject.toml
```

Then largely freeze it.

---

## On `3D-test`

Work here for the future application.

Fix these before adding visual complexity:

```text
P0  real Kepler solver
P0  remove incorrect 0.005 AU→pc scaling
P0  hierarchical coordinate frames
P0  real 3D orbital orientation
P0  remove fake orbital defaults
P1  epoch/phase model
P1  Astropy coordinates
P1  modern shader-based OpenGL
P1  provenance-aware assets
P2  improved plot/UI embedding
```

Then begin the modular rewrite.

---

# 24. What NOT to do next

Do **not** start by:

- downloading thousands of planet textures;
- writing atmospheric scattering shaders;
- adding N-body gravity to the monolith;
- adding more tabs to `stellar_navigator_3d.py`;
- converting every Matplotlib feature immediately;
- drawing thousands of planetary systems simultaneously;
- inventing missing orbital elements;
- merging `3D-test` into `main`;
- switching to Vulkan.

The most valuable next task is:

> **Build a modular scientific core and prove one star + one planet is physically and numerically correct in a modern OpenGL scene.**

---

# 25. Why OpenGL is the right choice here

For this project, Vulkan is unnecessary at this stage.

Modern OpenGL is capable of:

- thousands/millions of star points;
- instanced objects;
- textured planets;
- custom GLSL shaders;
- atmospheric rendering;
- orbit-line buffers;
- HDR;
- bloom;
- framebuffers;
- GPU post-processing;
- LOD;
- interactive camera control.

Python remains suitable because the heavy numerical/render work occurs in:

- NumPy/SciPy;
- Astropy;
- REBOUND C core;
- GPU shaders/OpenGL driver.

The key performance rule is:

```text
avoid per-object Python draw overhead
```

Use batching and instancing instead.

---

# 26. Long-term identity of the project

Do not limit the goal to:

> "a Python clone of NASA Eyes on Exoplanets."

A stronger identity would be:

> **An offline-first 3D exoplanet scientific explorer and educational workstation, combining catalog-accurate astronomical data, orbital mechanics, stellar physics, spectroscopy, provenance, and optional gravitational dynamics.**

NASA Eyes can inspire the interaction model.

Your program can distinguish itself through:

- explicit uncertainties;
- measured vs derived vs assumed values;
- source provenance;
- actual scientific plots;
- spectroscopy;
- coordinate inspection;
- Kepler-law diagnostics;
- optional N-body experiments;
- offline reproducibility;
- transparent visual provenance.

---

# 27. Recommended immediate next milestone

## Milestone: `3D scientific kernel v0.1`

Work on:

```text
3D-test
```

Goal:

Render one selected exoplanet system using:

- validated NASA/local data;
- Astropy units;
- a proper Kepler solver;
- real orbital period;
- eccentricity;
- inclination when known;
- explicit unknown orientation fields;
- AU-scale local system coordinates;
- a modern OpenGL shader;
- one star mesh/material;
- one planet mesh/material;
- camera controls;
- a simple information panel.

Acceptance test:

```text
No value shown as measured may actually be a visualization fallback.
No orbital distance may depend on an arbitrary scale factor.
No atmospheric spectrum may rely on positional column guesses.
The program must work from a validated local snapshot with the network disabled.
```

Once that passes, expand outward.

---

# 28. Repository-specific source map

Current important files observed in the repository:

## `main`

```text
exoplanet_analyzer.py
planet_molecules.csv
atmospheric_signatures.json
exoplanet_cache.feather
exoplanet_data_full.csv
dependencies.bat
exoplanet_analyzer.spec
tables/*.tbl
```

## `3D-test`

```text
stellar_navigator_3d.py
Lib_Installer.bat
README.md
```

The two programs should not continue evolving as unrelated monoliths.

The corrected scientific models from `main` should become shared/reimplemented modules inside the future 3D architecture.

---

# 29. External scientific references to retain

NASA Exoplanet Archive:

https://exoplanetarchive.ipac.caltech.edu/

Planetary Systems API documentation:

https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

TAP documentation:

https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

Atmospheric Spectroscopy documentation:

https://exoplanetarchive.ipac.caltech.edu/docs/atmospheres/atmospheres_home.html

NASA Eyes on Exoplanets:

https://eyes.nasa.gov/apps/exo/

NASA Eyes tutorial:

https://science.nasa.gov/tutorials/eyes-on-exoplanets-tutorial/

Astropy coordinates:

https://docs.astropy.org/en/stable/coordinates/

REBOUND:

https://rebound.readthedocs.io/

ModernGL:

https://moderngl.readthedocs.io/

---

# Final branch instruction

## Use `main` for:

**correcting, validating, documenting, and then preserving the 2D application.**

## Use `3D-test` for:

**all new 3D/OpenGL architecture and the future scientific application.**

The 3D branch is the branch you should actively work on for the project you described.

Do not treat `stellar_navigator_3d.py` as the final architecture. Treat it as the proof that Python + OpenGL works, then progressively replace it with a modular scientific engine.

