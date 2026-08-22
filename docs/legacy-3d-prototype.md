# The 3D prototype: `stellar_navigator_3d.py`

**This file is not scientifically authoritative.** It is kept as historical
prototype documentation and as a source of regression cases. Do not extend
it, and do not import it from anything under `src/`; the production code is
forbidden from even naming it, and
`tests/regression/test_legacy_isolation.py` enforces that.

What it is worth is the *interaction model*, which was right:

```
fly through hosts -> select a star -> enter its system -> inspect planets
                  -> open science panels
```

That model is carried forward. The implementation is not.

## Audited defects

Every entry below was verified against the file at the line given, not
inferred from the roadmap's description of it. Each has an assertion in
`tests/regression/test_legacy_isolation.py` proving the current code does
not repeat it.

| # | Defect | Where | Fixed by |
|---|---|---|---|
| 1 | No NASA solution policy: queries `ps` without `default_flag = 1`, so several published solutions survive for one planet | data query | `data/nasa_archive.py` `SolutionPolicy` |
| 2 | An unusable parallax becomes a fictional distance: `np.where(parallax_arcsec > 0, 1.0 / parallax_arcsec, 1e9)` - one billion parsecs instead of UNKNOWN | line 105 | `coordinates/frames.distance_from_parallax` |
| 3 | Hard-coded constants: `H_PLANCK = 6.626e-34`, which is also less precise than CODATA | line 27 | `physics/constants.py`, from Astropy |
| 4 | Fixed-function OpenGL: 11 calls to `glBegin`, `gluSphere`, `glMatrixMode`, `glTranslatef`, `glRotatef`, `glPushMatrix` | throughout | `rendering/gl_backend.py`, GL 3.3 core |
| 5 | First-order Kepler: `eccentric_anomaly = mean_anomaly + ecc*np.sin(mean_anomaly)` | line 290 | `physics/kepler.solve_kepler` |
| 6 | Fabricated missing values: `a = 1.0` (line 287), `e = 0.0`, `P = 365.25` (line 288) | 287-288 | `Status.UNKNOWN`, never substituted |
| 7 | Invalid AU/pc scale: `np.array([x,y,0.0])*0.005  # Scale AU to parsecs`, wrong by three orders of magnitude | lines 292, 322 | hierarchical frames; no conversion at all in `SystemFrame` |
| 8 | Coplanar systems: positions are literally `[x, y, 0.0]`, so no `i`, `omega` or `Omega` | line 292 | `physics/orientation.py` |

## Further defects found in the deeper audit

| # | Defect | Where | Notes |
|---|---|---|---|
| 6.1 | **Time units are inconsistent.** `time.time()*ORBIT_ANIMATION_SPEED` is in *seconds* and is fed into `(2*np.pi/period)*(current_time%period)` where `pl_orbper` is in *days*. The propagation is dimensionally wrong, not merely lacking an epoch. | lines 290, 399 | Fixed by carrying `astropy.units` on every element |
| 6.2 | **Material chosen by orbital distance:** `'gas_giant' if planet['pl_orbsmax'] > 1.0 else 'planet'`. Distance is not a composition classifier. | line 332 | `assets/procedural.planet_material` uses radius, temperature and density |
| 6.3 | **Display size coupled to world coordinates:** sphere and billboard sizes live in the same parsec-scale scene as star positions, mixing visibility with geometry. | renderer | `physical_radius` / `display_radius` separation, enforced by test |
| 6.4 | **Gaia cross-match by external identifier:** builds a large host-name list and joins on `original_ext_source_id`. | line 68 | Not ported. Future work should use a known `source_id`, or an ICRS cross-match with an explicit radius, ambiguity handling and match provenance |
| 6.5 | **Cache is not synchronisation:** `if os.path.exists(CACHE): read_feather(...)` - file exists, use forever. No version, schema, freshness, staging, atomicity, rollback or provenance. | lines 83-89 | `data/repository.py` + `data/synchronizer.py` |
| 6.6 | **Async selection race:** SIMBAD and SkyView threads write shared panel state, so a slow request for star A can overwrite the panel for star B. | lines 166-167 | Not ported. Future async work must carry a selection generation token and discard stale results |
| 6.7 | **Picking:** `np.linalg.norm(np.cross(positions - origin, direction))` is the perpendicular distance to an *infinite* line, so a star directly behind the camera scores as well as one in front; `argmin` over that distance then prefers an occluded object; and the threshold is a fixed world-space number, so the selection radius changes with perspective. | line 430 | `rendering/picking.py` rejects anything behind the camera and takes the nearest *hit* |
| 6.8 | **Frame loop will not scale:** `for _, star in self.star_data.iterrows()` runs over the whole table every frame, filtering planet rows for nearby systems. | line 395 | Pre-grouped records, instanced draws, batched orbits |

## Why it is kept

The prototype proved Python + OpenGL was viable for this project, and its
defect list became the specification for the replacement. Its own original
documentation follows, unchanged.

---

3D Exoplanet System Navigator
This is an interactive 3D celestial map that allows you to explore all known star systems that host exoplanets. The application uses real astronomical data from the Gaia mission, the NASA Exoplanet Archive, and the SIMBAD database to provide a scientifically accurate and recognizable representation of these fascinating systems.

When you fly close to a star, you will see a real-time, animated 3D representation of its planetary system, complete with textured planets and orbit lines.

Features
Exoplanet-Focused Universe: The map exclusively displays stars confirmed to host exoplanets, allowing for a focused exploration of known planetary systems like 51 Eridani and TRAPPIST-1.

Common Star Names: Fetches recognizable star names from the SIMBAD database instead of just catalog numbers.

Animated 3D Planetary Systems: As you approach a host star, its planets appear as sprites and begin to orbit in real-time along 3D paths.

"Google Earth" Style Zoom: Fly even closer to a planet, and its 2D sprite will seamlessly transition into a detailed, textured 3D sphere.

Interactive Camera: Fly through the galaxy with intuitive mouse and keyboard controls (WASD, mouse drag, scroll wheel).

Ray-Cast Selection: Click on any star to select it and bring up a detailed information panel.

Fully Functional Data Panel:

Details: Shows key data like the star's common name, distance, temperature, luminosity, and a list of its known planets.

Orbit Viewer: Displays a top-down 2D plot of the planetary system's orbits.

HR Diagram: Plots the selected star on a Hertzsprung-Russell diagram of all other exoplanet hosts.

Spectrum: Shows the star's theoretical black-body radiation curve.

Sky View: Fetches and displays real astronomical images of the star from professional sky surveys (Pan-STARRS and DSS).

Data Caching: Fetched data is cached locally for much faster startup times on subsequent runs.

Multi-Threaded Data Fetching: Sky survey images and star names are loaded in the background to keep the UI responsive.

Setup and Installation
This project uses Python and several external libraries. The following steps will guide you through setting up a dedicated environment using Anaconda/Miniconda.

1. Prerequisites
Anaconda or Miniconda: You must have a working installation. You can download Miniconda here.

2. Create the Conda Environment
Open your terminal (Anaconda Prompt on Windows, or your default terminal on macOS/Linux) and run the following commands:

# Create a new conda environment named 'astro3d' with Python 3.9
conda create --name astro3d python=3.9 -y

# Activate the new environment
conda activate astro3d

3. Install Dependencies
With the astro3d environment active, run the following command in your terminal to install all required libraries:

pip install pygame PyOpenGL numpy pandas astroquery tenacity pyarrow astropy Pillow

4. Running the Application
Once the environment is set up and the dependencies are installed:

Make sure your astro3d conda environment is active.

Navigate to the project directory in your terminal.

Run the main Python script:

python stellar_navigator_3d.py

Note on First Run: The very first time you launch, it may take several minutes to download and cross-match the exoplanet and Gaia catalogs. This is a complex, one-time process. The application will create .feather cache files in the directory, and all subsequent launches will be much faster.

How to Use
Navigate:

Look Around: Click and drag the left mouse button.

Pan: Use the W, A, S, D keys.

Zoom: Use the mouse scroll wheel.

Select a Star: Left-click on any point of light.

View an Animated System: Fly close (within ~3 parsecs) to any star. The planets will automatically appear and begin to orbit. Zoom closer to an individual planet to see it transition into a 3D sphere.

Explore Data: The panel on the right will update with the selected star's data. Click the different tabs to explore all the available visualizations and information.
