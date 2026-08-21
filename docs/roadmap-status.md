# Roadmap status

Tracks `EXOPLANET_2D_FIXES_AND_3D_OPENGL_ROADMAP.md` item by item.

## Section 23 - immediate action list, `main`

| Priority | Item | Status | Where |
|---|---|---|---|
| P0 | atmospheric `.tbl` parser | done | `spectroscopy/ipac.py` |
| P0 | NASA `ps` duplicate/reference handling | done | `data/nasa_archive.py` |
| P0 | fake semimajor-axis fallback | done | `physics/ephemeris.py`, `data/schema.py` |
| P0 | fake eccentricity fallback | done | `physics/orbital_elements.py` |
| P1 | physical-time orbital propagation | done | `physics/ephemeris.py` `TimeController` |
| P1 | correct HR terminology/physics | done | `physics/stellar.py`, `ui/plots/hr.py` |
| P1 | Astropy units/constants | done | `physics/constants.py`, `provenance.py` |
| P2 | packaging/assets | done | `assets/manager.py`, `exoplanet_analyzer.spec` |
| P2 | `pyproject.toml` | done | `pyproject.toml` |

## Section 4 - `3D-test` corrections

Implemented in the shared core, so porting the prototype is deletion rather
than rewriting.

| Priority | Item | Status | Where |
|---|---|---|---|
| P0 | real Kepler solver | done | `physics/kepler.py` |
| P0 | remove the 0.005 AU->pc scaling | done | `coordinates/system_frame.py` |
| P0 | hierarchical coordinate frames | done | `UniverseFrame` / `SystemFrame` / `PlanetFrame` |
| P0 | real 3D orbital orientation | done | `physics/orientation.py` |
| P0 | remove fake orbital defaults | done | `physics/orbital_elements.py` |
| P1 | epoch/phase model | done | `PhaseKnowledge`, `mean_anomaly_at` |
| P1 | Astropy coordinates | done | `coordinates/frames.py` |
| P1 | modern shader-based OpenGL | done | `rendering/gl_backend.py` |
| P1 | provenance-aware assets | done | `assets/manifest.py` |
| P2 | improved plot/UI embedding | not started | Phase 8 |

## Section 3 - detailed 2D corrections

All of 3.1 through 3.12 are implemented. See
[`original-2d-behaviour.md`](original-2d-behaviour.md) for the
item-by-item mapping from the old code to the new modules.

## Section 5 - production architecture

The `src/astro_explorer/` tree follows the roadmap's proposed layout. Two
modules were added that the roadmap did not name:

* `rendering/scene_builder.py` - the single, testable crossing point from
  science to render primitives, which the golden rule needs somewhere;
* `docs/roadmap-status.md` - this file.

`data/gaia.py` and `data/simbad.py` are not present: nothing consumes them
yet, and an empty module that pretends to be an integration is worse than
its absence. `physics/gravity.py` is likewise deferred with Phase 10.

## Section 21 - development phases

| Phase | Status |
|---|---|
| 0 - freeze and document | done (`docs/original-2d-behaviour.md`, `legacy/`) |
| 1 - correct the 2D baseline | done |
| 2 - extract reusable scientific core | done |
| 3 - replace legacy OpenGL internals | done: `rendering/gl_backend.py` renders through VAO/VBO/EBO, GLSL 3.3 core programs and instanced draws, offscreen. No `glBegin`, `gluSphere` or matrix stack anywhere - asserted by test. Picking and the sphere meshes were already in place. |
| 4 - one correct star + one planet | done: see [`vertical-slice.md`](vertical-slice.md). HD 80606 b (`e = 0.93183`) end to end, verified against independently derivable values. |
| 4a - orbital-semantics pass (review section 14) | done |
| 5 - complete host systems / multi-planet SystemFrame | next |
| 6-11 | not started |

### Phase 4 acceptance criteria

| Criterion | Status |
|---|---|
| one star renders | yes, `test_a_star_actually_appears` |
| one planet renders | yes, `test_the_hd80606b_slice_renders` |
| orbital period is physical | yes, `TimeController`; equal-time steps sweep equal areas |
| Kepler equation is solved | yes, residual < 1e-13 to `e = 0.9999` |
| eccentric orbit is correct | yes, `\|r\|` matches `a(1-e)` and `a(1+e)` to 1e-9 |
| inclination is applied | yes, orbit normal tilt equals 89.24 deg |
| orbital orientation status visible | yes, `Omega: UNKNOWN` + display normalisation |
| no fake missing-data defaults | yes, `tests/test_missing_data.py` |
| distance units are correct | yes, `SystemFrame` in AU with no conversion at all |

Phases 3 and 4 are complete without touching `stellar_navigator_3d.py`: the
prototype is not in this checkout, and the replacements were built from the
formula reference rather than ported from it.

## Review section 14 - orbital-semantics pass

Gating task list from `CURRENT_3D_VERTICAL_SLICE_REVIEW.md`, all complete.
See [`orbital-semantics.md`](orbital-semantics.md).

| Task | Status | Where |
|---|---|---|
| preserve raw omega | done | `OrbitalElements.argument_of_periastron` |
| store periastron convention | done | `physics/orbital_semantics.py` |
| preserve publication/reference | done | `clean_reference`, `reference_url` |
| add `pl_orbtper` | done | `epoch_periastron` |
| preserve time-system/epoch metadata | done | `physics/epoch.py` |
| geometry-valid vs phase-valid | done | `OrbitValidity` |
| partial vs full orientation | done | `OrbitValidity` |
| test the 180-degree conversion | done | `tests/physics/test_orbital_semantics.py` |
| preserve unknown Omega | done | carried over, still tested |
| keep visualisation assumptions explicit | done | `describe_orbit`, dashed orbits |

Review section 6 (physical vs display radius) is enforced by
`test_display_radius_never_reaches_the_science_layers`.

## Formula reference coverage

`ORBITAL_MECHANICS_FORMULAS_3D_EXOPLANET.md`, section by section.

| Section | Where |
|---|---|
| 1-3 mean motion, `M(t)`, Kepler's equation | `physics/ephemeris.py`, `physics/kepler.py` |
| 4-5 perifocal position, orbital distance | `physics/orientation.py` |
| 6 periapsis / apoapsis | `OrbitalElements.periapsis` / `.apoapsis` |
| 7 true anomaly | `physics/kepler.py` |
| 8-11 3D orientation, rotations, expanded form | `physics/orientation.py` |
| 12 Kepler's third law | `physics/ephemeris.py` |
| 13 Kepler's second law | `swept_area`, `swept_area_binned` |
| 14 specific angular momentum | `specific_angular_momentum` |
| 15 perifocal velocity | `perifocal_velocity` |
| 16 vis-viva | `vis_viva_speed` |
| 17 specific orbital energy | `specific_orbital_energy` |
| 18 distance between objects | `coordinates/frames.separation_pc` |
| 19 spherical to Cartesian | `coordinates/frames.cartesian_pc` (via Astropy) |
| 20 pc/AU and hierarchical frames | `coordinates/system_frame.py` |
| 21 unknown orientation | `Status.ASSUMED_FOR_VISUALIZATION` |
| 22 suggested unit tests | all implemented; see the table below |
| 23 science-to-render boundary | `rendering/renderer.py` + architecture tests |
| 24 recommended milestone | done; [`vertical-slice.md`](vertical-slice.md) |

## Section 22 - minimum scientific test suite

| Group | Tests | File |
|---|---|---|
| Orbital mechanics | circular, low e, Mercury-like, high e, near-parabolic, period consistency, axis derivation, inclination transform, unknown node, periapsis/apoapsis | `tests/physics/test_kepler.py` |
| 3D orientation | i = 0, i = 90 deg, omega = 90 deg, Omega = 90 deg, combined vs the expanded reference equations | `tests/physics/test_orientation.py` |
| Conservation | Kepler's second law, areal velocity, specific energy, vis-viva, angular momentum | `tests/physics/test_conservation.py` |
| Hierarchical frames | frame mixing refused, conversions, float32 precision guard | `tests/coordinates/test_system_frame.py` |
| Vertical slice | catalogue to float32 primitives for HD 80606 b | `tests/regression/test_vertical_slice.py` |
| OpenGL | real GL 3.3 context, pixels, lighting, instancing, depth | `tests/regression/test_gl_backend.py` |
| Stellar physics | Sun blackbody, Wien peak, luminosity derivation, unit consistency | `tests/physics/test_stellar.py` |
| Coordinates | known RA/Dec/distance object, ICRS round-trip, pc/ly, AU/pc, floating origin | `tests/coordinates/test_frames.py` |
| Spectroscopy | column mapping, asymmetric uncertainties, metadata, spectra kept separate | `tests/spectroscopy/test_ipac.py` |
| Missing data | unknown e, unknown a, unknown stellar mass, unreliable parallax, unknown node | `tests/test_missing_data.py` |
| Architecture | the golden rule, the render contract, shader hygiene | `tests/regression/test_architecture.py` |
| End to end | the section 27 acceptance criteria | `tests/regression/test_end_to_end.py` |

The section 22 requirement that "no missing-data test should silently create
Earth-like values" is enforced by
`test_no_record_field_silently_equals_an_earth_like_default`.

## Section 27 - acceptance criteria

| Criterion | Test |
|---|---|
| No value shown as measured may actually be a visualization fallback | `test_no_value_shown_as_measured_is_a_visualisation_fallback` |
| No orbital distance may depend on an arbitrary scale factor | `test_no_orbital_distance_depends_on_an_arbitrary_scale_factor` |
| No atmospheric spectrum may rely on positional column guesses | `test_columns_are_addressed_by_name`, `test_a_well_formed_table_with_wrong_columns_is_rejected` |
| Must work from a validated local snapshot with the network disabled | `test_the_program_runs_from_a_snapshot_with_the_network_disabled` |

## Not applicable in this checkout

* **Section 1, branch strategy and tagging.** This directory is not a git
  repository, so `main` and `3D-test` cannot be created or tagged here.
* **`stellar_navigator_3d.py` itself.** The prototype file is absent from
  this checkout, so its code cannot be deleted here; the replacements all
  exist (see the Section 4 table above).
* **Section 17, N-body.** REBOUND is declared as the `dynamics` extra and
  nothing else. Phase 10.
* **Section 11 remote sources.** Gaia and SIMBAD synchronisation is designed
  for (`SolutionPolicy`, the staging pipeline) but only the NASA TAP fetcher
  is implemented.
