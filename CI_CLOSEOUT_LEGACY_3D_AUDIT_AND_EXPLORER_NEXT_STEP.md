# CI Close-Out, Legacy 3D Audit, and Explorer Next Step

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Audited branch tip:** `f0515bb8af1820bc97ff1c2f4dbbfac2dec09ea7`  
**Current reported suite:** 547 tests  
**Verdict:** CI/multi-planet milestone accepted. Preserve the legacy prototype, do not extend it, and move to explorer UX.

---

## 1. Phase provenance

Current behavior is scientifically appropriate:

| System | Provenance | Status |
|---|---|---|
| HD 80606 b | `PERIASTRON_EPOCH` | `CONSTRAINED` |
| Kepler-11 ×6 | `TRANSIT_CONJUNCTION_NORMALIZED` | `PARTIALLY_CONSTRAINED` |
| TRAPPIST-1 ×7 | `ASSUMED_ZERO_PHASE` | `ASSUMED` |
| HD 219134 | mixed | 2 / 2 / 2 |

A periastron epoch does not require a known \(\omega\) to remain phase-constrained. A stellar-reflex conversion is `DERIVED`, not assumed.

For conjunction:

\[
u=\omega+\nu=\frac{\pi}{2}
\]

so:

\[
\nu=\frac{\pi}{2}-\omega
\]

is exact for the chosen conjunction definition.

The approximation is equating conjunction with the exact minimum sky-projected separation. Its scale is approximately:

\[
e\cos(\omega)\cos^2(i)
\]

and vanishes as:

\[
i\rightarrow90^\circ
\]

Keeping `conjunction_offset_scale()` is the correct approach.

---

## 2. NaN/text normalization

Moving missing-text handling into:

```text
astro_explorer/text.py
```

is correct.

It should continue handling `None`, Python/NumPy/Pandas NaNs, empty strings, whitespace and archive null spellings without false positives such as:

```text
Nancy et al. 2020
```

---

## 3. Physical clock regression

Keep the Kepler-11 regression permanently:

```text
inner planet ≈ 11.49 revolutions
outer planet = 1.00 revolution
```

A deliberately normalized-clock implementation should fail this test.

---

## 4. CI assessment

The current workflow is structurally strong:

```text
Python 3.11 + 3.12
architecture tests first
full test suite
offline snapshot check
Mesa/llvmpipe GL 3.3
shader/context verification
OpenGL backend tests
multi-planet tests
actual renders
frame existence assertion
artifact upload
```

The first false-green CI run exposed two useful defects.

### MSAA capability mismatch

The demos requested 8× MSAA while llvmpipe supported fewer samples.

Correct solution:

```text
requested samples
→ query ctx.max_samples
→ clamp
→ log degradation
```

Anti-aliasing is a quality setting, so graceful degradation is correct.

### Render failure returned success

Keep the distinction:

```text
optional ModernGL absent
    -> benign when rendering is optional

renderer present + render requested + render fails
    -> non-zero process exit
```

The separate PNG-existence assertion is also valuable.

---

# 5. Audit of `stellar_navigator_3d.py`

The audit confirms that the roadmap's criticism of the old prototype was accurate.

Do not rebuild the production explorer by extending this file.

Use it only as:

```text
historical prototype
interaction reference
proof that Python + OpenGL was viable
source of regression cases
```

---

## Confirmed legacy defects

### No NASA solution policy

The old script queries `ps` without filtering `default_flag = 1`, so multiple literature solutions may survive.

### Invalid parallax becomes fictional distance

The prototype uses a fallback equivalent to:

```python
dist_pc = np.where(parallax_arcsec > 0, 1.0 / parallax_arcsec, 1e9)
```

An invalid parallax therefore becomes one billion parsecs instead of `UNKNOWN`.

### Hard-coded physical constants

Planck, light-speed and Boltzmann constants are directly hard-coded rather than supplied by the trusted scientific constants layer.

### Fixed-function OpenGL

The old code uses GLU, `glBegin/glEnd`, matrix-stack transforms and other fixed-function constructs.

### First-order Kepler approximation

The old propagator approximates:

\[
E\approx M+e\sin M
\]

rather than solving:

\[
M=E-e\sin E
\]

### Fictional missing orbital values

The old path substitutes values equivalent to:

```text
a = 1 AU
e = 0
P = 365.25 days
```

when data is missing.

### Invalid AU/pc scaling

The old renderer uses approximately:

```python
* 0.005
```

for orbital scale.

Physically:

\[
1\,{\rm AU}\approx4.8481368\times10^{-6}\,{\rm pc}
\]

The modern hierarchical frame architecture is the correct solution.

### Coplanar systems

Planet positions are effectively:

```text
[x, y, 0]
```

without full \(i,\omega,\Omega\) orientation.

---

# 6. Additional defects found in the legacy audit

## 6.1 Time units are inconsistent

This is worth adding to the legacy-defect documentation.

The Archive's:

```text
pl_orbper
```

is in **days**.

The legacy animation uses:

```python
time.time() * ORBIT_ANIMATION_SPEED
```

where `time.time()` is in **seconds**.

The old animator therefore mixes seconds and days in the same anomaly calculation.

This is more serious than merely lacking a proper epoch: the propagation is dimensionally inconsistent.

Do not fix the legacy file. Document it.

---

## 6.2 Planet material classification is physically invalid

The old renderer effectively uses orbital distance to decide whether a planet is a gas giant.

A criterion like:

```text
semimajor axis > 1 AU
```

is not a composition classifier.

Future material selection should instead rely on measured/derived physical properties such as:

```text
radius
mass
density
scientific class
temperature / irradiation
asset provenance
```

---

## 6.3 Display size is coupled to world coordinates

Legacy sphere/billboard sizes are expressed in the same parsec-scale scene as star positions.

This contaminates geometric meaning with visibility scaling.

The modern `physical_radius` / `display_radius` separation correctly prevents this.

---

## 6.4 Gaia cross-match should not be ported

The legacy code builds a large host-name list and tries to join Gaia using external identifiers.

Do not reuse this design.

Future Gaia integration should prefer:

```text
known Gaia source_id
→ direct match
```

or, when no trusted ID exists:

```text
ICRS coordinate cross-match
+ explicit angular radius
+ ambiguity handling
+ match provenance
```

Remote queries should be batched.

---

## 6.5 Cache is not synchronization

The legacy Feather behavior is essentially:

```text
file exists → use forever
file absent → download
```

It lacks:

```text
snapshot version
schema migration
freshness
staging validation
atomic replacement
rollback
provenance
```

Do not reuse it for the final offline catalog.

---

## 6.6 Async selection race

SIMBAD and SkyView requests run in background threads and write shared panel state.

If the user selects A and then B:

```text
A starts
B starts
B finishes
A finishes later
```

A can overwrite B's panel with stale information.

Future async work should carry a:

```text
selection_id / generation token / object_id
```

and discard stale results.

---

## 6.7 Legacy picking should not be reused

The old code finds stars near a ray using a fixed world-space threshold.

Potential failures:

```text
objects behind camera
perspective-dependent selection radius
wrong visible object chosen
poor scaling to large star counts
```

For the modern explorer, prefer a GPU integer-ID picking buffer or a properly indexed CPU ray-intersection implementation.

---

## 6.8 Frame loop will not scale

The old code loops over the stellar table every frame and filters planet rows for nearby systems.

Do not reproduce dataframe filtering in the render loop.

Use:

```text
pre-grouped system records
spatial index / culling
GPU batched stars
nearby-system cache
```

---

# 7. Legacy disposition

Do not spend a sprint repairing `stellar_navigator_3d.py`.

Recommended safeguard:

```text
test_legacy_isolation
```

which ensures production modules/entry points never import the legacy script.

Keep the legacy documentation, but make it obvious that the old executable is not scientifically authoritative.

---

# 8. Current milestone verdict

## PASS

The CI/multi-planet milestone is complete.

Do not add more CI complexity unless a real new failure class justifies it.

---

# 9. Next milestone: explorer experience

Recommended sequence:

## Explorer A — navigation and selection

1. Universe/System view state machine.
2. `UniverseFrame -> SystemFrame` camera transition.
3. Production object picking.
4. Screen-space labels + decluttering.
5. LOD based on projected screen size rather than arbitrary world-distance thresholds.

## Explorer B — information workflow

6. Selected-star/system panel.
7. Planet selection.
8. Per-field provenance/status.
9. Interactive physical time controls.
10. Explicit display-scale disclosure.

## Explorer C — science overlays

11. Habitable-zone rendering.
12. Orbital-plane/orientation overlays.
13. Distance/coordinate panel.
14. HR/blackbody/spectroscopy access from selected systems.

## Explorer D — catalog scale

15. Offline synchronized catalog.
16. Search/filter.
17. Spatial indexing.
18. Catalog version/refresh UI.

Still defer:

```text
REBOUND / N-body gravity
final research classification
large texture library
complex atmospheric shaders
```

---

# 10. Immediate implementation target

Build:

> **UniverseFrame → SystemFrame transition + production object picking + screen-space labels**

Acceptance criteria:

```text
[ ] camera enters/leaves a system without precision loss
[ ] frame switches are explicit and type-safe
[ ] picking cannot select objects behind the camera
[ ] selected identity survives LOD transitions
[ ] labels never modify scientific coordinates
[ ] labels are decluttered
[ ] selected object label is always visible
[ ] LOD is based on projected size/view state
[ ] production code does not import stellar_navigator_3d.py
[ ] all 547+ tests stay green
[ ] headless GL CI still produces verified render artifacts
```

After that, build the system information panel and habitable-zone overlay.

---

## Final judgment

The legacy audit was worth doing.

Carry forward the interaction idea:

```text
fly through hosts
select a star
enter its system
inspect planets
open science panels
```

Do not carry forward the old implementation.
