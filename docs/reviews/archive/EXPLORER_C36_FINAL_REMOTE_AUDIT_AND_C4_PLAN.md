# Explorer C3.6 Final Remote Audit and C4 Plan

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote HEAD:** `a16fb5928ae0ccdf9f29a06ebaf69cc74d80880e`  
**Normalization parent:** `63a4d555bfe255c0834d0142debaf8b7bad34c6d`  
**CI run:** `32906237254`  
**Verdict:** **EXPLORER C3.6 CLOSED — PASS**

---

## 1. Remote state and history — PASS

The public `3D-test` branch now points exactly to:

```text
a16fb5928ae0ccdf9f29a06ebaf69cc74d80880e
```

and the C3.6 commit's parent is the separate line-ending normalization commit:

```text
1508738  Explorer C3.5.1
    ↓
63a4d55  Normalize coordinates/inspector.py line endings to LF
    ↓
a16fb59  Explorer C3.6: astrometric epoch and stellar space motion
```

The normalization/science split survived exactly as intended.

---

## 2. Remote CI — PASS

GitHub Actions run `32906237254` is green in all three jobs:

```text
Tests (Python 3.11)             PASS
Tests (Python 3.12)             PASS
OpenGL 3.3 core (software Mesa) PASS
```

The Python jobs ran the named:

```text
Architecture / golden-rule tests
Explorer C3 coordinate/distance tests
Explorer C3.5 SystemFrame→ICRS tests
Explorer C3.6 astrometric epoch / space-motion tests
Full test suite
Offline check
```

The OpenGL job also passed:

```text
GL 3.3 context + all shader compilation
OpenGL backend tests
multi-planet rendering tests
Explorer navigation/picking/labels
scientific overlay tests
four real render demos
frame existence assertion
artifact upload
```

The `rendered-frames` artifact belongs to the exact C3.6 head SHA and is not expired.

---

# 3. DIRECTION_ONLY knowledge boundary — PASS

The pushed code now makes the missing-radial-velocity case explicit rather than allowing Astropy's default behavior to manufacture knowledge.

`MotionKnowledge` separates:

```text
REFERENCE_EPOCH_ONLY
DIRECTION_ONLY
FULL_SPACE_MOTION
```

and the DIRECTION_ONLY tier is explicitly described as a model-dependent zero-RV / no-perspective realization rather than as a uniquely determined future 3D state.

Important remote invariants:

```text
missing RV stays UNKNOWN
a measured zero remains a measured zero
DIRECTION_ONLY never becomes FULL_SPACE_MOTION
DIRECTION_ONLY cross-epoch distance is withheld
DIRECTION_ONLY realization carries explicit model notes
```

The implementation also rebases propagated realizations onto their observed root before another propagation, so repeated calls cannot turn one approximation into apparent new evidence.

---

# 4. PropagatedAstrometry — PASS

`PropagatedAstrometry` now carries:

```text
position
obstime
status
source_state
blockers
MotionKnowledge
motion_applied
propagated proper-motion components
radial velocity where scientifically justified
```

`as_state()` preserves an originally unknown RV in the DIRECTION_ONLY case rather than reading Astropy's internally realized radial component back as a measurement.

That is the correct epistemic boundary.

---

# 5. Reference-epoch exception — PASS

C3.6 correctly treats the reference epoch as a special physically valid case.

If:

```text
requested time == catalogue reference epoch
```

then a dated RA/Dec/distance can be used at that instant without inventing proper motion or radial velocity.

For a different time, the required motion tier is enforced.

This avoids the opposite error of demanding motion data merely to report the position at the instant it was measured.

---

# 6. TimedOrbitalState / common-time rule — PASS

The planet's epoch is now structural.

A bare AU vector can still be rendered, but it cannot open the scientific absolute-position gate.

For a published result:

```text
host astrometry time
==
planet orbital-state time
==
target-star astrometry time
```

must hold as a physical instant.

The public `SystemSlice` entry points derive all states from one requested time, rather than expecting callers to coordinate independent numbers.

This closes the previous class of errors where individually correct states from different years could be combined into one plausible coordinate.

---

# 7. Time-scale boundary — PASS

The stellar and orbital clocks now cross through named functions:

```text
astropy_time(...)
orbital_time_jd(...)
```

rather than call sites independently choosing:

```text
.jd
.tdb.jd
.tcb.jd
.utc.jd
```

An exact requested time does not erase uncertainty in an orbital epoch whose published scale was unstated.

Keep this permanently.

---

# 8. Complete physical orientation gate — PASS

Absolute planet publication now independently requires physical orientation as well as temporal phase.

The gate checks, as applicable:

```text
inclination
planet-frame argument of periapsis
periastron convention
node convention
node sense
```

This correctly prevents a real transit/periastron timing anchor from standing in for a fully known 3D orientation.

The circular-orbit special case remains conservatively blocked until phase-anchor semantics are modeled explicitly. That is an acceptable safe policy.

---

# 9. Gaia DR3 provider/cache — PASS

The pushed Gaia path has the right identity and offline architecture:

```text
NASA gaia_dr3_id
    ↓ exact source_id
Gaia DR3 row
    ↓ validation
committed local cache
    ↓
offline GaiaHostIndex
```

Strong details include:

```text
source_id kept as a string
no cone-search fallback
Gaia ref_epoch stored as a real TCB Time
pmra mapped directly to pm_ra_cosdec
missing PM/RV stays UNKNOWN
correlation coefficients retained for future covariance work
schema + Gaia release validated
atomic file replacement
normal application runtime performs no Gaia network access
```

The remote offline CI explicitly verifies that the committed cache remains scientifically heterogeneous:

```text
HD 80606   -> FULL_SPACE_MOTION
TRAPPIST-1 -> DIRECTION_ONLY
```

so losing the radial-velocity distinction would fail the check.

---

# 10. Gaia cache refresh — one non-blocking policy worth tightening later

There is one cache-policy detail to keep on the radar.

The refresh layer intentionally permits a requested Gaia ID to return no row, and it also permits some returned rows to fail validation while accepted rows are committed.

Because the cache write is a full replacement, a partial refresh can therefore remove a previously cached source.

For immutable Gaia DR3 data this should be unusual, but it means the prose:

```text
"any failure leaves the previous validated cache in place"
```

is stronger than the exact implementation.

Before scaling this sync to a much larger production catalogue, choose and test one explicit policy:

### Whole-batch transactional policy

```text
any requested source unexpectedly missing/invalid
    -> keep entire previous cache
```

or:

### Per-source merge policy

```text
successfully refreshed source -> replace
transiently failed source     -> retain previous validated record
confirmed removed/invalid ID  -> explicit tombstone/removal
```

This is **not a C3.6 blocker** for the current six-host committed DR3 cache.

---

# 11. Two documentation cleanups

## A. HD 219134 proper-motion example

`coordinates/astrometry.py` currently says that a star moving about:

```text
2.1 arcsec/year
```

moves by:

```text
~0.14 mas over a decade
```

while also correctly giving the corresponding linear displacement as about 135 AU at 6.5 pc.

The angular value is a typo.

The decade angular displacement is approximately:

```text
21 arcsec
```

and:

```text
21 arcsec × 6.5 AU/arcsec ≈ 136.5 AU
```

So the 135 AU conclusion is consistent; `0.14 mas` is not.

Fix that sentence opportunistically in the next commit.

## B. roadmap-status.md has legacy statements

The roadmap now correctly lists C3.6 as done, but older sections still say things such as:

```text
data/gaia.py is not present
only NASA TAP synchronization is implemented
```

Those statements are now stale.

Clean those old sections when C4 next touches the docs.

No documentation-only push is needed.

---

# 12. Push-approval review file

`EXPLORER_C36_PUSH_APPROVAL.md` should **not** receive its own standalone commit.

Move it to:

```text
docs/reviews/archive/
```

at the beginning of the next meaningful feature commit, matching the established archive policy.

---

# 13. Explorer C3.6 final status

```text
remote exact SHA                         PASS
two-commit ancestry                      PASS
Python 3.11 CI                           PASS
Python 3.12 CI                           PASS
software Mesa CI                         PASS
render artifact                          PASS

MotionKnowledge tiers                    PASS
DIRECTION_ONLY epistemic boundary        PASS
realization rebasing                     PASS
reference-epoch publication              PASS
TimedOrbitalState                        PASS
three-clock common-time rule             PASS
orbital/Astropy time conversion          PASS
full orientation publication gate        PASS

Gaia exact-ID integration                PASS
Gaia TCB reference epoch                 PASS
Gaia proper-motion convention            PASS
offline cache/runtime                    PASS
atomic cache write                       PASS

partial-refresh retention policy         FUTURE HARDENING
Gaia covariance propagation              FUTURE RESEARCH SLICE
Gaia parallax systematic treatment       FUTURE RESEARCH SLICE
```

## **Explorer C3.6 is CLOSED.**

---

# 14. Next milestone — Explorer C4

The next roadmap item should now return to the original scientific-explorer experience:

```text
HR diagram
blackbody spectrum
atmospheric spectrum
```

Do not build all three in one giant commit.

Recommended sequence:

```text
C4a — HR diagram integration
C4b — blackbody spectrum integration
C4c — atmospheric spectroscopy integration
```

Push/audit each as a coherent vertical slice.

---

# 15. C4a — HR diagram integration

The project already has the stellar-physics and HR plotting core.

C4a should connect the currently selected host to that scientific plot without reimplementing stellar calculations.

### Required model rules

```text
plot consumes the same StarRecord / stellar-physics model as the panel
renderer/UI does not recalculate luminosity or temperature
unknown Teff/luminosity produces an explicit unavailable state
measured/derived values remain distinguishable
```

### HR semantics

Use the real H-R convention:

```text
x-axis: effective temperature, hotter to the left
y-axis: luminosity relative to Sun, logarithmic
```

Do not silently turn a color-magnitude diagram into an H-R diagram.

### Acceptance tests

```text
[ ] selected host marker matches the panel's Teff/luminosity values
[ ] hotter stars plot farther left
[ ] one decade in luminosity has equal log-axis spacing
[ ] unknown Teff -> no fake x coordinate
[ ] unknown luminosity -> no fake y coordinate
[ ] derived luminosity is labelled derived
[ ] selection changes marker without mutating stellar data
[ ] plot uses CPU scientific values, never render color/radius
[ ] offline runtime needs no network
```

---

# 16. C4b — blackbody spectrum

Use the existing Planck-law implementation as the single source of truth.

Recommended outputs:

```text
spectral radiance vs wavelength
Wien peak
host Teff
optional comparison with Sun
```

### Important scientific rules

```text
blackbody = idealized stellar continuum model
not an observed stellar spectrum

wavelength/frequency forms must never be mixed
units must be explicit
plot normalization must not contaminate physical values
```

### Acceptance tests

```text
[ ] Sun peak agrees with Wien's law
[ ] numerical integral follows Stefan-Boltzmann relation within tolerance
[ ] changing plot normalization leaves Wien peak unchanged
[ ] unknown Teff -> no spectrum
[ ] selected host uses the same Teff as the stellar panel
```

---

# 17. C4c — atmospheric spectroscopy

This should reconnect the corrected IPAC spectroscopy work to the selected planet.

Keep three layers separate:

```text
measurement
    wavelength + value + uncertainty

interpretation
    molecule / feature evidence

visual annotation
    labels / bands / styling
```

A molecule label must never become a detected molecule merely because a wavelength falls near a known band.

### Acceptance tests

```text
[ ] every plotted measurement corresponds one-to-one with the source table
[ ] asymmetric uncertainty survives
[ ] wavelength units remain explicit
[ ] source/reference remains available
[ ] molecular annotation does not alter measurements
[ ] missing atmosphere data -> explicit unavailable state
[ ] WASP-39 b CO2 feature remains at the correct wavelength/depth
[ ] selection switches spectra without mutating source data
[ ] offline cache only
```

---

# 18. Still defer

Keep these out of C4:

```text
N-body / REBOUND gravity
final exoplanet classification system
large texture acquisition
advanced atmospheric scattering shader
free-flight camera polish
```

The scientific plot integration should be completed before adding another large simulation subsystem.

---

# Immediate instruction

**C3.6 is closed. Start C4a — HR diagram integration.**

At the beginning of that working tree:

1. move `EXPLORER_C36_PUSH_APPROVAL.md` into `docs/reviews/archive/`;
2. fix the `0.14 mas` HD 219134 documentation typo;
3. remove/update the stale roadmap statements that say Gaia integration does not exist;
4. then build the selected-host HR plot as the next isolated vertical slice.

Do not make a standalone cleanup push for items 1–3; let them travel with C4a.
