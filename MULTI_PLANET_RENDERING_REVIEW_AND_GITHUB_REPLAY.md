# Multi-Planet Rendering Review, Phase Provenance, and GitHub Replay Plan

**Repository:** `DanielCMBC/astro`  
**Active development branch:** `3D-test`  
**Current milestone:** Multi-planet rendering with scientific phase/orientation provenance  
**Status:** Pass with one semantic refinement before replaying commits into the real GitHub history

---

## 1. Current milestone assessment

The current implementation is a strong pass.

```text
[PASS] Raw pl_orblper preserved
[PASS] PeriastronConvention added
[PASS] Resolved angle has its own provenance/status
[PASS] ORIENTATION_FULL requires a stated convention
[PASS] TimeScale metadata exists
[PASS] Orbital-validity flags exist
[PASS] Planet/star reference names separated
[PASS] ADS/publication provenance retained
[PASS] Science packages cannot access display_radius
[PASS] Multi-planet rendering implemented
[PASS] Orbit geometry batching implemented
[PASS] Planet instancing implemented
[PASS] Phase provenance tracked per planet
[PASS] Physical clock verified on Kepler-11
```

---

## 2. Periastron convention handling

Keep the raw archive value immutable.

```text
PLANET
    raw value
    status = MEASURED

STELLAR_REFLEX
    raw + 180°
    status = DERIVED

AS_REPORTED
    raw value
    status = ASSUMED_FOR_VISUALIZATION
```

Conceptually:

```python
omega_raw = archive_value
omega_convention = PeriastronConvention.AS_REPORTED
```

If the paper reports the host star's reflex orbit, convert:

\[
\omega_{\rm planet}
=
(\omega_{\rm star}+180^\circ)
\bmod 360^\circ
\]

or:

\[
\omega_{\rm planet}
=
(\omega_{\rm star}+\pi)
\bmod 2\pi
\]

This conversion must remain explicit and testable.

---

## 3. `ORIENTATION_FULL` semantics

`ORIENTATION_FULL` should require:

- known inclination \(i\);
- known argument of periapsis \(\omega\);
- known longitude of ascending node \(\Omega\);
- known/stated periastron convention.

Three numerical angles under an unstated convention are still ambiguous.

Otherwise use:

```text
ORIENTATION_PARTIAL
```

---

## 4. Time-scale model

Keep these distinctions explicit:

```text
BJD_TDB
HJD_UTC
BKJD
BTJD
JD_UNSPECIFIED
```

Do not treat them as interchangeable numeric offsets.

Proper conversion may depend on:

- time standard;
- target coordinates;
- observer geometry;
- spacecraft/observatory ephemeris;
- barycentric/heliocentric correction.

If conversion is implemented later, use an Astropy-based time/coordinate service with explicit prerequisites.

A documented worst-case difference such as:

```text
549 s
```

is useful as a warning, but must not become a universal conversion constant.

---

## 5. Science/display firewall

Keep:

```text
physical_radius
display_radius
```

strictly separate.

The science packages should continue to fail if they attempt to access or name `display_radius`.

The display radius must never affect:

- gravity;
- orbital dynamics;
- transit calculations;
- density;
- collision geometry;
- atmospheric scale height;
- distance calculations;
- scientific plots.

---

## 6. Multi-planet rendering architecture

The batching strategy is sound.

Independent `LINE_STRIP` paths cannot simply be concatenated without joining end-to-end, so explicitly indexed segments with tests preventing boundary crossing are appropriate.

Planet rendering grouped by:

```text
material
LOD
```

with instanced draws is also correct.

Example:

```text
2 instanced draws for 6 planets
```

This is a good approach for scaling system rendering while avoiding per-planet Python draw overhead.

---

## 7. Phase provenance

The phase-provenance correction is important.

Previously, TRAPPIST-1 planets could be counted as physically placed despite lacking usable epochs.

The corrected API returns something equivalent to:

```python
(anomaly, is_assumed)
```

and the panel distinguishes:

```text
observationally constrained phase
```

from:

```text
assumed/display phase
```

A planet should never be counted as physically placed simply because the renderer assigned a convenient anomaly.

---

## 8. Kepler-11 transit-epoch handling

The current numerical placement should be kept, but the semantics should be refined.

The observed quantity is:

```text
transit epoch
```

The following are visualization/model assumptions:

```text
ω = 0°
ν at transit = 90° under the selected normalization
```

For an ideal edge-on orbit:

\[
u = \omega + \nu
\]

with inferior conjunction near:

\[
u = 90^\circ
\]

under the selected convention.

For real systems with \(i\neq90^\circ\), nonzero eccentricity, nonzero impact parameter, or incomplete orientation, exact mid-transit geometry is more subtle.

---

## 9. Recommended Kepler-11 phase semantics

Keep the placement, but record provenance like:

```text
epoch_source          = TRANSIT
phase_anchor          = OBSERVED
omega                 = 0°
omega_status          = ASSUMED_FOR_VISUALIZATION
anomaly_mapping       = CONJUNCTION_NORMALIZED
phase_status          = PARTIALLY_CONSTRAINED
```

A possible enum:

```text
PhaseProvenance:
    PERIASTRON_EPOCH
    TRANSIT_EPOCH
    TRANSIT_CONJUNCTION_NORMALIZED
    ASSUMED_ZERO_PHASE
    UNKNOWN
```

This captures the important distinction that the temporal anchor may be observed while the full orbital orientation is not.

---

## 10. Physical-clock validation

The physical clock was successfully validated on Kepler-11.

Over one orbital period of Kepler-11 g:

```text
innermost planet = 11.49 revolutions
outermost planet = 1.00 revolution
```

Keep this as a permanent regression test.

It verifies that planets advance according to their own orbital periods rather than sharing a normalized animation clock.

---

## 11. Missing-string / NaN handling

Pandas may represent a missing string column as float `NaN`.

Naive truthiness or:

```python
value or ""
```

can therefore leak the literal string:

```text
nan
```

into the UI.

Normalize explicitly for:

- `None`;
- `NaN`;
- empty string;
- whitespace-only string.

This should become a reusable data/UI boundary utility.

---

## 12. GitHub history problem

The current local repository has no shared ancestry with the real GitHub repository.

A normal push will therefore be rejected as unrelated history.

Do not:

- force-push the unrelated local root;
- replace the remote history;
- merge unrelated histories just to satisfy Git.

The correct action is to replay the local commits onto the real `3D-test` history.

---

## 13. Safe GitHub replay procedure

Preserve the current local work:

```bash
git branch backup/local-three-commits
```

Check remotes:

```bash
git remote -v
```

If needed:

```bash
git remote add origin https://github.com/DanielCMBC/astro.git
```

Fetch:

```bash
git fetch origin
```

Create a new branch from the real GitHub `3D-test`:

```bash
git switch -c 3D-test-replay origin/3D-test
```

Replay the local commits in chronological order:

```bash
git cherry-pick <commit-1> <commit-2> <commit-3>
```

Resolve conflicts carefully, rerun the complete test suite, then push:

```bash
git push origin HEAD:3D-test
```

---

## 14. Git replay principle

Desired history:

```text
REAL GitHub 3D-test history
        ↓
new local replay branch
        ↓
cherry-pick scientific commits
        ↓
run complete verification suite
        ↓
normal fast-forward push
```

Avoid:

```text
unrelated local history
        ↓
force push
        ↓
replace GitHub history
```

---

## 15. Verification required after cherry-pick

Rerun at minimum:

```text
[ ] all existing unit tests
[ ] golden science/render separation tests
[ ] spectroscopy tests
[ ] HD 80606 b validation
[ ] high-eccentricity Kepler solver tests
[ ] Kepler II equal-area tests
[ ] Kepler III consistency tests
[ ] shader compilation against real GL 3.3 core
[ ] frame mismatch tests
[ ] precision-loss rendering tests
[ ] Kepler-11 physical-clock test
[ ] TRAPPIST-1 assumed-phase test
[ ] HD 219134 mixed-data system test
[ ] orbit-boundary index tests
[ ] instanced draw tests
[ ] periastron convention tests
[ ] time-scale metadata tests
```

Cherry-pick conflicts can introduce semantic regressions even when the application still starts.

---

## 16. Current project status

```text
Scientific data semantics       ✅
IPAC spectroscopy parsing       ✅
NASA default solution policy    ✅
Missing-data provenance         ✅
Kepler solver                   ✅
Kepler II                       ✅
Kepler III                      ✅
3D orbital transforms           ✅
Frame/type safety               ✅
Modern OpenGL pipeline          ✅
One-planet vertical slice       ✅
Periastron provenance           ✅
Epoch/time-scale model          ✅
Multi-planet rendering          ✅
Phase provenance                ✅
Physical multi-planet clock     ✅
Transit-epoch normalization     ⚠ vocabulary refinement
Real GitHub history replay      ← DO THIS NOW
```

---

## 17. Milestone verdict

**PASS**

The multi-planet milestone is accepted.

Before moving on:

1. refine the transit-anchored phase vocabulary;
2. replay the commits into the real GitHub history;
3. rerun the entire verification suite.

---

## 18. Next milestone after GitHub replay

After replay and validation, consider the multi-planet rendering milestone complete.

Next focus:

```text
LOD behavior
labels
object picking
system information UI
habitable-zone rendering
system-level provenance display
camera/system transitions
offline synchronized catalog layer
```

Do not implement REBOUND/gravity yet.

The project should first complete the main observational/exploration workflow and keep it scalable and scientifically transparent.
