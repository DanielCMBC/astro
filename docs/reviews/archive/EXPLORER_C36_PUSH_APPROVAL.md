# Explorer C3.6 Push Approval

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote baseline verified:** `1508738d3d14888e3799012a9f57a4d3d6d47e1f`  
**Normalization commit:** `63a4d555bfe255c0834d0142debaf8b7bad34c6d`  
**C3.6 commit:** `a16fb5928ae0ccdf9f29a06ebaf69cc74d80880e`  
**Reported full suite:** `1005 passed, 1 skipped`  
**Reported C3.6 suite:** `81 tests`  
**Verdict:** **APPROVED TO PUSH AS A NORMAL FAST-FORWARD**

---

## 1. The duplicate audit issue is closed

The repeated review document was byte-identical to the earlier audit and did not contain a new set of instructions.

The P0 from that audit — the missing full-orientation publication gate — has already been implemented, committed, tested, and re-verified.

There is no missing "third audit" required before this push.

---

## 2. Orientation publication gate — PASS

The C3.6 implementation now correctly separates:

```text
temporal phase knowledge
```

from:

```text
physical orbital orientation knowledge
```

A real observed phase anchor alone cannot publish a unique absolute celestial position.

The following are now independently enforced:

```text
inclination
planet-frame periapsis orientation
periastron convention
node convention
node sense
astrometric epoch / common-time state
```

The eleven regressions supplied cover the important cases:

```text
transit epoch + assumed omega + eccentric orbit -> blocked
periastron epoch + unknown omega -> blocked
assumed inclination -> blocked
fully observed orientation -> passes
unstated periastron convention -> blocked
stellar-reflex omega -> converted and may pass as DERIVED
display-normalized omega/i -> blocked
timing + orientation blockers -> reported together
planet->star distance -> inherits same orientation gate
node gates -> necessary but not sufficient
circular orbit -> conservatively blocked on periapsis
```

This closes the P0.

---

## 3. Circular-orbit policy — conservative refusal approved

The optional circular-orbit exemption was deliberately **not** implemented.

That is acceptable.

For a mathematically circular orbit:

```text
e = 0
```

periapsis direction is not physically meaningful.

However, introducing a clean exemption requires reasoning about:

```text
phase-anchor kind
whether eccentricity is genuinely scientific or display-normalized
whether a periastron epoch itself remains a meaningful anchor
```

That would couple phase and orientation semantics more deeply.

The documented conservative policy:

```text
keep absolute publication blocked until that special case is modeled explicitly
```

is scientifically safe.

The regression pinning this refusal prevents a future developer from mistaking it for an accidental omission.

No additional circular-orbit tests are required for C3.6.

---

## 4. C3.6 overall scientific architecture — PASS

The following previously audited contracts are now reported complete:

```text
UNSPECIFIED node convention fails closed
SystemSlice uses provenance-aware orbital propagation
AstrometricState / PropagatedAstrometry / MotionKnowledge
Gaia DR3 exact-ID provider
Gaia J2016.0 TCB reference epoch
pmra -> pm_ra_cosdec without a second cos(dec)
atomic offline cache
DIRECTION_ONLY zero-RV model disclosure
DIRECTION_ONLY cannot manufacture RV knowledge
TimedOrbitalState
three-way common-time matching
single Time -> orbital-clock conversion
epoch_resolved boolean removed
complete physical orientation gate
planet->star common-time + orientation gate
```

This is now a coherent coordinate-physics slice.

---

## 5. DIRECTION_ONLY policy — PASS

The model now correctly treats a missing-RV propagation as:

```text
model-dependent direction realization
```

rather than as fully observed six-dimensional motion.

Important preserved invariants:

```text
RV remains UNKNOWN
DIRECTION_ONLY does not become FULL_SPACE_MOTION
chaining rebases to the observed root
zero-RV approximation is disclosed
3D absolute publication stays blocked
planet->star exact distance stays blocked
```

The adversarial `RV=-100/+100 km/s` long-baseline test is strong evidence that this distinction has physical consequences.

---

## 6. Common-time contract — PASS

The introduction of:

```text
TimedOrbitalState
```

makes the planet's instant structural rather than implicit.

Absolute position and planet-to-star distance now require the physical instants of:

```text
host astrometry
planet orbital state
target-star astrometry
```

to agree.

Using `same_instant()` rather than comparing raw JD numbers is the correct design.

---

## 7. Time-scale boundary — PASS

The centralized:

```text
orbital_time_jd(Time, TimeScale)
```

is the correct single conversion boundary between Astropy time and the orbital clock.

A precise requested instant does not retroactively resolve an unstated catalogue epoch scale.

Keep the existing uncertainty semantics for `JD_UNSPECIFIED`.

---

## 8. Gaia provider/cache — PASS

The provider design is appropriate:

```text
NASA gaia_dr3_id
    -> exact Gaia source_id
    -> validate row
    -> validate ref_epoch/release/schema
    -> atomic cache write
    -> offline GaiaHostIndex
```

Failing closed on:

```text
wrong source
duplicate rows
schema/release mismatch
malformed ID
```

is preferable to silently selecting a plausible nearby source.

No cone-search fallback should be added when an exact Gaia ID exists.

---

## 9. Line-ending split — PASS

Keep the commits separate:

```text
63a4d555  normalize Python line endings
a16fb592  Explorer C3.6 scientific changes
```

This preserves readable history and a clean science diff.

Do not squash them together before pushing.

---

## 10. Review archive — PASS

The duplicate root review has been removed.

The archived copy under:

```text
docs/reviews/archive/
```

is already included in the C3.6 commit.

That matches the established repository policy.

---

## 11. Verification status

Reported:

```text
full suite        1005 passed, 1 skipped
C3.6 suite        81 tests
verify_gl.py      PASS
demos             4 demos / 11 frames
working tree      clean
```

The remote branch is still at the C3.5.1 baseline, so both local commits can be pushed as an ordinary fast-forward.

---

# 12. Push instruction

Preflight:

```bash
git fetch origin
git merge-base --is-ancestor origin/3D-test HEAD
git rev-parse origin/3D-test
git status --short
```

The remote SHA should still be:

```text
1508738d3d14888e3799012a9f57a4d3d6d47e1f
```

and the working tree should be clean.

Then:

```bash
git push origin HEAD:3D-test
```

No force push.

---

# 13. What to report after the push

Send:

```text
branch: 3D-test
remote SHA: <full a16fb592... SHA>
parent normalization SHA: 63a4d555bfe255c0834d0142debaf8b7bad34c6d
CI run: <run id>
CI status: green / failing
```

The remote audit will verify:

```text
commit ancestry
C3.6 named CI step
Gaia offline cache behavior
DIRECTION_ONLY knowledge boundary
TimedOrbitalState/common-time path
orientation publication gate
OpenGL regression safety
artifact production
```

---

# Final decision

**Push C3.6 now as a normal fast-forward.**

There is no missing third local audit blocking this push.
