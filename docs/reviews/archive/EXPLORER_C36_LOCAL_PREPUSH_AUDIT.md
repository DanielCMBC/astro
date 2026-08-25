# Explorer C3.6 Local Pre-Push Audit

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote baseline verified:** `1508738d3d14888e3799012a9f57a4d3d6d47e1f`  
**Reported local suite:** `972 passed, 1 skipped`  
**Reported delta:** `+48 tests`  
**Verdict:** **STRONG IMPLEMENTATION, BUT DO NOT PUSH YET. Two scientific contracts need tightening before the C3.6 commit becomes the remote checkpoint.**

---

## 1. What I approve

The following design changes are good and should stay.

### Node-convention hardening

A valued node with `NodeConvention.UNSPECIFIED` no longer gets silently interpreted as the standard astronomical PA.

The two-part repair is particularly important:

```text
scientific interpretation
    -> refuse the unstated convention

display path
    -> replace with an explicit ASSUMED_FOR_VISUALIZATION normalization
```

That prevents a measured-looking caption from travelling with an assumed guide.

### SystemSlice propagation

Routing `SystemSlice.state()` through the provenance-aware `orbital_elements.state_at_mean_anomaly()` path is the right cleanup.

Keep removing high-level hand-built angle triplets rather than making structural regexes chase them.

### Astrometric state model

The split:

```text
AstrometricState
PropagatedAstrometry
MotionKnowledge
```

is the right architecture.

The three knowledge levels:

```text
REFERENCE_EPOCH_ONLY
DIRECTION_ONLY
FULL_SPACE_MOTION
```

are also the right kind of distinction.

### Gaia cache/provider

These choices are good:

```text
NASA gaia_dr3_id -> exact Gaia DR3 source_id
source_id preserved as string
Gaia ref_epoch validated
Gaia pmra -> Astropy pm_ra_cosdec directly
missing values remain UNKNOWN
atomic cache writes
offline GaiaHostIndex
committed validated cache
```

Keeping `source_id` out of floating-point storage is especially sensible.

### epoch_resolved boolean removal

Deleting the old:

```python
epoch_resolved=True
```

production escape hatch is a real improvement.

A publication gate should be opened by an inspectable astrometric state, not by a boolean assertion.

---

# 2. P0 before push: DIRECTION_ONLY is still model-dependent when RV is missing

This is the biggest scientific issue in the current description.

Astropy's `SkyCoord.apply_space_motion()` explicitly states that:

> if no radial velocity is supplied, radial velocity is assumed to be zero.

That means:

```text
RA/Dec + distance + proper motion + missing RV
```

does **not** produce a purely measurement-determined propagated direction.

It produces the direction implied by the additional model:

```text
RV = 0
```

under Astropy's straight-line constant-velocity propagation.

For a finite-distance source, radial motion changes the apparent angular motion through perspective acceleration.

Therefore:

> `DIRECTION_ONLY` can be useful, but it is not an exact data-only future direction when radial velocity is unknown.

---

## 3. Required DIRECTION_ONLY policy

Choose one explicit policy.

### Preferred policy

Keep `DIRECTION_ONLY`, but define it as:

```text
a first-order / zero-RV astrometric realization,
not a uniquely determined full space-motion solution
```

It may be displayed as a derived/approximate sky direction, but it must not:

```text
unlock a 3D Cartesian absolute position
unlock exact planet->star distance
be described as FULL_SPACE_MOTION
turn the missing radial velocity into a measured/derived physical RV
```

Its note/provenance should explicitly say something equivalent to:

```text
radial velocity unavailable; propagated direction uses the
zero-RV / no-perspective-acceleration approximation
```

If you dislike that approximation being part of science output, the stricter alternative is to withhold cross-epoch direction entirely when RV is unavailable.

---

# 4. Do not let chaining manufacture radial-velocity knowledge

You wrote that:

```text
PropagatedAstrometry.as_state()
carries the motion at the new epoch so chained propagation keeps the perspective term
```

That is excellent for:

```text
FULL_SPACE_MOTION
```

but potentially dangerous for:

```text
DIRECTION_ONLY
```

because Astropy may have created a radial component from a propagation that began with an assumed zero RV.

Do not let:

```text
unknown RV
    -> Astropy zero-RV realization
    -> resulting differential
    -> new AstrometricState with apparently known RV
```

happen.

### Required rule

For `DIRECTION_ONLY`:

```text
as_state() must preserve radial_velocity = UNKNOWN
```

and preferably:

```text
all subsequent direction propagation is recomputed from the original
reference-epoch state
```

rather than chaining a model realization as if it were a new observation.

It is fine for `FULL_SPACE_MOTION.as_state()` to carry the evolved velocity.

---

# 5. Add adversarial tests for the missing-RV case

Add tests that prove the knowledge boundary, not just the arithmetic.

At minimum:

```text
[ ] DIRECTION_ONLY never returns a known radial_velocity
[ ] DIRECTION_ONLY.as_state() preserves RV = UNKNOWN
[ ] DIRECTION_ONLY never becomes FULL_SPACE_MOTION after propagation
[ ] chained DIRECTION_ONLY cannot promote an Astropy-generated RV into evidence
[ ] FULL_SPACE_MOTION may chain and remains FULL_SPACE_MOTION
```

Also add a nearby/high-proper-motion synthetic star and compare two physically possible RVs:

```text
RV = +100 km/s
RV = -100 km/s
```

over a deliberately long baseline.

The resulting sky directions should differ.

That test proves why missing RV means the exact cross-epoch direction is not uniquely determined.

---

# 6. P0 before push: the planet orbital vector must carry the same physical time

The second major issue is the common-time contract.

Your description establishes that:

```text
host origin and host tangent basis
```

come from the same `PropagatedAstrometry`.

Good.

But that does **not by itself prove** that the local planet vector being added was evaluated at the same physical instant.

A bare:

```python
np.ndarray  # planet offset in AU
```

has no epoch.

So this remains possible unless the API structurally prevents it:

```text
host astrometry @ t1
+
planet local vector computed @ t2
```

The result would be numerically valid vector arithmetic and scientifically wrong.

---

# 7. Make planet time structural

Do not rely only on the caller remembering to use the same `time_jd`.

Prefer one of these designs.

### Option A — timed orbital-state wrapper

```python
@dataclass(frozen=True)
class TimedOrbitalState:
    state: StateVector
    obstime: Time
    phase: PhaseSolution
```

Then:

```python
absolute_planet_position(
    host_astrometry: PropagatedAstrometry,
    planet_state: TimedOrbitalState,
    ...
)
```

can verify the instants.

### Option B — compute internally

Have a high-level API such as:

```python
system.absolute_planet_position(record, target_time)
```

which internally:

```text
propagates host -> target_time
propagates planet -> target_time
checks node gates
forms the result
```

This is even harder to misuse.

Either is better than accepting a bare offset plus an unrelated propagated host.

---

# 8. Planet->star distance needs three clocks, not two

You reported that:

```text
planet_to_star_distance compares the two astrometric obstimes
```

That verifies:

```text
host @ t == target star @ t
```

but the planet also has a time.

The strict rule is:

```text
host astrometry obstime
==
target-star astrometry obstime
==
planet orbital-state obstime
```

as physical instants.

Add three mismatch tests:

```text
[ ] host t1 + target t1 + planet t2 -> refuse
[ ] host t1 + target t2 + planet t1 -> refuse
[ ] host t2 + target t1 + planet t1 -> refuse
```

Do not compare only formatted dates or raw JD numbers if different Astropy time scales can represent the same instant.

---

# 9. Define the orbital-clock conversion point explicitly

The stellar side now correctly uses `astropy.time.Time`.

The orbital side historically uses the project's canonical JD/epoch machinery.

C3.6 needs one explicit conversion point:

```text
target physical Time
    ->
orbital time coordinate used by phase_at/state_at
```

Do not let different call sites independently choose:

```text
time.jd
time.tdb.jd
time.tcb.jd
time.utc.jd
```

Document which one the physical orbital clock consumes and why.

If the underlying published orbital epoch has an unspecified scale, keep the existing uncertainty/provenance rather than pretending the time scale became known because the target time is precise.

---

# 10. Gaia reference epoch — keep it as real metadata

Gaia DR3 `ref_epoch` is a Julian Year in TCB, and DR3 uses J2016.0.

Even if every accepted cache row must equal the DR3 constant, the actual astrometric state should still carry a concrete:

```python
Time(ref_epoch, format="jyear", scale="tcb")
```

or its exact project equivalent.

Do not reduce the source epoch to:

```text
validated == True
```

and then lose the actual time object.

Add a test asserting:

```text
reference_epoch.scale == "tcb"
```

and the correct J2016.0 instant.

---

# 11. MotionKnowledge wording

I would slightly tighten the names/description.

`FULL_SPACE_MOTION` is fine.

`REFERENCE_EPOCH_ONLY` is fine.

`DIRECTION_ONLY` may sound more observationally complete than it is across epochs.

Consider documenting it as:

```text
DIRECTION_ONLY:
proper-motion information exists, but complete 3D space motion does not.
Cross-epoch direction is model-dependent when RV is unavailable.
```

You do not necessarily need to rename the enum.

Just prevent the UI/docs from saying:

```text
the direction is fully known at t
```

when Astropy had to assume RV=0.

---

# 12. The statement "epoch gate is open for every Gaia-backed host" should be narrowed

Prefer:

```text
every Gaia-backed host now has explicit astrometric epoch knowledge
```

rather than:

```text
the epoch gate is open for every Gaia-backed host
```

because publication still depends on the requested result.

For example:

```text
REFERENCE_EPOCH_ONLY
    -> exact at reference epoch only

DIRECTION_ONLY
    -> no strict cross-epoch 3D Cartesian publication

FULL_SPACE_MOTION
    -> eligible for strict cross-epoch 3D propagation
```

That wording will prevent a future caller from treating all Gaia-backed rows as equally publishable.

---

# 13. Gaia cache/provider — additional guards worth keeping

Your reported provider design is good.

Before push, make sure tests include:

```text
[ ] returned Gaia source_id exactly matches requested gaia_dr3_id
[ ] zero rows -> fail closed
[ ] multiple rows -> fail closed
[ ] ref_epoch mismatch -> fail closed
[ ] malformed source_id never passes through float
[ ] cache schema/release mismatch -> fail closed
[ ] interrupted refresh cannot replace valid cache with partial data
[ ] offline index performs no network call
```

Do not silently cone-match if an exact NASA Gaia ID is present but fails.

An identity failure is better than a plausible nearby wrong star.

---

# 14. Distance provenance — non-blocking future hardening

Preserve:

```text
parallax measurement
distance used by the application
their separate provenance
```

as distinct concepts.

For high-S/N nearby stars, inverse parallax may be numerically excellent.

For the general exoplanet catalogue, however:

```text
distance = 1/parallax
```

is not universally a research-grade distance estimator.

Do not architect C3.6 so that Gaia parallax and "true distance" become permanently synonymous.

This is not a blocker for the six validated hosts if their chosen distance source is explicit and appropriate.

---

# 15. CRLF issue — handle separately

Yes: **handle `coordinates/inspector.py` line-ending normalization separately.**

Do not let the C3.6 science commit contain a full-file rewrite whose only cause is:

```text
CRLF -> LF
```

because that makes the most important coordinate-science diff harder to audit and `git blame` noisier.

Preferred local history:

```text
commit A:
Normalize coordinates/inspector.py line endings

commit B:
Explorer C3.6: astrometric epoch and space motion
```

If useful, add a small `.gitattributes` rule in the normalization commit:

```gitattributes
*.py text eol=lf
```

but do not run a repository-wide renormalization unless you deliberately want a separate formatting milestone.

The key requirement is: C3.6's semantic diff should remain readable.

---

# 16. What does not need changing

Do **not** reopen these:

```text
ResolvedOrientation
C4
HR diagram
blackbody
spectroscopy UI
gravity
free-flight camera
```

`ResolvedOrientation` remains a reasonable later refactor, not a C3.6 blocker.

The C2/C3/C3.5 test edits you described are also legitimate consequences of making the old hidden assumptions explicit.

---

# 17. Required pre-push work

Before I approve C3.6 for push:

```text
1. Harden DIRECTION_ONLY so missing RV remains visibly/model-wise missing.
2. Prevent DIRECTION_ONLY chaining from manufacturing radial-velocity knowledge.
3. Make the planet orbital state's obstime structural.
4. Verify all three times in planet->star distance.
5. Pin one explicit target-Time -> orbital-clock conversion.
6. Keep Gaia ref_epoch as an actual TCB Time object.
7. Split CRLF normalization from the science commit.
8. Re-run the full suite and real Mesa checks.
```

Then report:

```text
normalization commit SHA
C3.6 commit SHA
full test count
C3.6 test count
verify_gl.py result
demo/frame count
```

Do **not** push yet.

---

# 18. Scientific references used for this audit

Astropy documents that `SkyCoord.apply_space_motion()` assumes radial velocity `0` when no RV is supplied and evolves positions under straight-line constant velocity:

https://docs.astropy.org/en/latest/coordinates/apply_space_motion.html

Gaia DR3 documentation states:

```text
reference epoch: J2016.0
time coordinate: TCB
positions/proper motions: ICRS
pmra = mu_alpha * cos(dec)
```

Sources:

https://www.cosmos.esa.int/web/gaia/dr3

https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html
