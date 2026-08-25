# Explorer C3.6 Second Local Audit — One Remaining Absolute-Orientation Gate

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote baseline verified:** `1508738d3d14888e3799012a9f57a4d3d6d47e1f`  
**Normalization commit:** `63a4d555bfe255c0834d0142debaf8b7bad34c6d`  
**C3.6 commit:** `e39b164f341feac02689551a3c9f847995b7c0c4`  
**Reported local suite:** `992 passed, 1 skipped`  
**Verdict:** **Seven/eight prior concerns are closed cleanly. Do not push yet: one P0 publication gate remains — complete physical orbital orientation, not just node orientation, must be required for an absolute celestial planet position.**

---

## 1. Remote baseline is unchanged

The public `3D-test` branch is still at:

```text
1508738d3d14888e3799012a9f57a4d3d6d47e1f
```

so the two local commits remain safely unpushed.

The line-ending split is exactly the right history shape:

```text
1508738  C3.5.1 remote baseline
    ↓
63a4d55  line-ending normalization
    ↓
e39b164  Explorer C3.6 astrometric epoch and space motion
```

Keep that separation.

---

# 2. DIRECTION_ONLY hardening — PASS

The new policy addresses the Astropy missing-RV issue correctly.

Accepted invariants:

```text
DIRECTION_ONLY:
    proper motion exists
    radial velocity remains UNKNOWN
    propagated direction is explicitly model-dependent
    zero-RV approximation is disclosed
    no 3D absolute position publication
    no planet->star exact distance publication
```

The particularly important fix is:

```text
PropagatedAstrometry.as_state()
    does NOT read Astropy's zero-RV realization back as measured motion
```

and:

```text
propagate_astrometry()
    rebases DIRECTION_ONLY chains onto observed_root()
```

so repeated propagation cannot manufacture radial-velocity knowledge.

The ten-hop chain test is exactly the kind of adversarial regression this needed.

The `-100/+100 km/s` high-proper-motion stress case is also excellent: it demonstrates physically that missing RV leaves a family of possible future sky directions rather than one observation-determined answer.

---

# 3. TimedOrbitalState and common-time contract — PASS

Making the planet epoch structural closes the second major concern from the previous audit.

Accepted architecture:

```text
TimedOrbitalState
    physical state vector
    astropy Time
    PhaseSolution
```

A bare vector no longer qualifies for absolute-position publication.

Likewise:

```text
SystemSlice.absolute_planet_position(record, t)
SystemSlice.planet_to_star_distance(record, t, other)
```

derive the relevant states from one target instant rather than trusting callers to coordinate several unrelated values.

The three mismatch permutations are the right regressions.

Using:

```text
same_instant()
```

instead of comparing naked JD numbers is also correct because two different time scales can encode the same physical instant.

---

# 4. DIRECTION_ONLY chaining / FULL_SPACE_MOTION chaining — PASS

The distinction is now appropriate:

```text
DIRECTION_ONLY
    rebases to observation
    missing RV remains missing

FULL_SPACE_MOTION
    may carry the propagated velocity forward
```

That prevents a model realization from being promoted to an observation while retaining correct perspective behavior for complete six-dimensional states.

---

# 5. One orbital-clock conversion point — PASS

Centralizing the conversion in:

```text
physics/epoch.py::orbital_time_jd(Time, TimeScale)
```

is the correct solution.

The target instant is precise, but it does not magically determine an unstated source time system.

So:

```text
JD_UNSPECIFIED
```

continues to carry its existing time-scale uncertainty rather than becoming exact merely because the new Astropy target time is exact.

Keep the inverse relationship between:

```text
astropy_time(...)
orbital_time_jd(...)
```

under regression.

---

# 6. Gaia reference epoch — PASS

The Gaia source epoch remains a real:

```python
Time(ref_epoch, format="jyear", scale="tcb")
```

rather than being reduced to a validation boolean.

Testing:

```text
scale == TCB
J2016.0 physical instant
```

for every cached source is correct.

The identity/cache hardening is also strong:

```text
exact requested source_id
duplicate row -> fail closed
wrong source_id -> fail closed
schema/release mismatch -> fail closed
atomic-cache preservation
no cone fallback
```

---

# 7. CRLF separation — PASS

The normalization split should stay exactly as it is.

A science commit should not make `coordinates/inspector.py` look like a complete rewrite merely because Git normalized the line endings.

The `.gitattributes` rule:

```gitattributes
*.py text eol=lf
```

is reasonable if it is confined to the normalization commit and does not cause an accidental repository-wide renormalization.

---

# 8. The judgment call on `is_observationally_anchored`

Your choice is **correct for the time gate itself**.

A `PARTIALLY_CONSTRAINED` transit phase can have a real observed temporal anchor even when `omega` was normalized.

The existing phase model explicitly says exactly this:

```text
timing observed
orbital orientation normalized for display
```

Therefore:

```text
is_observationally_anchored
```

is the appropriate answer to:

> Is the planet's phase tied to a real observation in time?

It is **not**, by itself, the appropriate answer to:

> Is the planet's unique absolute 3D celestial position known?

Those are separate gates.

---

# 9. P0 before push: the current description only mentions node gates, but absolute position needs the rest of orientation too

This is the remaining issue.

You wrote:

> the timing is a real observation, and its orientation is already gated twice by the node checks.

The node checks are necessary but not sufficient.

A unique absolute planet vector also depends on the physical orientation encoded by:

```text
inclination i
planet-frame argument of periapsis omega
longitude/node convention and node sense Omega
```

The existing project model already recognizes this distinction.

`OrbitalElements.orientation_known` requires all three orientation angles to be scientific, and `OrbitValidity.ORIENTATION_FULL` additionally requires the periastron convention to be determinate.

The phase model separately defines `PARTIALLY_CONSTRAINED` precisely for the case where the temporal anchor is observed while some orientation was normalized.

So C3.6 must not let:

```text
observed phase anchor
+
resolved Omega
```

stand in for:

```text
fully sufficient physical orientation
```

---

# 10. Concrete failure modes

## A. Transit epoch + unknown omega + eccentric orbit

For a transit:

\[
u = \omega + \nu = \pi/2
\]

at the anchor instant.

If `omega` is unknown and normalized for display, the time of transit is real, but for an eccentric orbit the mapping:

\[
\nu \rightarrow E \rightarrow M
\]

depends on `omega`.

After advancing by:

\[
M(t)=M_{\rm transit}+n(t-t_{\rm transit})
\]

the unique physical position at a later time is not determined without the real `omega`.

So `PARTIALLY_CONSTRAINED` may be suitable for visualization and must **not** by itself unlock a unique ICRS position.

## B. Periastron epoch + missing omega

A periastron epoch gives:

```text
M = 0
```

exactly.

That is why the phase object correctly calls the temporal phase `CONSTRAINED` even without `omega`.

But an absolute 3D position still needs to know **which direction periapsis points** in the orbital plane.

So:

```text
PhaseStatus.CONSTRAINED
```

also cannot substitute for complete orientation.

## C. Missing inclination

Even a perfect node and a perfect temporal phase do not locate the orbital plane in 3D if `i` was merely normalized for visualization.

---

# 11. Required publication gate

Add a separate orientation-publication gate.

Conceptually:

```text
absolute position requires:

shape/state physically available
+
temporal phase observationally anchored
+
inclination scientifically constrained
+
periapsis orientation scientifically constrained when physically relevant
+
periastron convention determinate
+
node convention stated
+
node sense resolved
+
non-pole tangent frame
+
common coordinate epoch / astrometry
```

Do not merge all of this into `PhaseStatus`.

Phase and orientation are deliberately separate epistemic dimensions in this codebase.

---

# 12. Recommended implementation

A small helper is enough.

For example:

```python
absolute_orientation_blockers(elements) -> tuple[str, ...]
```

with reasons such as:

```text
INCLINATION_UNRESOLVED
PERIAPSIS_DIRECTION_UNRESOLVED
PERIASTRON_CONVENTION_UNSTATED
NODE_CONVENTION_UNSTATED
NODE_SENSE_UNRESOLVED
```

Then:

```text
absolute_position_blockers
=
astrometric blockers
+ timing blockers
+ orientation blockers
```

Keep returning **all** reasons.

Do not just add:

```python
if not elements.orientation_known:
    return False
```

because the C3.5 node-sense/convention rules are now richer than the old boolean.

---

# 13. Circular-orbit nuance

Do not blindly require a meaningful `omega` for a truly circular orbit.

For:

\[
e = 0
\]

periapsis has no physical direction.

So the correct rule is more nuanced than:

```text
omega must always be measured
```

A good policy is:

```text
scientifically established circular orbit:
    periapsis direction is not required as a physical orientation degree of freedom

non-circular orbit:
    planet-frame omega / equivalent in-plane orientation must be scientifically constrained
```

Do **not** use:

```text
display-normalized e = 0
```

to obtain this exemption.

Only a scientifically supported circular model may make `omega` irrelevant.

If exact-zero versus effectively-circular classification is not yet modeled robustly, the conservative first implementation may simply keep such cases blocked and document the future refinement.

---

# 14. Required tests

Add at minimum:

```text
[ ] transit epoch + assumed omega + eccentric orbit does not publish absolute position
[ ] periastron epoch + unknown omega + eccentric orbit does not publish absolute position
[ ] missing/assumed inclination does not publish absolute position
[ ] measured i + resolved physical omega + resolved node can pass orientation gate
[ ] unstated periastron convention blocks
[ ] stellar-reflex omega converted to planet frame may pass as DERIVED
[ ] display-normalized omega never passes as physical orientation
[ ] display-normalized inclination never passes
[ ] timing blocker and orientation blockers are reported together
[ ] planet->star distance inherits the exact same orientation gate
```

If implementing the circular exception now:

```text
[ ] scientifically established e=0 does not require an arbitrary periapsis direction
[ ] display-assumed e=0 does not receive that exemption
```

---

# 15. Why the remote suite could not catch this yet

The current remote C3.5.1 code still deliberately withholds all real absolute positions on the coordinate-epoch gate.

So the full orientation-publication path has never been reachable for a real catalog planet.

C3.6 is the first slice that opens the astrometric side of that gate.

That is precisely when this latent separation between:

```text
phase knowledge
```

and:

```text
orientation knowledge
```

must become enforceable.

---

# 16. Review files — yes, archive them

Move the two untracked audit Markdown files from the repository root into:

```text
docs/reviews/archive/
```

Include them with the revised C3.6 commit.

Do not create a standalone remote documentation commit.

Because C3.6 is still local, this is the right moment to absorb them into the normal archive convention.

---

# 17. Commit plan

Keep:

```text
63a4d555bfe255c0834d0142debaf8b7bad34c6d
    Normalize Python line endings
```

unchanged.

Amend or replace only the unpushed C3.6 commit so it contains:

```text
current C3.6 implementation
absolute orientation publication gate
new regressions
archived audit Markdown files
```

Then report the revised C3.6 SHA.

No history rewrite is involved because none of these commits have been pushed.

---

# 18. Current verdict

```text
DIRECTION_ONLY model dependency              PASS
DIRECTION_ONLY no knowledge promotion        PASS
TimedOrbitalState                            PASS
three-way common-time contract               PASS
central orbital clock conversion             PASS
Gaia TCB reference epoch                     PASS
Gaia identity/cache guards                   PASS
CRLF commit separation                       PASS
observational phase-anchor judgment          PASS AS A TIME GATE
complete absolute-orientation gate           P0 REQUIRED
```

## Do not push yet.

After the orientation gate and tests are added:

1. rerun the full suite;
2. rerun `verify_gl.py`;
3. render all demos;
4. send the revised C3.6 SHA + counts;
5. then C3.6 should be ready for normal fast-forward push.
