# Explorer C3 Final Remote Audit and Next Coordinate-Physics Slice

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote HEAD:** `f470cfd4e42d2ed6dce02039226bbd22aa43672b`  
**Corrective parent commit:** `b8b44572ca607e57fa6c9dadaefd22be5883ae86`  
**CI run:** `32653714483`  
**Verdict:** **EXPLORER C3 CLOSED — PASS**

---

## 1. Remote state — verified

The remote `3D-test` branch now points exactly to:

```text
f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

with the intended history:

```text
5129422  Explorer C2
    ↓
b8b4457  Fix framebuffer depth writes for scientific overlays
    ↓
f470cfd  Explorer C3: coordinate and distance inspector
```

The split requested in the pre-commit review survived the push intact.

---

## 2. Remote CI — verified green

GitHub Actions run:

```text
32653714483
```

completed successfully.

The remote jobs passed:

```text
Tests (Python 3.11)             PASS
Tests (Python 3.12)             PASS
OpenGL 3.3 core (software Mesa) PASS
```

The Python jobs explicitly ran:

```text
architecture / golden-rule tests
Explorer C3 coordinate/distance tests
full suite
offline four-system check
```

The OpenGL job ran:

```text
GL 3.3 context and shader verification
OpenGL backend tests
multi-planet rendering tests
Explorer navigation/picking/labels
scientific overlay tests
real render demos
frame existence assertion
artifact upload
```

A non-expired `rendered-frames` artifact exists for the exact C3 head SHA.

---

# 3. No separate remote workflow run is needed for `b8b4457`

Do **not** trigger a manual workflow merely to give `b8b4457` its own remote Actions badge.

Why:

1. `b8b4457` is the direct parent of the remotely tested C3 tip.
2. The C3 CI checkout contains the exact framebuffer fix.
3. The full suite and GL backend tests exercised the corrected depth-mask regressions at the remote tip.
4. The corrective commit was also independently checked locally in an isolated worktree with its own 763-test state.
5. A separate temporary branch/run would add process noise but little additional evidence.

The meaningful question is not:

> did GitHub attach a badge to the intermediate SHA?

It is:

> was the corrective code independently bisectable, and did the final remotely tested tree exercise it?

The answer to both is yes.

Keep `b8b4457` as the clean bisect point; no extra workflow run is necessary.

---

# 4. Framebuffer depth-mask fix — remotely present and accepted

The corrective commit explicitly replaces the invalid context-level pseudo-state with the framebuffer-owned depth mask.

The remote commit documents that:

```text
Context.depth_mask
```

was only an inert arbitrary Python attribute, while:

```text
Framebuffer.depth_mask
```

is the real depth-write control.

The corrected backend now scopes overlay state through a shared:

```text
_overlay_pass()
```

and preserves/restores the framebuffer depth mask.

The two complementary regressions are exactly right:

```text
test_depth_mask_belongs_to_the_framebuffer_not_the_context
    -> API ownership

test_an_orbit_behind_a_translucent_zone_still_shows_through
    -> visible rendering consequence
```

That closes the false-green test class that survived C1/C2.

---

# 5. C3 inspector architecture — PASS

The pushed inspector obeys the correct invariant:

> every scientific coordinate and distance comes from float64 scientific state, never from display geometry.

The module does not depend on the renderer.

Accepted structure:

```text
InspectorRow
    scalar scientific parameter

CoordinateRow
    x/y/z + mandatory frame + unit + provenance

NoteRow
    non-numeric scientific qualifier such as phase provenance
```

`CoordinateRow` structurally refuses a frame-less triplet.

That is better than relying on the UI to remember to label coordinates later.

---

# 6. Astropy owns celestial transforms — PASS

C3 does not introduce another RA/Dec transformation implementation.

The celestial path remains:

```text
scientific SkyPosition
    ↓
Astropy SkyCoord
    ↓
ICRS / Galactic
```

A star with a known direction but unknown radial distance may legitimately expose:

```text
RA / Dec
Galactic l / b
```

while withholding Cartesian position.

This distinction is correct.

---

# 7. Host–planet distance — PASS

The production instantaneous separation comes from the norm of the propagated float64 physical state:

\[
r = \left|\mathbf r_{\rm planet,local}\right|
\]

The identity:

\[
r=a(1-e\cos E)
\]

is retained only as an independent regression.

This is excellent because it preserves:

```text
one physical propagator
one production answer
one independent mathematical check
```

rather than two competing production implementations.

HD 80606 b remains the right high-eccentricity stress case.

---

# 8. Periapsis / apoapsis provenance — PASS

C3 correctly prevents exact arithmetic from laundering an assumption into a quotable scientific result.

For:

\[
r_{\rm peri}=a(1-e)
\]

and:

\[
r_{\rm apo}=a(1+e)
\]

the result inherits the relevant epistemic weakness of the inputs.

Examples:

```text
measured/derived physical inputs
    -> DERIVED result

ASSUMED_FOR_VISUALIZATION input
    -> ASSUMED_FOR_VISUALIZATION result

UNKNOWN input
    -> UNKNOWN result
```

Keep this policy.

---

# 9. Detached-system behavior — PASS

TRAPPIST-1 continues to act as a useful honesty test.

A detached/unlocated system can report:

```text
local SystemFrame coordinates
host↔planet separation
orbital information
```

while absolute values remain:

```text
UNKNOWN
```

It never becomes:

```text
0 pc
```

or the Solar origin.

That extends the Explorer B rule correctly into C3.

---

# 10. Absolute planet celestial position — correctly withheld

The pushed C3 implementation now refuses to publish:

```text
planet absolute ICRS x/y/z
```

because the program still lacks an explicit:

\[
R_{\rm SystemFrame\rightarrow ICRS}
\]

basis transformation.

This is the correct decision.

The row remains in the inspector only as an unresolved result so the UI can explain:

```text
why the coordinate is unavailable
```

instead of displaying a bare unknown.

The important principle is now enforced:

> a basis error is not converted into an uncertainty label.

Even a measured longitude of ascending node does not, by itself, define the missing local-to-global basis mapping.

---

# 11. All-planets celestial-triplet guard — excellent

Keep permanently:

```text
test_no_published_planet_triplet_claims_a_celestial_frame
```

This protects the architecture against the same basis mistake reappearing through another API.

It is stronger than a one-system regression because it sweeps every published planet in every located committed system.

---

# 12. Phase provenance row — PASS

Adding phase provenance to the inspector row model is important.

A row such as:

```text
Distance from host: 0.2057 AU
```

must travel with whether that instantaneous state is:

```text
CONSTRAINED
PARTIALLY_CONSTRAINED
ASSUMED
```

The shared row-comparison helper also prevents future row types from silently dropping out of invariance tests.

---

# 13. Duplicate review file — remove the root copy

You now have two byte-identical copies of the push-approval review:

```text
repo root
docs/reviews/archive/
```

Keep the archived copy.

Delete the untracked root duplicate.

Do not create another commit solely for that deletion if it is untracked.

Desired local state:

```text
docs/reviews/archive/<review>.md    retained
/<review>.md                        removed
```

The archive README already says archived reviews are historical and current `docs/` wins on disagreement, which is the right authority policy.

---

# 14. Roadmap status is now stale and should be updated with the next slice

The pushed `docs/roadmap-status.md` still says:

```text
C3 distance and coordinate inspector | not started
C4 scientific plot integration       | not started
```

C3 is now remotely complete.

Update this with the next coordinate-physics commit rather than making a docs-only push.

Recommended new entries:

```text
C3 distance and coordinate inspector
    DONE
    coordinates/inspector.py
    tests/regression/test_explorer_c3.py

C3.5 SystemFrame -> ICRS basis
    NEXT
    required for absolute planet celestial position

planet -> selected star distance
    BLOCKED ON C3.5
```

Keep C4 as the scientific plot integration milestone:

```text
HR diagram
blackbody
atmospheric spectra
```

This avoids silently changing the numbering of the existing roadmap.

---

# 15. Recommended next slice: Explorer C3.5 — SystemFrame → ICRS basis

Do this **before C4**.

It is a small coordinate-physics prerequisite, not UI polish.

The goal is to define a mathematically explicit local tangent frame at the host.

Given host ICRS right ascension \(\alpha\) and declination \(\delta\), construct an orthonormal local basis such as:

\[
\hat e_r =
\begin{bmatrix}
\cos\delta\cos\alpha\\
\cos\delta\sin\alpha\\
\sin\delta
\end{bmatrix}
\]

\[
\hat e_{\rm east} =
\begin{bmatrix}
-\sin\alpha\\
\cos\alpha\\
0
\end{bmatrix}
\]

\[
\hat e_{\rm north} =
\begin{bmatrix}
-\sin\delta\cos\alpha\\
-\sin\delta\sin\alpha\\
\cos\delta
\end{bmatrix}
\]

Then explicitly define how the program's orbital-reference axes map into this triad.

A common sky-plane convention could be something like:

```text
SystemFrame +z   = line of sight / chosen radial direction
SystemFrame +x   = chosen tangent reference direction
SystemFrame +y   = completes right-handed basis
```

but **do not choose the mapping casually**.

It must match the convention assumed by:

\[
R_z(\Omega)R_x(i)R_z(\omega)
\]

and by the interpretation of the catalogued longitude of ascending node.

---

# 16. The line-of-sight sign must be explicit

This is a classic source of 180° mistakes.

Depending on convention:

```text
+z
```

may point:

```text
observer -> star
```

or:

```text
star -> observer
```

Those are opposite.

Document it and test it.

The SystemFrame→ICRS mapping should state explicitly:

```text
which direction +z points
what +x means on the sky
what increasing Omega means
whether the basis is right-handed
```

Do not infer this later from a render that "looks right."

---

# 17. C3.5 should distinguish known and unknown Ω

Once the tangent basis exists:

### Ω measured

The full orbital sky-plane orientation can be mapped into ICRS.

Then the planet's absolute instantaneous vector can be formed:

\[
\mathbf r_{\rm planet,ICRS}
=
\mathbf r_{\rm host,ICRS}
+
R_{\rm System\rightarrow ICRS}
\mathbf r_{\rm planet,System}
\]

with explicit unit conversion.

### Ω unknown

The scalar host↔planet distance remains known.

But the absolute azimuth about the line of sight remains unconstrained.

Therefore:

```text
absolute planet ICRS position
```

must remain unavailable as a unique observed coordinate unless the user explicitly requests a normalized display realization.

A normalized realization must not be promoted to a physical catalog coordinate.

---

# 18. Planet→selected-star distance should return after C3.5

Once a physically meaningful absolute planet vector exists, restore the original requested feature:

\[
D =
\left|
\mathbf r_{\rm selected\ star,ICRS}
-
\mathbf r_{\rm planet,ICRS}
\right|
\]

This should be available only when:

```text
host absolute position is known
planet phase is available
required SystemFrame→ICRS orientation is sufficiently constrained
selected star absolute position is known
```

Otherwise return `UNKNOWN` with a reason.

Do not silently fall back to host↔star distance.

---

# 19. C3.5 acceptance tests

At minimum:

```text
[ ] local tangent triad is orthonormal
[ ] basis determinant is +1
[ ] e_r points exactly along host ICRS radial direction
[ ] east/north directions agree with Astropy finite-difference checks
[ ] SystemFrame→ICRS→SystemFrame round trip
[ ] zero local offset returns host absolute coordinate
[ ] 1 AU radial offset changes only expected radial component
[ ] 1 AU east offset follows local tangent east
[ ] 1 AU north offset follows local tangent north
[ ] known Omega orientation maps consistently with orbital transform
[ ] unknown Omega cannot produce unique physical planet ICRS position
[ ] normalized Omega remains ASSUMED_FOR_VISUALIZATION
[ ] absolute planet coordinate never uses float32 render state
[ ] planet→selected-star distance agrees with norm in common float64 frame
[ ] detached host still has no absolute planet coordinate
[ ] all C3 tests remain green
```

Also compare key transformations against Astropy wherever Astropy can provide an independent reference.

---

# 20. No manual run on `b8b4457`

For the record:

```text
manual workflow_dispatch on b8b4457
```

is **not requested**.

The final remote CI run on `f470cfd` already exercised the framebuffer fix and its regressions.

The local isolated-worktree verification preserves the intermediate commit as a meaningful bisect point.

That is enough.

---

# 21. C3 final status

```text
remote HEAD                              PASS
history structure                        PASS
corrective parent commit                 PASS
Python 3.11 CI                           PASS
Python 3.12 CI                           PASS
OpenGL software Mesa CI                  PASS
C3 named CI step                         PASS
render artifact                          PASS
real framebuffer depth-mask fix          PASS
behavioral depth regression              PASS
float64 inspector                        PASS
Astropy celestial transforms             PASS
detached-system honesty                  PASS
peri/apo provenance                      PASS
phase provenance row                     PASS
absolute planet ICRS withheld            PASS
all-planets celestial-frame guard        PASS
C3 roadmap status                        UPDATE NEXT COMMIT
SystemFrame→ICRS basis                   NEXT SLICE
```

## **Explorer C3 is CLOSED.**

---

# Immediate actions

1. Delete the untracked duplicate review from the repository root.
2. Keep the archived copy under `docs/reviews/archive/`.
3. Do **not** run a separate CI workflow for `b8b4457`.
4. Start **Explorer C3.5 — SystemFrame→ICRS basis**.
5. In that commit, update `docs/roadmap-status.md` to mark C3 done and add C3.5 + the blocked planet→star distance item.
6. Keep C4 reserved for HR/blackbody/spectroscopy integration.
