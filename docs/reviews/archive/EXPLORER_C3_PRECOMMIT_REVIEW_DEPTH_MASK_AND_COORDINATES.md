# Explorer C3 Pre-Commit Review — Depth-Mask Recovery, Coordinate Semantics, and Commit Plan

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Recovered HEAD:** `5129422c16d0f20d1a8cfb680ca65a7e460268ee`  
**Working tree:** uncommitted C2 follow-ups + Explorer C3  
**Reported test count:** 804  
**Verdict:** **Recovery PASS. Depth-mask fix PASS. C3 is nearly ready, but one absolute-coordinate semantic issue must be corrected before commit. Split the work into two commits.**

---

## 1. Recovery check — PASS

The recovered repository state is trustworthy:

```text
HEAD = 5129422
matches origin/3D-test
git fsck clean except one harmless dangling tree
755-test C2 baseline reproduces exactly
```

Reproducing the previously audited:

```text
755 passed
```

is particularly strong evidence that the repository recovery did not silently alter code or data.

No recovery work is required.

---

## 2. Depth-mask bug — confirmed as a real bug

The bug you found is real.

The previously pushed backend used assignments such as:

```python
self.ctx.depth_mask = False
```

and:

```python
self.ctx.depth_mask = True
```

during overlay passes.

In ModernGL, depth-write masking belongs to the **Framebuffer**, not the `Context`.

The correct API is conceptually:

```python
framebuffer.depth_mask = False
```

ModernGL's own documentation defines:

```text
Framebuffer.depth_mask
```

as the control that enables/disables writes to the depth buffer.

Therefore the former:

```python
ctx.depth_mask = ...
```

was not changing OpenGL depth-write state.

The fact that the Python context object accepted the arbitrary attribute made this especially dangerous: the code and the old test could agree with each other while both being disconnected from actual GPU state.

This is a textbook false-green test.

---

## 3. Why the old test could not catch it

The old pattern was effectively:

```python
renderer.ctx.depth_mask = True
...
renderer.ctx.depth_mask = False
...
assert renderer.ctx.depth_mask is True
```

That only tested a Python attribute attached to the context object.

It did **not** test the actual framebuffer's writable-depth state.

The new regression:

```text
test_depth_mask_belongs_to_the_framebuffer_not_the_context
```

is exactly the sort of test required here.

Keep it permanent.

---

## 4. `_overlay_pass()` centralization — approve

Replacing three manually duplicated state blocks with one scoped overlay-state manager is a good correction.

The overlay pass should own all state it changes:

```text
blend enable/disable
blend function
framebuffer depth mask
line width
```

and restore those values after the pass.

The most important invariant is now structural:

```text
scientific overlay draw
    ↓
real framebuffer depth writes disabled
    ↓
draw
    ↓
previous framebuffer depth mask restored
```

This is substantially stronger than the previous convention.

---

## 5. C2 node-provenance follow-up — PASS

Replacing:

```text
Ascending node: ... (measured)
```

with the parameter's actual provenance closes the small semantic gap from the C2 audit.

A known node can now honestly remain:

```text
MEASURED
DERIVED
...
```

without presentation text silently promoting it to an observation.

The additional derived-node regression is worthwhile even if the current real snapshot does not naturally exercise that path.

---

## 6. Explorer C3 architecture — strong overall

The reported C3 model follows the correct rule:

> scientific coordinate/distance values are computed from float64 scientific state, not render geometry.

Strong decisions include:

- `InspectorRow` for scalar science values;
- `CoordinateRow` for triplets;
- frame name required structurally for every coordinate triplet;
- Astropy owns ICRS/Galactic transformations;
- Galactic direction remains available when distance is unknown;
- instantaneous host–planet distance uses the norm of the propagated physical position;
- the analytical identity \(r=a(1-e\cos E)\) is only an independent test;
- periapsis/apoapsis propagate input provenance;
- detached systems retain local coordinates while absolute quantities remain `UNKNOWN`;
- adversarial tests prove display exaggeration, LOD, orbit sampling, and camera placement cannot change scientific inspector values.

These are exactly the right C3 foundations.

---

## 7. P0 before commit: do not label an unrotated local offset as an ICRS planet position

Your own caution in Section 20 is correct, and I would go one step further.

The current orbital local vector lives in the program's system/orbital basis.

The orientation module defines the exoplanet reference plane as the sky plane, but its `+x` reference direction is an abstract reference direction used by the orbital transform.

The host's absolute Cartesian position, by contrast, lives in a global ICRS Cartesian basis.

Therefore this operation:

\[
\mathbf r_{\rm host,ICRS}
+
\mathbf r_{\rm planet,SystemFrame}
\]

is **not a valid vector addition** unless an explicit basis transform exists:

\[
R_{\rm SystemFrame\rightarrow ICRS}
\]

and has been applied to the local vector first.

The issue is not merely that \(\Omega\) is usually unknown.

Even when \(\Omega\) is known, the code needs a formally defined mapping between:

```text
SystemFrame +x/+y/+z
```

and:

```text
ICRS Cartesian basis
```

including the host's local tangent basis and the reference direction from which \(\Omega\) is measured.

So:

> downgrading the row to `ASSUMED_FOR_VISUALIZATION` is honest about uncertainty, but calling the resulting triplet `ICRS` is still geometrically misleading.

---

## 8. Recommended C3 fix

For C3, **do not publish an absolute planet ICRS coordinate yet**.

Keep:

```text
host absolute ICRS position        available when located
planet local SystemFrame x/y/z     available when phase is computable
host ↔ planet scalar distance      available from physics
```

but set:

```text
planet absolute ICRS position      NOT AVAILABLE YET
```

until a real basis transform exists.

Possible inspector row:

```text
Absolute planet position:
not resolved — SystemFrame→ICRS orientation is not fully defined
```

This is better than publishing a numerically dominated but basis-inconsistent coordinate.

At 66 pc, the AU-scale offset is tiny compared with the host vector, but **small error is not the same as correct coordinates**.

Do not let scale hide a basis error.

---

## 9. What a later true absolute planet ICRS transform needs

A correct transform needs an explicit local triad at the host.

For example, using host ICRS coordinates \((\alpha,\delta)\), construct a tangent basis such as:

```text
e_r      radial line of sight
e_east   increasing RA direction
e_north  increasing Dec direction
```

Then define exactly what the orbital-reference `+x` axis means relative to that tangent basis.

Only then can the orbital vector be mapped:

\[
\mathbf r_{\rm local}
\rightarrow
\mathbf r_{\rm ICRS}
\]

and added to the host:

\[
\mathbf r_{\rm planet,ICRS}
=
\mathbf r_{\rm host,ICRS}
+
R_{\rm local\rightarrow ICRS}
\mathbf r_{\rm local}
\]

If \(\Omega\) is unknown, the radial distance from host to planet remains physical, but the absolute sky-plane direction remains unconstrained.

This deserves its own explicit milestone rather than being smuggled into C3.

---

## 10. C3 coordinate semantics after that fix

Recommended C3 output:

### Host

```text
RA / Dec                   ICRS
Galactic l / b             Galactic
catalog distance           pc
ICRS Cartesian x/y/z       pc
Earth/Sun ↔ host           pc / ly
```

### Planet

```text
SystemFrame x/y/z          AU
host ↔ planet distance     AU
periapsis                  AU
apoapsis                   AU
phase provenance           enum/text
absolute ICRS position     unavailable unless transform fully constrained
```

For a detached system:

```text
host absolute position     UNKNOWN
Earth ↔ host distance      UNKNOWN
planet local x/y/z         may be known
host ↔ planet distance     may be known
```

This is scientifically clean.

---

## 11. Galactic direction without distance — PASS

Your treatment here is correct.

A sky direction can be transformed:

```text
ICRS RA/Dec
    ↔
Galactic l/b
```

without a radial distance.

So a star may have:

```text
known sky direction
unknown parallax/distance
```

and still legitimately expose:

```text
l
b
```

while Cartesian Galactic coordinates remain unavailable.

Keep the distinction explicit.

---

## 12. Periapsis/apoapsis provenance — PASS

The provenance propagation rule is correct.

If:

```text
a = ASSUMED_FOR_VISUALIZATION
e = MEASURED
```

then:

\[
r_{\rm peri}=a(1-e)
\]

must not become a plain `DERIVED` scientific quantity that visually hides the assumption.

The result should preserve the weakest/most assumption-dependent provenance of its required inputs.

Likewise for:

\[
r_{\rm apo}=a(1+e)
\]

Keep tests covering:

```text
measured + measured -> derived
derived + measured -> derived
assumed + measured -> assumed
unknown input -> unknown result
```

or the equivalent status policy used by the project.

---

## 13. Host–planet instantaneous distance — PASS

Using:

```python
np.linalg.norm(physical_position_float64)
```

as the production value is the correct implementation.

Using:

\[
r=a(1-e\cos E)
\]

only as an independent regression is excellent because it avoids building two production calculations that could later disagree.

For HD 80606 b, test at least:

```text
periapsis
intermediate phase
apoapsis
```

to preserve the extreme dynamic range.

---

## 14. C3 adversarial display-invariance tests — excellent

The reported tests that deliberately change:

```text
display exaggeration
LOD
orbit sampling
camera placement
```

and first assert that the scenes actually differ before asserting that inspector values do not change are particularly strong.

That avoids a weak test of the form:

```text
change setting
setting accidentally changes nothing
science stays same
PASS
```

Keep this pattern.

---

## 15. Commit strategy: split the work into two commits

Do **not** make all of this one C3 commit.

The depth-mask fix is an independent rendering-core bug that affects previously closed C1/C2 behavior.

Make two commits.

### Commit 1 — corrective rendering commit

Suggested message:

```text
Fix framebuffer depth writes for scientific overlays
```

Include:

```text
_overlay_pass()
Framebuffer.depth_mask fix
C2 node-provenance annotation fix
GL contract regressions
depth-mask ownership regression
```

Do not include C3 feature files here.

This creates a clean bisect point for the real rendering bug.

### Commit 2 — Explorer C3

Suggested message:

```text
Explorer C3: coordinate and distance inspector
```

Include:

```text
coordinates/inspector.py
C3 regression tests
docs/explorer-c.md C3 section
CI C3 step
archived C2/C3 review docs as appropriate
```

Before creating this second commit, remove/disable the basis-inconsistent absolute planet ICRS row.

---

## 16. Why two commits are better

If a future render regression appears, Git history should be able to answer:

```text
Did framebuffer state change break this?
```

without mixing that question with:

```text
Did the coordinate inspector break this?
```

Likewise, the C3 scientific feature should remain independently reviewable.

You can still push both commits together in one fast-forward later after review.

---

## 17. Verification before asking for push approval

After the two commits exist locally, run:

```text
architecture / golden-rule tests
full suite
offline check
verify_gl.py
GL backend tests
C1 scientific overlay tests
C2 orientation tests
C3 coordinate/distance tests
all demos
frame-count assertion
```

I would also add one explicit visual regression that proves depth-mask behavior is real:

```text
an orbit/guide behind a translucent zone remains visible as intended
```

or another pixel-level arrangement where incorrect depth writes produce a different result.

The ownership test proves the API is correct; a behavioral pixel test proves the rendering consequence is correct.

---

## 18. C3 acceptance status

Based on the local report:

```text
repository recovery                         PASS
C2 node provenance                         PASS
Framebuffer depth-mask ownership           PASS
overlay state centralization               PASS
scientific float64 inspector architecture  PASS
frame-required CoordinateRow               PASS
Astropy frame transforms                   PASS
direction without distance                 PASS
host/planet scalar distance                PASS
peri/apo provenance                        PASS
detached semantics                         PASS
display-invariance adversarial tests       PASS
absolute planet ICRS basis                 FIX BEFORE COMMIT
commit separation                          REQUIRED
remote verification                        PENDING
```

---

## 19. Immediate instruction

1. **Keep the depth-mask fix. It is correct and important.**
2. **Remove or mark unavailable the absolute planet ICRS coordinate until a real SystemFrame→ICRS basis transform exists.**
3. **Split the working tree into two commits: rendering fix, then C3 feature.**
4. Run the complete 804+ suite and real Mesa verification again.
5. Send both local SHAs and the final test count.
6. Then I will approve the fast-forward push.

Do not push yet.
