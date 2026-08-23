# Explorer C2 Local Review and Push Approval

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Local commit reported:** `5129422`  
**Remote tip verified before push:** `4ab8e54adfe5ae0d1df5eb03eab5d86d087c2ff5`  
**Reported test count:** 755  
**Verdict:** **C2 PASS LOCALLY — APPROVED TO PUSH**

---

## 1. Remote state before the push

The remote `3D-test` branch is still at the Explorer C1 commit:

```text
4ab8e54adfe5ae0d1df5eb03eab5d86d087c2ff5
```

So the reported C2 commit remains local at review time.

Use a normal fast-forward push only.

Recommended preflight:

```bash
git fetch origin
git merge-base --is-ancestor origin/3D-test HEAD
```

then:

```bash
git push origin HEAD:3D-test
```

Do not force-push.

---

# 2. C1 follow-ups — approved

All four follow-ups from the C1 remote audit are resolved in the right place.

### Habitable-zone wording

The new wording correctly describes the 2D band as a:

```text
reference-plane cross-section
```

through a physically radial/spherical shell.

This prevents the visualization from implying that the HZ exists only in the system reference plane.

### Kepler-11 wording

The phrase:

```text
lies starward of the inner HZ boundary
```

is much better than:

```text
inside the inner edge
```

because it cannot be confused with "inside the habitable zone."

### Review archive

Moving the generated C1 review documents into:

```text
docs/reviews/archive/
```

matches the repository's documentation-authority policy.

### `edge_color`

Implementing the HZ boundary rings rather than deleting `edge_color` was the better choice.

The resulting batching model:

```text
all HZ fills -> one TRIANGLES draw
all HZ boundaries -> one LINES draw
```

is clean and preserves the visible meaning of the scientific inner/outer limits.

The regression that verifies `edge_color` changes real rendered pixels is particularly valuable because it proves the field is not merely passed around.

The GL-state restoration test should remain permanent.

---

# 3. C2 architecture — PASS

The most important C2 decision is correct:

> The renderer receives finished guide geometry, not orbital elements.

The render contract:

```text
RenderGuide
GuideStyle.SOLID | GuideStyle.DASHED
polyline geometry
stroke
colour
label
```

contains no raw:

```text
inclination
argument of periapsis
longitude of ascending node
eccentricity
periastron convention
scientific status
```

That preserves the project's central firewall.

The correct flow is:

```text
catalogue data
    ↓
resolved orbital semantics
    ↓
physics/orientation.py
    ↓
finished guide vectors
    ↓
scene builder
    ↓
RenderGuide
    ↓
OpenGL
```

Do not weaken this later.

---

# 4. Reusing the production orbital transform — PASS

Guide geometry uses the same rotation convention as the orbital propagator:

\[
\mathbf r =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf r_{pf}
\]

This is exactly what C2 needed.

There must never be:

```text
physics interpretation of i/ω/Ω
+
visualization interpretation of i/ω/Ω
```

as two independent implementations.

Using a shared orientation/transform implementation prevents this class of drift.

---

# 5. DERIVED as solid — APPROVED

Your first judgment call is correct.

A value such as:

```text
stellar-reflex ω
    ↓
+180°
    ↓
planetary ω
```

is not a guess when the source convention is known.

It is a deterministic transformation of measured/published information.

Therefore:

```text
DERIVED -> solid
```

is scientifically defensible.

The important requirement is that the legend/panel still says:

```text
DERIVED
```

so solid must not mean "directly measured."

Recommended semantic interpretation:

```text
solid
    = scientifically constrained geometry

dashed
    = geometry depends on a visualization assumption
```

This is better than using line style as a direct copy of the provenance enum.

---

# 6. Measured inclination + normalized azimuth — APPROVED, with one conceptual rule

Your second judgment call is also sound.

Suppose:

```text
i = MEASURED
Ω = UNKNOWN
Ω_display = 0°
```

Then the **tilt magnitude** is observationally constrained, but its absolute azimuth around the reference axis is not.

A plane embedded into the 3D scene using `Ω_display = 0°` therefore depends on an assumption.

Drawing that plane dashed is appropriate.

However, preserve this conceptual distinction in the panel/docs:

```text
inclination magnitude
    MEASURED

absolute azimuth of the displayed plane
    ASSUMED_FOR_VISUALIZATION
```

Do not let the dashed plane imply that the inclination measurement itself is uncertain or fabricated.

A future richer UI could show the measured inclination angle separately from the assumed plane azimuth, but C2 does not require that additional complexity.

---

# 7. `show_normalised=False` by default — PASS

This is the correct default.

When Ω is unknown, the most scientifically conservative default visualization is:

```text
do not draw an absolute node/orientation guide
```

rather than automatically normalizing it.

Only when the user explicitly enables normalized orientation should the renderer show the assumed geometry.

Then the scene must disclose:

```text
Ω is observationally unknown.
The displayed azimuth is normalized to Ω = 0° for visualization.
```

This is exactly the right distinction between:

```text
UNKNOWN
```

and:

```text
ASSUMED_FOR_VISUALIZATION
```

---

# 8. Styling table — approved

The current policy is coherent:

| Scientific state | Guide |
|---|---|
| `MEASURED` | solid |
| `DERIVED` | solid + labelled derived |
| `ASSUMED_FOR_VISUALIZATION` | dashed + textual disclosure |
| `UNKNOWN` | absent unless normalized display explicitly enabled; normalized geometry dashed |

Keep the additional rule:

> Provenance must never be communicated by colour alone.

That improves both accessibility and scientific clarity.

---

# 9. One selected orbit at a time — PASS

Showing detailed orientation guides only for the selected planet is the better UX.

Drawing:

```text
plane
normal
node
periapsis direction
inclination guide
```

for every planet in a multi-planet system would quickly become unreadable.

Selection-scoped guides also make provenance easier to explain.

Keep system-wide orbit paths separate from selected-orbit orientation guides.

---

# 10. Batched guide draw using the orbit program — acceptable

Reusing the existing orbit GLSL program is reasonable as long as:

- the program only consumes finished geometry/style;
- guides and orbital paths use separate scene primitives;
- guide dashing is driven only by guide style;
- no orbital-science semantics leak into the shader;
- batching does not join independent guide segments.

There is no architectural reason to create a seventh shader merely because the line means something different scientifically.

Semantic meaning belongs upstream.

---

# 11. Synthetic measured/derived demo cases — APPROVED

The fact that no planet in the current snapshot has a measured Ω is important.

Do **not** invent a real system with full orientation just to make the demo look complete.

Constructed fixtures are the correct way to exercise:

```text
MEASURED
DERIVED
ASSUMED
```

render paths when the validated snapshot cannot provide all three naturally.

The conditions are:

```text
[PASS] every synthetic frame is explicitly labelled constructed/synthetic
[PASS] no synthetic fixture is presented as an observed exoplanet solution
[PASS] real-system scientific tests remain separate
[PASS] synthetic fixtures test renderer semantics, not catalog claims
```

Based on your report, that is exactly what you are doing.

Keep it.

---

# 12. One future improvement: separate "geometry confidence" from "parameter provenance"

The current line-style policy already implicitly does this well.

Consider formalizing it later.

For example:

```python
GuideGeometryStatus:
    CONSTRAINED
    NORMALIZED
    UNKNOWN
```

while the panel separately retains:

```text
MEASURED
DERIVED
ASSUMED_FOR_VISUALIZATION
UNKNOWN
```

Why?

A guide can be built from a mixture such as:

```text
i = MEASURED
ω = DERIVED
Ω = ASSUMED_FOR_VISUALIZATION
```

The individual parameter provenance is richer than one line style.

The displayed geometry as a whole is:

```text
NORMALIZED / assumption-dependent
```

This is **not required before the C2 push**. Your current implementation is already semantically defensible.

Just keep this distinction in mind before orientation overlays become more elaborate.

---

# 13. C2 validation — strong

The reported increase:

```text
720 -> 755 tests
```

and the fact that every prior C2 acceptance criterion has a named test is excellent.

Especially important are tests covering:

```text
i = 0°
i = 90°
Ω = 90°
ω = 90°
combined rotation
periapsis arrow vs actual periapsis
unknown Ω
normalized Ω
stellar-reflex conversion
AS_REPORTED ambiguity
scientific-state immutability
renderer angle firewall
selection behavior
HZ reference-plane invariance
```

Those collectively test the semantics, not merely whether lines appear.

---

# 14. CI changes — approved

Adding a distinct:

```text
Scientific overlay tests
```

step is useful.

Including `orientation_demo` in the Mesa render job is even more important because the principal C2 distinction is:

```text
solid vs dashed
```

which can be correct in Python scene objects but broken during GPU upload/shader rendering.

This is exactly the kind of behavior that CI pixels should exercise.

Raising the expected frame count to 11 is appropriate only because the workflow now intentionally produces at least 11 frames.

Keep the existence assertion rather than relying on artifact-upload warnings.

---

# 15. One check I want after the push

Because C2 reuses the orbit shader and introduces another batched line primitive, the remote audit should explicitly inspect:

```text
guide batching boundaries
dash attribute propagation
GL state restoration
draw-call counts
```

Nothing in your report suggests a problem.

I just want those verified against the actual pushed implementation rather than inferred from the summary.

---

# 16. C2 local verdict

```text
C1 follow-up wording                    PASS
HZ boundary edge rendering              PASS
HZ GL state restoration                 PASS
orientation geometry reuse              PASS
renderer/science firewall               PASS
MEASURED styling                        PASS
DERIVED styling                         PASS
ASSUMED styling                         PASS
UNKNOWN/default-hidden behavior         PASS
normalized orientation disclosure       PASS
selected-planet scoping                 PASS
synthetic demonstration honesty         PASS
755 local tests                         PASS
six shader programs                     PASS
11 demo frames                          PASS
remote CI                               PENDING PUSH
```

## C2 is approved for push.

---

# 17. Push instruction

The remote branch is currently still at C1.

Run:

```bash
git fetch origin
git merge-base --is-ancestor origin/3D-test HEAD
```

If that succeeds:

```bash
git push origin HEAD:3D-test
```

No force push.

Afterward send:

```text
branch: 3D-test
commit: <full SHA for 5129422>
CI run: <run id>
CI status: green / failing
```

Then perform the final remote C2 audit.

---

# 18. What comes after C2

If the remote C2 audit is green, proceed to:

## Explorer C3 — coordinate and distance inspector

Recommended first scope:

```text
Earth ↔ host distance
host ↔ selected planet instantaneous distance
periapsis
apoapsis
ICRS coordinates
Galactic coordinates
SystemFrame local coordinates
```

Scientific rule:

> distance values must come from the scientific coordinate/orbital state, never from visually exaggerated render geometry.

Do not use `display_radius`, rendered positions after arbitrary presentation transforms, or label/picking geometry to calculate scientific distances.
