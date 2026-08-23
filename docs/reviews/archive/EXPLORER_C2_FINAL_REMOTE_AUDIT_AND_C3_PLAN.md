# Explorer C2 Final Remote Audit and Explorer C3 Plan

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Commit:** `5129422c16d0f20d1a8cfb680ca65a7e460268ee`  
**CI run:** `32638586616`  
**Verified suite:** 755 tests  
**Verdict:** **EXPLORER C2 CLOSED — PASS**

---

## 1. Remote verification

The pushed C2 commit is the actual head tested by GitHub Actions.

Remote CI completed successfully for:

```text
Tests (Python 3.11)             PASS
Tests (Python 3.12)             PASS
OpenGL 3.3 core (software Mesa) PASS
```

The OpenGL job also ran:

```text
context + shader verification
OpenGL backend tests
multi-planet rendering tests
Explorer navigation/picking/labels
scientific overlay tests (C1 + C2)
real render demos
frame-count assertion
artifact upload
```

The `rendered-frames` artifact exists for this exact head SHA and is not expired.

---

# 2. Guide batching — PASS

The remote implementation uses explicit line-segment indices for every polyline.

Conceptually:

```text
guide A vertices
    -> (0,1), (1,2), ...

guide B vertices
    -> offset + (0,1), offset + (1,2), ...
```

No `LINE_STRIP` is used across independent guides.

Therefore a closed plane ring, a two-point node line, an inclination arc, and an arrow cannot accidentally join end-to-end.

All guides are packed into one batch and rendered with:

```text
1 LINES draw
```

for any non-empty guide set.

This is the correct batching architecture.

---

# 3. Dash propagation — PASS

The guide pass reuses the orbit line shader and carries, per vertex:

```text
position
cumulative arclength
colour
dash period
```

The backend computes:

```text
solid guide  -> dash period = 0

dashed guide -> nonzero dash period
```

so the scientific distinction between constrained and display-normalized geometry survives batching and reaches the GPU.

This was also exercised remotely by the orientation demo on software Mesa.

---

# 4. HZ edge rendering — PASS

The previous dead `edge_color` contract is now real.

The zone pass performs:

```text
all fills  -> one TRIANGLES draw
all edges  -> one LINES draw
```

The inner and outer loops are indexed independently and explicitly closed.

This preserves the meaning of the model boundaries without introducing a second scientific calculation.

---

# 5. C2 scientific/rendering firewall — PASS

`RenderGuide` contains only finished rendering information:

```text
points_local
GuideStyle
colour
label
```

It does **not** contain:

```text
inclination
omega
Omega
eccentricity
periastron convention
scientific provenance enum
```

The geometry is built upstream using the same production orbital rotation:

\[
\mathbf r = R_z(\Omega)R_x(i)R_z(\omega)\mathbf r_{pf}
\]

That is the key C2 success: there is still only one scientific interpretation of the orbital elements.

---

# 6. Provenance/styling policy — PASS

The current policy is scientifically coherent:

| element state | guide |
|---|---|
| `MEASURED` | solid |
| `DERIVED` | solid, labelled derived |
| `ASSUMED_FOR_VISUALIZATION` | dashed + textual disclosure |
| `UNKNOWN` | absent by default; optional normalized display is dashed |

The important interpretation is:

```text
solid  = geometry is scientifically constrained

dashed = the displayed geometry depends on a visualization assumption
```

This is better than making line style a literal copy of the provenance enum.

A deterministic stellar-reflex +180° conversion is therefore correctly solid while still described as `DERIVED` in text.

---

# 7. Measured inclination with unknown node — PASS

The current behavior is correct:

```text
i measured
Omega unknown
Omega_display = 0°
```

The inclination magnitude remains a real measurement.

But the absolute azimuth of the drawn plane is not observed, so the full displayed plane is dashed.

Keep the conceptual distinction explicit:

```text
inclination magnitude: MEASURED
absolute azimuth: UNKNOWN
visual placement: ASSUMED_FOR_VISUALIZATION
```

`show_normalised=False` should remain the default.

---

# 8. Synthetic orientation fixtures — PASS

No planet in the validated snapshot currently has a measured longitude of ascending node.

Using constructed fixtures for the measured/derived/assumed renderer cases is therefore the honest solution.

Keep these rules:

```text
synthetic fixtures are clearly labelled
synthetic fixtures test renderer semantics
synthetic fixtures are never presented as observed exoplanet solutions
real catalog systems remain separate regression cases
```

---

# 9. One small semantic follow-up: ascending-node annotation

The remote `orientation_guides()` implementation currently says, whenever the raw node is known:

```python
"Ascending node: ... (measured)."
```

That is correct for the current synthetic measured-node case and likely for any directly catalogued node.

However, the architecture now claims generic support for:

```text
MEASURED
DERIVED
ASSUMED_FOR_VISUALIZATION
UNKNOWN
```

So this line should eventually use the node parameter's actual provenance rather than hard-code `measured`.

Prefer:

```python
"Ascending node: {value} ({provenance_word})."
```

This is **non-blocking** for C2 because the current catalog has no derived-node path, but it is worth correcting before provenance-rich C3/C4 UI becomes broader.

Suggested regression:

```text
test_a_known_node_annotation_uses_its_actual_provenance
```

---

# 10. One render-state contract refinement

The zone and guide passes correctly preserve the previous line width.

They then return depth writes and blending to the renderer's expected default state:

```text
depth_mask = True
BLEND disabled
```

This is correct **if the renderer formally guarantees those pass-entry invariants**.

The implementation is not a fully general snapshot/restore of arbitrary previous blend/depth state.

Choose one of two designs later:

### A. Formal pass preconditions

Document and test:

```text
all overlay passes enter with:
    depth writes enabled
    blending disabled
```

and restoring those defaults is correct.

### B. True state snapshot

Track and restore the actual previous:

```text
blend enabled/disabled
depth-mask state
line width
```

Either is valid.

Do not reopen C2 for this; the current renderer pipeline and remote CI are internally consistent.

---

# 11. GuideGeometryStatus — still optional

The earlier idea of formalizing a guide-level state such as:

```text
CONSTRAINED
NORMALIZED
UNKNOWN
```

remains useful but is still not necessary.

The current system already separates:

```text
per-parameter provenance in the scientific/UI layer
```

from:

```text
solid/dashed geometry semantics in the renderer
```

Only introduce a separate guide-geometry status if later overlays become complicated enough that this distinction is repeatedly reconstructed.

---

# 12. Review-file archive

The locally moved:

```text
EXPLORER_C2_LOCAL_REVIEW_AND_PUSH_APPROVAL.md
```

should remain under:

```text
docs/reviews/archive/
```

and travel with the first C3 commit.

Do not create a separate push only for that review file.

---

# 13. C2 final status

```text
remote fast-forward                      PASS
Python 3.11                              PASS
Python 3.12                              PASS
software Mesa GL 3.3                     PASS
scientific overlay CI step               PASS
orientation render demo                  PASS
render artifact                          PASS
guide segment isolation                  PASS
dash propagation                         PASS
HZ fill + edge batching                  PASS
draw-call counts                         PASS
orientation transform reuse              PASS
renderer/raw-angle firewall              PASS
unknown-node disclosure                  PASS
normalized-node default hidden           PASS
measured/derived/assumed styling          PASS
synthetic-case honesty                    PASS
node annotation provenance wording       FOLLOW-UP, NON-BLOCKING
GL pass-state contract wording            FOLLOW-UP, NON-BLOCKING
```

## Explorer C2 is CLOSED.

---

# 14. Explorer C3 — coordinate and distance inspector

C3 should now make the explorer numerically useful outside the orbit drawing itself.

The central rule is:

> **All scientific distances and coordinates come from float64 scientific state, never from display geometry.**

Never compute scientific values from:

```text
RenderStar.position_local after presentation transforms
RenderPlanet display geometry
display_radius
picking spheres
LOD geometry
label positions
camera-relative presentation transforms
```

---

# 15. C3 first scope

Implement a read-only coordinate/distance model for the selected star/planet.

Recommended fields:

## Host star

```text
ICRS RA
ICRS Dec
catalog distance
Cartesian ICRS position
Galactic longitude l
Galactic latitude b
Galactic Cartesian position, if useful
Earth/Sun -> host separation
light-travel time
```

## Selected planet

```text
host -> planet instantaneous distance
periapsis distance
apoapsis distance
SystemFrame x/y/z
current orbital phase provenance
```

Optional later:

```text
planet -> arbitrary selected star distance
```

---

# 16. Use Astropy for astronomical frame transforms

Do not create another hand-written RA/Dec frame engine.

Use:

```python
astropy.coordinates.SkyCoord
```

with explicit frames:

```text
ICRS
Galactic
Galactocentric when/if needed
```

The scientific layer should own these transformations.

The renderer should never know RA, Dec, parallax, or Galactic longitude.

---

# 17. Earth/Sun-to-host distance semantics

For the current explorer, the catalog star distance is effectively the host's barycentric/heliocentric catalog distance at astronomical scale.

The inspector should display provenance:

```text
value
unit
source
uncertainty/status
```

If the absolute stellar position is unknown:

```text
Earth/Sun -> host distance = UNKNOWN
```

A detached local system must **never** claim an Earth distance.

This directly extends the detached-frame rule established in Explorer B.

---

# 18. Host-to-planet instantaneous distance

Compute this from the physical orbital state.

Equivalent forms include:

\[
r = a(1-e\cos E)
\]

or:

\[
r = \frac{a(1-e^2)}{1+e\cos\nu}
\]

The preferred implementation should reuse the current propagated position/state:

```python
r = np.linalg.norm(planet_position_au)
```

where `planet_position_au` is the float64 physics result **before** conversion to render coordinates.

Do not recompute from a render position merely because the numbers happen to match in `SystemFrame` today.

---

# 19. Periapsis / apoapsis

Expose:

\[
r_{peri}=a(1-e)
\]

\[
r_{apo}=a(1+e)
\]

using the existing scientific `Parameter` objects if they already exist.

If `a` or `e` is assumed for visualization, the inspector must show that provenance.

A visualization assumption must not silently become a measured periapsis/apoapsis value.

---

# 20. Planet absolute position

If the host's absolute position is known, the planet's instantaneous absolute position is conceptually:

\[
\mathbf r_{planet,abs}
=
\mathbf r_{host,abs}
+
\mathbf r_{planet,local}
\]

But keep the scales explicit:

```text
host absolute position     pc, float64
planet local position      AU, float64
```

Convert the local offset explicitly before addition.

Do not route this through float32 GPU coordinates.

If the host is detached/unlocated:

```text
absolute planet position = UNKNOWN
local SystemFrame position = known/available
```

---

# 21. Distance to another selected star

When this feature is added:

\[
D = \left|\mathbf r_2-\mathbf r_1\right|
\]

Use one common float64 astronomical frame.

For planet-to-other-star:

\[
D =
\left|
\mathbf r_{other\ star}
-
(\mathbf r_{host}+\mathbf r_{planet/local})
\right|
\]

At parsec scales the AU offset is tiny, but retaining it is scientifically cleaner and costs little.

---

# 22. C3 provenance model

Every inspector row should support:

```text
value
unit
uncertainty
status
source/reference
frame
reference epoch where relevant
```

Coordinate rows should also identify the frame explicitly:

```text
ICRS
Galactic
SystemFrame
```

Do not display a naked `x/y/z` triplet without telling the user which frame and unit it belongs to.

---

# 23. C3 acceptance tests

At minimum:

```text
[ ] known RA/Dec/distance -> SkyCoord round-trip
[ ] ICRS -> Galactic -> ICRS round-trip within tolerance
[ ] Earth/host distance agrees with catalog scientific position
[ ] detached system has no absolute distance
[ ] detached system still reports local SystemFrame coordinates
[ ] host/planet instantaneous distance equals norm of physics state
[ ] host/planet distance agrees with a(1-e cos E)
[ ] periapsis equals a(1-e)
[ ] apoapsis equals a(1+e)
[ ] high-e HD 80606 b distance behaves correctly near peri/apo
[ ] no C3 scientific function imports rendering primitives
[ ] changing display exaggeration leaves all C3 distances unchanged
[ ] changing LOD leaves all C3 values unchanged
[ ] changing camera position leaves object-object distances unchanged
[ ] unknown absolute host position never becomes zero pc
[ ] absolute planet position uses float64 frame conversion
[ ] every x/y/z row includes frame + unit metadata
[ ] uncertainty/provenance survive coordinate presentation
[ ] full C1/C2 suite remains green
```

---

# 24. Recommended C3 worked systems

Use deliberately different cases:

```text
HD 80606 b
    high eccentricity
    strong instantaneous-distance variation

Kepler-11
    multi-planet system
    useful for local SystemFrame values

TRAPPIST-1
    detached/unlocated regression in the current snapshot
    local coordinates work, absolute distance remains unavailable

HD 219134
    mixed-data regression
```

These are already familiar to the test suite and exercise different failure modes.

---

# 25. Do not add C4 yet

Keep C3 as another clean vertical slice.

Do not bundle:

```text
HR diagram
blackbody spectrum
atmospheric spectroscopy UI
```

into the C3 commit.

First make coordinate and distance semantics independently green and remotely auditable.

---

# Immediate instruction

**C2 is closed. Start Explorer C3 as a separate coordinate/distance vertical slice.**

At the beginning of the C3 working tree, include the two non-blocking C2 cleanups:

```text
use actual provenance word in the known-node annotation
formalize/document overlay pass state preconditions, or fully snapshot blend/depth state
```

Then build the read-only coordinate/distance model entirely from scientific float64 state.
