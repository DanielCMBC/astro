# Explorer A Review — Frame Transitions, Picking, Labels, and Explorer B Instructions

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Reviewed commit:** `96c6d1e074f018f9b23fc2f720cbf846d942180a`  
**Reported test count:** 604  
**Verdict:** Explorer A is a strong pass, with one semantic fix required before Explorer B.

---

## 1. What passed

The current Explorer A implementation successfully introduces:

- camera-position-derived frame selection;
- absolute camera position in float64 parsecs;
- explicit frame conversion before float32 rendering;
- hard failure on unsafe narrowing;
- logarithmic system-approach waypoints;
- analytic ray/sphere object picking;
- behind-camera rejection;
- nearest-hit selection;
- selection persistence across LOD and scene rebuilds;
- screen-space labels that never mutate scientific coordinates;
- priority-pinned selected labels;
- projected-size LOD;
- legacy-prototype quarantine;
- CI rendering of the pc→AU approach.

The overall architecture is sound.

---

## 2. Frame-selection design

The principle is correct:

> Coordinate-frame choice should not be a second mutable mode that can drift out of sync with camera position.

The useful relationship is:

```text
absolute camera position (float64 pc)
          ↓
active coordinate frame
          ↓
camera expressed locally
          ↓
float32 GPU boundary
```

Keep this.

---

## 3. Wording correction for the 0.485 pc engage radius

Current policy:

```text
FLOAT32_SAFE_MAGNITUDE = 1e6
FRAME_ENGAGE_MARGIN    = 10

engage_radius
    = 1e6 / 10 AU
    = 1e5 AU
    = 0.485 pc
```

The resulting radius is derived from the project's **chosen precision budget and safety margin**.

It is therefore more precise to say:

> `0.485 pc` is derived from the current float32 precision policy.

rather than:

> nobody picked it; it fell uniquely out of the float32 mantissa.

The constants `1e6` and `10` are engineering policy choices.

---

## 4. Precision at the transition boundary

Around:

\[
10^5\ {\rm AU}
\]

float32 spacing is on the order of:

\[
7.8\times10^{-3}\ {\rm AU}
\]

or roughly:

\[
1.2\times10^6\ {\rm km}
\]

That may be visually harmless at that distance because inner-system geometry is far below a pixel, but it should not be described as high scientific spatial precision.

A stronger future rendering regression would test **screen-space error**:

```text
project(float64 reference)
project(float32 rendered coordinate)

assert difference < 0.25 pixel
```

for visible bodies.

This measures what actually matters to the renderer.

---

# 5. Required fix before Explorer B: unknown system location must not become the Sun

The current `Explorer.focus()` path effectively does:

```python
position = None

if star has a usable position:
    position = star.position.cartesian_pc()

if position is None:
    target = self.target(host_name)
    position = target.position_pc if target else np.zeros(3)

self.system = SystemFrame.for_host(host_name, position)
```

The fallback:

```python
np.zeros(3)
```

creates a scientific semantic problem.

It converts:

```text
absolute host position = UNKNOWN
```

into:

```text
absolute host position = Solar/UniverseFrame origin
```

This violates the project's central rule that unknown values must not become convenient numeric placeholders.

TRAPPIST-1 being absent from the neighbourhood view is correct. However, an unknown-position system must never become accidentally navigable as though it were located at `(0,0,0) pc`.

### Required behavior

Universe→System navigation must require a known absolute host position.

Example:

```python
class UnknownSystemPositionError(ValueError):
    pass
```

and:

```python
if absolute_position is None:
    raise UnknownSystemPositionError(
        f"{host_name} has no usable absolute position"
    )
```

A system without a known galactic position may still be viewed locally.

But distinguish:

```text
FLY TO SYSTEM
```

from:

```text
OPEN DETACHED LOCAL SYSTEM
```

A local system may define the host star as `(0,0,0)` **in its own frame**.

It must not claim `(0,0,0) pc` as an absolute location.

### Required tests

```text
test_unknown_absolute_position_cannot_be_navigated_to
test_detached_system_can_open_locally
test_detached_system_never_appears_at_solar_origin
test_detached_system_has_no_universe_distance
```

This is the one change I consider blocking before Explorer B.

---

# 6. Separate coordinate frame from visual presentation state

Right now:

```text
active_frame = SystemFrame
```

also means:

```text
ViewState.SYSTEM
```

and `scene()` immediately swaps to the system scene.

Numerical frame selection and visual presentation are different concerns.

At:

\[
0.485\ {\rm pc}\approx100,000\ {\rm AU}
\]

the `SystemFrame` may be numerically appropriate while the best visual presentation still resembles the stellar-neighbourhood view.

For the final Eyes-style flight experience, separate:

```text
active_coordinate_frame
```

from:

```text
presentation_state / scene_mode
```

Possible presentation flow:

```text
UNIVERSE
   ↓
HYBRID / CROSSFADE
   ↓
SYSTEM
```

The coordinate frame can still be derived purely from precision requirements.

The presentation state must never control scientific coordinate interpretation.

This does not need to block the first Explorer B panel, but should be done before the final timed/free-flight camera transition.

---

# 7. Selection identity

Using a catalogue name is much better than using:

```text
array index
render handle
LOD instance
```

because names survive render rebuilds.

However, before the synchronized offline catalog becomes authoritative, introduce a stable entity key.

Recommended concept:

```python
Selection(
    entity_id="planet:nasa:HD_80606_b",
    display_name="HD 80606 b",
    kind="planet",
    host_id="star:nasa:HD_80606",
)
```

Why?

Display names and aliases may change across catalog releases.

Long-term identity should be separate from human-readable naming.

Potential stable keys:

```text
internal deterministic ID
NASA canonical planet key
Gaia source_id for stars
catalog aliases stored separately
```

This fits naturally into Explorer B.

---

# 8. Picking review

The current analytic picker is a large improvement over the legacy implementation.

It:

- uses actual ray/sphere intersections;
- rejects bodies behind the camera;
- chooses the nearest hit;
- inflates tiny click targets in screen-space terms;
- avoids GPU readback.

This is appropriate for the current scene size.

A minor future refinement:

`_effective_radius()` currently uses full `distance_to_center` for world-per-pixel conversion.

Perspective scaling is more directly tied to camera-space forward depth.

For wide fields of view, consider using:

```text
along_view
```

instead.

This is not blocking.

---

# 9. LOD review

Projected-size LOD is correct.

Conceptually:

\[
{\rm projected\ size}
\propto
\frac{R}{d}
\frac{H}{\tan(FOV_y/2)}
\]

This correctly responds to:

- distance;
- body size;
- viewport height;
- field of view.

Keep the invariant:

```text
LOD may change:
    mesh complexity

LOD must never change:
    scientific coordinates
    physical radius
    identity
    provenance
```

---

# 10. Legacy quarantine

The legacy-isolation tests are valuable.

The historical script now documents and guards against recurrence of:

```text
fake 1e9 pc distance
hard-coded constants
first-order Kepler
fake a/e/P values
bad AU→pc scaling
coplanar orbits
seconds-vs-days animation
semimajor-axis-based gas-giant classification
cache-as-synchronisation
stale async UI writes
infinite-line picking
dataframe filtering in render loop
fixed-function OpenGL
```

Do not repair the legacy script.

Keep it quarantined.

---

# 11. Documentation test-count mismatch

`docs/explorer.md` currently reports:

```text
603 tests
```

while the Explorer A commit reports:

```text
604 tests
```

Small fix:

prefer:

```text
all tests stay green | full pytest suite
```

instead of hard-coding a count in durable documentation.

Keep exact counts in commit/release notes.

---

# 12. Explorer A verdict

## PASS WITH ONE REQUIRED FIX

Fix before Explorer B:

> Unknown absolute host position must never silently become `(0,0,0) pc`.

Everything else is ready to build on.

---

# 13. Explorer B recommended order

After fixing unknown-position navigation:

1. Introduce stable entity IDs.
2. Resolve entity ID → scientific record.
3. Build a read-only system information panel model.
4. Display parameter rows with:
   - value;
   - unit;
   - uncertainty;
   - status;
   - reference/source.
5. Add planet selection.
6. Connect interactive time controls to the existing physical propagator.
7. Display phase status:
   - `CONSTRAINED`;
   - `PARTIALLY_CONSTRAINED`;
   - `ASSUMED`;
   - `UNKNOWN`.
8. Keep display assumptions visually separate from scientific measurements.

Still no gravity.

---

# 14. Explorer B acceptance criteria

```text
[ ] unknown absolute system position never becomes Sun position
[ ] detached local system cannot claim Earth/system distance
[ ] selection uses stable entity key
[ ] display-name/alias changes do not invalidate selection
[ ] UNKNOWN is never converted into a numeric UI placeholder
[ ] every displayed scientific parameter exposes provenance
[ ] planet selection does not mutate scientific orbit state
[ ] time control uses physical TimeController
[ ] constrained and assumed phases are visually distinct
[ ] selection survives LOD and scene rebuilds
[ ] async panel updates carry a selection/generation token
[ ] full test suite remains green
[ ] headless GL CI still produces verified frames
```

---

## Final recommendation

Explorer A is a successful milestone.

Keep the pure position-derived coordinate-frame idea.

Fix the unknown-position fallback immediately.

Then proceed to Explorer B, while planning a later separation between coordinate-frame selection and visual scene-transition state before the final interactive flight experience is implemented.
