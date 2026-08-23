# Explorer C1 Final Remote Audit and C2 Instructions

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Commit:** `4ab8e54adfe5ae0d1df5eb03eab5d86d087c2ff5`  
**CI run:** `32635904507`  
**Reported / verified suite:** 720 tests  
**Verdict:** **EXPLORER C1 CLOSED — PASS**

---

## 1. Remote state

The remote `3D-test` branch now points to:

```text
4ab8e54adfe5ae0d1df5eb03eab5d86d087c2ff5
```

and the commit is a direct child of the Explorer B close-out commit:

```text
183c68e9a56d14b8cee7c621dcd199bc426ab483
        ↓
4ab8e54adfe5ae0d1df5eb03eab5d86d087c2ff5
```

This was therefore a clean one-commit fast-forward milestone.

---

## 2. Remote CI

GitHub Actions run `32635904507` completed successfully.

### Python 3.11

```text
architecture / golden-rule tests    PASS
full suite                          PASS
offline check                       PASS
```

### Python 3.12

```text
architecture / golden-rule tests    PASS
full suite                          PASS
offline check                       PASS
```

### OpenGL 3.3 software Mesa

```text
GL 3.3 context                      PASS
all declared shaders                PASS
OpenGL backend tests                PASS
multi-planet rendering              PASS
explorer navigation/picking/labels  PASS
real demo renders                   PASS
frame-existence assertion           PASS
artifact upload                     PASS
```

A rendered-frames artifact is attached to this exact commit.

---

## 3. Scientific architecture — PASS

The C1 scientific/rendering boundary is correct.

The flow is:

```text
stellar physics
    ↓
StarRecord.habitable_zone
    ↓
scene builder
    ↓
RenderZone with finished geometry
    ↓
OpenGL
```

`RenderZone` does not contain:

```text
luminosity
effective temperature
Kopparapu coefficients
scientific HZ model inputs
AU boundary calculation logic
scientific provenance
```

Therefore the renderer cannot recompute a boundary.

Keep this invariant permanently.

---

## 4. Single source of truth — PASS

The information panel and the 3D overlay both read:

```text
StarRecord.habitable_zone
```

The Kopparapu polynomial remains in the stellar-physics module.

The regression:

```text
test_the_overlay_and_the_panel_agree
```

correctly prevents a future second HZ implementation from drifting away from the panel/science result.

---

## 5. Model-domain handling — PASS

The code refuses to extrapolate the current Kopparapu fit outside its adopted:

```text
2600 K <= Teff <= 7200 K
```

range.

Unknown / invalid cases produce:

```text
HabitableZone(UNKNOWN, UNKNOWN)
```

which becomes:

```text
no RenderZone
```

instead of a default or clamped band.

TRAPPIST-1 is a good regression case because the current stellar temperature lies below this fitted range.

This behavior should stay.

---

## 6. Provenance boundary — PASS

The HZ boundaries remain scientific `Parameter` objects upstream with:

```text
Status.DERIVED
units
model provenance
```

while the renderer receives only the final geometry.

This is the same science/display firewall used successfully elsewhere in the project.

---

## 7. Disclaimer — PASS

Every rendered HZ scene includes a disclaimer that the region is an irradiation-based model and not evidence that a specific world is habitable.

Keep this explicit.

The HZ model does not establish:

```text
atmosphere
surface liquid water
pressure
albedo
stellar activity tolerance
volatile retention
geological state
biosphere
```

so the overlay must never visually imply those conclusions.

---

## 8. Flat annulus — PASS, with wording refinement approved

The physical HZ is radial: geometrically it is a spherical shell around the host star.

The current flat annulus is best understood as a **reference-plane cross-section** through that shell.

Apply the wording refinement you held back.

Preferred annotation:

```text
Habitable-zone cross-section:
radial irradiation boundaries shown in the system reference plane.
The physical region is a spherical shell around the star.
```

Do not rotate the HZ annulus into a selected planet's orbital plane in C2.

It represents stellar radial distance, not an orbital-plane property.

---

## 9. Kepler-11 wording refinement — APPROVED

Apply this small wording fix too.

Avoid:

```text
Kepler-11 g is inside the inner edge
```

because “inside” can be misread as “inside the habitable zone.”

Prefer:

```text
Kepler-11 g lies starward of the 1.007 AU inner HZ boundary,
so it is outside this irradiation-defined habitable zone on the hot side.
```

or:

```text
Kepler-11 g lies interior to the HZ inner boundary.
```

This is a presentation correction, not a physics change.

---

## 10. One implementation mismatch found in the remote audit

`RenderZone` contains:

```python
edge_color
```

and documents it as:

```text
Edge colour for the two boundary loops.
```

However, the current GL path never consumes `edge_color`.

`_batch_zones()` packs only:

```text
position
zone.color
```

and `_draw_zones()` performs only the filled `TRIANGLES` draw.

So the field currently promises visible boundary-ring styling that the renderer does not implement.

This is **not a C1 scientific failure**: the actual HZ radii and filled geometry are correct.

But the render contract and backend should agree.

### Recommended fix

I prefer implementing the boundaries rather than deleting `edge_color`.

Batch all HZ boundary rings into one additional line draw:

```text
all zone fills   -> 1 TRIANGLES draw
all zone edges   -> 1 LINES draw
```

This gives the scientific inner/outer limits a crisp visual boundary while preserving batching.

Alternative:

```text
remove edge_color entirely
```

if fill-only is intentionally the final design.

Do not leave a dead field whose documentation says it is rendered.

### Add regression

```text
test_zone_edge_style_is_consumed_by_renderer
```

and keep GL-state restoration covered.

---

## 11. GL render-state behavior — PASS

The current `_draw_zones()`:

```text
enables blending
sets depth_mask = False
draws
restores depth_mask = True
disables blending
```

so the zone pass does not currently leak those states into the orbit pass.

Keep this pattern when adding boundary rings.

If boundary edges are added, either:

```text
draw fill + edges inside one scoped zone pass
```

or explicitly restore state after both.

---

## 12. HZ model versioning — not blocking, but plan it

The current `HabitableZone` model explicitly names:

```text
Kopparapu et al. 2013
runaway greenhouse / maximum greenhouse
```

and the coefficient validity range is explicit.

That is good.

Before supporting additional HZ prescriptions, do not keep only generic:

```text
inner
outer
```

with silently changing meaning.

Introduce something like:

```text
HZModel.KOPPARAPU_2013_CONSERVATIVE
```

and explicit boundary kinds:

```text
RUNAWAY_GREENHOUSE
MAXIMUM_GREENHOUSE
RECENT_VENUS
EARLY_MARS
```

when/if multiple definitions are added.

C1 does not need this expansion.

---

## 13. Review-file policy

The untracked:

```text
EXPLORER_C1_REVIEW_AND_PUSH_RECOMMENDATION.md
```

should follow the repository convention already established.

### Recommendation

Move it to:

```text
docs/reviews/archive/
```

if you want the audit trail.

Do **not** leave it at repository root.

If you do not care about preserving the intermediate review, delete it instead.

Given that earlier reviews are already archived, I would archive this one for consistency.

Do not make a separate remote commit just for this file.

Include the archive move with the next C2 commit.

---

## 14. Do not make a C1.1 push just for the small refinements

C1 is remotely green and scientifically complete.

Apply these at the start of the C2 working tree:

```text
[ ] reference-plane cross-section wording
[ ] Kepler-11 “starward of inner boundary” wording
[ ] archive the untracked C1 review file
[ ] resolve the dead edge_color contract
```

Then let them travel with the next meaningful C2 commit.

There is no value in churning the shared branch solely for wording.

---

## 15. Explorer C1 final status

```text
remote commit                           PASS
fast-forward history                    PASS
Python 3.11                             PASS
Python 3.12                             PASS
offline mode                            PASS
OpenGL 3.3 software Mesa                PASS
render artifact                         PASS
one HZ science implementation           PASS
panel / overlay agreement               PASS
unknown-data behavior                   PASS
model-domain refusal                    PASS
provenance upstream                     PASS
habitability disclaimer                 PASS
visual parameter inertness              PASS
renderer/science firewall               PASS
edge_color/backend consistency          FOLLOW-UP, NON-BLOCKING
```

## C1 is CLOSED.

---

## 16. Explorer C2 — orbital-orientation overlay

Proceed to C2 next.

The central scientific issue is not drawing lines; it is making sure **unknown orientation never looks observed**.

Recommended overlays:

```text
system reference plane
selected orbit plane
orbit normal vector
ascending-node line
periapsis direction
inclination indicator
```

Prefer showing detailed orientation guides for the **selected planet** rather than drawing every plane at once.

---

## 17. C2 coordinate convention must be explicit

Document what the system reference axes mean.

The overlay must use the exact same convention and rotation order as the physical position propagator:

\[
\mathbf r =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf r_{pf}
\]

Do not create a second interpretation of these angles in visualization code.

---

## 18. Unknown longitude of ascending node is the key C2 case

For many exoplanets:

```text
Ω = UNKNOWN
```

The display normalization may use:

```text
Ω_display = 0°
```

but the overlay must say that this is not a measured sky orientation.

Recommended presentation:

```text
Ascending node: unknown
Display normalization: Ω = 0°
Absolute rotation about the line of sight is unconstrained.
```

An assumed node should use clearly different styling from a measured one.

For example:

```text
measured      solid
derived       solid + derived legend
assumed       dashed
unknown       absent unless normalized display is explicitly enabled
```

Do not encode scientific meaning by colour alone.

---

## 19. Periastron-direction provenance

The periapsis arrow must respect the already-correct `PeriastronConvention` model:

```text
PLANET
STELLAR_REFLEX -> converted +180°, DERIVED
AS_REPORTED -> ambiguous / assumed for visualization
UNKNOWN
```

The C2 renderer should receive only the finished arrow endpoints/style.

It must not know about stellar-reflex conversion.

---

## 20. Recommended render contract for C2

Prefer a generic finished-guide primitive rather than giving the GL backend orbital elements.

Conceptually:

```python
RenderGuideLine(
    identifier=...,
    points_local=...,
    style=...,
    label=...,
)
```

or an equivalent finished-geometry structure.

The critical rule is that it must **not** contain raw:

```text
inclination
omega
Omega
eccentricity
```

for the renderer to interpret.

The scene builder / scientific geometry layer should produce the finished vectors.

---

## 21. C2 acceptance tests

At minimum:

```text
[ ] i = 0° orbit plane matches reference plane
[ ] i = 90° orbit normal rotates correctly
[ ] Ω = 90° rotates line of nodes correctly
[ ] ω = 90° rotates periapsis direction correctly
[ ] combined Ω/i/ω matches the production orbital transform
[ ] periapsis arrow agrees with actual periapsis position
[ ] measured Ω draws as measured
[ ] unknown Ω is not presented as observed
[ ] Ω=0 display normalization is marked ASSUMED_FOR_VISUALIZATION
[ ] stellar-reflex ω conversion remains DERIVED
[ ] AS_REPORTED convention cannot produce ORIENTATION_FULL
[ ] overlay geometry does not modify orbital state
[ ] renderer receives no raw orbital angles
[ ] HZ annulus remains in system reference plane
[ ] selection changes which planet's guides are shown without changing its orbit
[ ] all existing C1 tests remain green
[ ] software-Mesa CI renders measured/derived/assumed orientation examples
```

---

## Immediate instruction

**C1 is closed.**

Start C2 after making the four small working-tree follow-ups:

```text
cross-section wording
Kepler-11 wording
archive C1 review
resolve edge_color contract
```

Do not push those separately; let them accompany the first coherent C2 commit.
