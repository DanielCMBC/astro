# Explorer C1 Review — Habitable-Zone Overlay and Push Recommendation

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Local commit reported:** `4ab8e54`  
**Remote tip at review time:** `183c68e9a56d14b8cee7c621dcd199bc426ab483`  
**Reported test count:** 720  
**Verdict:** **C1 PASS — PUSH C1 NOW AS ITS OWN VERTICAL SLICE**

---

## 1. Push strategy

Do **not** wait for C2–C4.

Explorer C1 is a complete, independently testable scientific vertical slice:

```text
stellar model
    ↓
habitable-zone model
    ↓
scene-builder translation
    ↓
finished render geometry
    ↓
OpenGL draw
    ↓
visual disclosure
```

That is exactly the sort of boundary that deserves its own remote CI checkpoint.

Pushing C1 separately gives:

- a clean bisect point;
- a smaller review surface;
- independent CI evidence;
- easier rollback;
- clearer history;
- less chance that C2/C3/C4 regressions obscure whether the HZ overlay itself was sound.

C2–C4 touch meaningfully different concerns and should not be bundled merely because they share the Explorer C label.

After re-fetching and confirming ancestry:

```bash
git fetch origin
git merge-base --is-ancestor origin/3D-test HEAD
git push origin HEAD:3D-test
```

No force push.

---

## 2. Scientific/data architecture — PASS

The strongest design choice is that the renderer never sees the scientific inputs required to compute the habitable zone.

`RenderZone` carries finished geometry and visual attributes, not luminosity, stellar effective temperature, Kopparapu coefficients, model names, provenance, or raw AU boundary inputs.

Keep the rule:

```text
science determines geometry
renderer displays geometry
```

The GPU should never be able to change a physical HZ boundary.

---

## 3. One-source-of-truth HZ model — PASS

The overlay and information panel consume the same:

```text
StarRecord.habitable_zone
```

object.

The regression:

```text
test_the_overlay_and_the_panel_agree
```

is especially valuable because it prevents a second HZ implementation from appearing in rendering code.

Keep the Kopparapu evaluation in the stellar-physics layer only.

---

## 4. Renderer-purity test — good solution

A raw-text grep is too naive because comments and documentation may legitimately mention scientific terms.

The token-aware/code-only scan is a better architecture test.

The durable rule is:

> Rendering modules may consume a finished `RenderZone`; they may not calculate HZ boundaries.

If the architecture tests later become AST-based, this rule can become even more structural.

---

## 5. Unknown/out-of-domain inputs — PASS

Returning no zone when the chosen model is outside its adopted validity range is exactly right.

TRAPPIST-1 is therefore a useful counterexample.

The correct behavior is:

```text
required model input outside validity domain
        ↓
HZ boundaries = UNKNOWN
        ↓
no zone geometry
        ↓
annotation explains why
```

Do not clamp to the model limit, substitute a Solar value, or draw a generic zone.

---

## 6. Provenance survives upstream — PASS

It is correct that provenance remains in the scientific object while `RenderZone` itself carries only finished visual geometry.

The architecture should remain:

```text
scientific HZ edge
    value
    unit
    DERIVED status
    model provenance
        ↓
scene builder
        ↓
finished render geometry
```

The panel/UI explains provenance; the renderer only draws.

---

## 7. Disclaimer — PASS

Keep the explicit statement that the overlay represents a **stellar-irradiation habitable-zone model**, not evidence that a planet is habitable.

Actual habitability can depend on atmosphere, albedo, pressure, volatile inventory, stellar activity, tidal state, atmospheric loss, geology, and evolutionary history.

---

## 8. Flat annulus vs spherical shell — acceptable

The physical radial HZ is a 3D shell around the star.

The current visualization is a **reference-plane cross-section** of that shell.

That is acceptable and may be clearer than a large translucent sphere.

Recommended UI wording:

```text
Habitable-zone cross-section
Radial irradiation boundaries shown in the system reference plane.
The physical region is a spherical shell around the star.
```

A future optional shell/wireframe mode can be considered later, but it is not required now.

---

## 9. Kepler-11 wording refinement

Avoid saying:

> Kepler-11 g is “inside the inner edge”

because users may interpret that as “inside the habitable zone.”

Prefer:

```text
Kepler-11 g lies interior to the HZ inner boundary.
```

or:

```text
Kepler-11 g lies starward of the 1.007 AU inner boundary,
so it is outside this irradiation-defined habitable zone on the hot side.
```

That wording is much harder to misunderstand.

---

## 10. HZ boundary naming

If the current model has one fixed documented inner/outer prescription, generic `inner` and `outer` fields are fine for C1.

Before supporting multiple HZ prescriptions, make the boundary semantics explicit, for example:

```text
runaway_greenhouse
maximum_greenhouse
recent_venus
early_mars
```

Do not silently change HZ definitions while keeping identical generic labels.

This is not blocking C1.

---

## 11. Geometry/rendering — PASS

CPU triangulation of the annulus is appropriate:

- the GPU receives finished geometry;
- one indexed triangle draw can batch the zone;
- physical boundary calculations never enter shader code;
- tessellation/sample count remains a visual parameter.

Disabling depth writes while keeping the overlay readable is reasonable.

One useful future regression, if not already covered by the renderer abstraction:

```text
test_zone_draw_restores_gl_state
```

so blend/depth state cannot leak into later passes.

---

## 12. Visual-parameter inertness — PASS

Tests proving that sample count, colour, and opacity do not change HZ science are exactly the right golden-rule tests.

Extend that family later to thickness, render resolution, camera transforms, and LOD as those features appear.

---

## 13. Documentation archive — keep it

Keep generated historical reviews under:

```text
docs/reviews/archive/
```

for now.

The archive README already establishes the correct authority rule:

```text
archived review = historical context
docs/* = current specification
```

There is no need to delete them merely for tidiness.

---

## 14. C1 acceptance result

Based on the reported implementation:

```text
same HZ model as panel                  PASS
renderer does not recalculate HZ        PASS
unknown/out-of-domain gives no zone     PASS
provenance preserved upstream           PASS
habitability disclaimer                 PASS
visual controls scientifically inert    PASS
CPU finished geometry                   PASS
real GL renderer path                    PASS
six shader programs compile              PASS
720 tests                                PASS locally
three demos render                       PASS locally
```

C1 is complete enough to push.

---

## 15. What I want after the push

Send:

```text
branch: 3D-test
commit: <full SHA>
CI: green / failing
```

Then perform the remote C1 audit.

---

## 16. C2 should be next, separately

After C1 is remotely green, proceed to **C2 — orbital-orientation overlays**.

Recommended C2 scope:

```text
orbital plane/reference plane
ascending-node line
periapsis direction
inclination indicator
orientation legend/provenance
```

Golden rules:

```text
MEASURED orientation
    may be drawn as scientific information

DERIVED stellar-reflex conversion
    must remain marked derived

ASSUMED_FOR_VISUALIZATION
    may be drawn only with explicit assumed styling/disclosure

UNKNOWN
    must never become a scientific-looking line without disclosure
```

The overlay should consume the existing resolved orientation state and must not reinterpret catalog conventions itself.

---

# Final decision

**Push C1 now. Do not bundle C2–C4 into the same remote milestone.**

Small, scientifically coherent vertical slices are exactly what is making this project easy to audit and hard to corrupt.
