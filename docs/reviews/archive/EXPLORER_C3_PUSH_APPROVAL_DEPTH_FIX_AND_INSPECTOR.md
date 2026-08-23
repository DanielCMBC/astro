# Explorer C3 Push Approval — Depth-Mask Fix + Coordinate Inspector

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote tip verified before push:** `5129422c16d0f20d1a8cfb680ca65a7e460268ee`  
**Local corrective commit:** `b8b4457`  
**Local C3 commit:** `f470cfd`  
**Reported full-suite count:** 808  
**Verdict:** **APPROVED TO PUSH BOTH COMMITS AS ONE NORMAL FAST-FORWARD**

---

## 1. Recovery and commit structure — PASS

The recovered repository reproduced the certified C2 baseline exactly:

```text
755 passed
```

and the new work is split cleanly:

```text
5129422  Explorer C2
    ↓
b8b4457  Fix framebuffer depth writes for scientific overlays
    ↓
f470cfd  Explorer C3: coordinate and distance inspector
```

This is exactly the history structure requested.

The first commit is now a genuine bisect point: checked out in isolation, it contains no C3 feature files and passes its own 763-test state.

That separation is valuable and should be preserved.

---

## 2. Framebuffer depth-mask correction — PASS

The discovered bug was significant:

```python
self.ctx.depth_mask = ...
```

created/read a harmless Python attribute instead of controlling actual depth-buffer writes.

The corrected implementation routes the state through the active framebuffer and centralizes the overlay state contract.

This means C1/C2 overlay rendering is now doing what the architecture always intended:

```text
overlay starts
    ↓
actual framebuffer depth writes disabled
    ↓
translucent zone / orbit / guide draw
    ↓
previous framebuffer depth-write state restored
```

Keep the ownership regression permanently.

---

## 3. Behavioral pixel regression — strong PASS

The new test is materially stronger than the previous state test.

It constructs a scene with actual depth separation:

```text
translucent HZ annulus in front
orbit ring behind
```

and compares correct behavior against a deliberately reintroduced depth-write bug.

Reported result:

| Mode | Images differ | Luminance ratio |
|---|---:|---:|
| correct depth writes off | yes | 1.051 |
| bug depth writes on | no — bit-identical | 1.000 |

This proves the state correction changes the intended visible outcome.

### Keep the test tolerant

For CI portability, do not require an exact `1.051` value.

Prefer a condition such as:

```text
correct image != bug/reference image
relative luminance difference > chosen safety threshold
```

with a modest margin.

Mesa/llvmpipe rasterization should be deterministic enough for this project, but the scientific regression should test the visual property rather than one exact renderer-specific number.

---

## 4. C2 node-provenance follow-up — PASS

Known-node annotations now use actual provenance rather than hard-coded `(measured)`.

The new derived-node regression closes the semantic hole while keeping the existing measured case unchanged.

C2's provenance model is therefore cleaner after the corrective commit than it was at remote closure.

---

# 5. Absolute planet ICRS correction — PASS

The previous basis-inconsistent addition is gone.

This is now the correct behavior:

```text
Host absolute ICRS position:
    available when the host is located

Planet local SystemFrame position:
    available when phase is computable

Planet absolute ICRS position:
    unresolved until SystemFrame→ICRS basis rotation exists
```

The inspector explicitly says why the absolute coordinate is unavailable rather than substituting a number.

That is scientifically preferable to publishing a coordinate that is numerically close but geometrically undefined.

The key rule is now:

> unknown or unresolved basis transformation does not become an assumed celestial coordinate.

---

## 6. `frame` / node arguments removed from the unresolved absolute-position API — PASS

Removing those parameters was correct.

Their presence would imply either:

```text
this is merely a frame-conversion problem
```

or:

```text
a measured Ω alone unlocks the transformation
```

Neither is true until the local tangent-basis convention is explicitly defined.

The test asserting those parameters are absent is a useful API-level guard.

---

# 7. `planet_to_star_distance` removal — approve, but keep the feature on the roadmap

Removing the function is scientifically correct **for now**.

A true exoplanet→other-star distance needs the planet's absolute position:

\[
D=
\left|
\mathbf r_{\rm star2,ICRS}
-
\mathbf r_{\rm planet,ICRS}
\right|
\]

and that requires the missing SystemFrame→ICRS basis transform.

Do not approximate this by simply ignoring the AU offset or adding an unrotated local vector while calling the result exact.

However, keep this feature explicitly on the roadmap because it was part of the original explorer goal:

```text
selected exoplanet -> selected star distance
```

Implement it after the local tangent-basis / absolute-planet-position milestone.

---

# 8. Published celestial-triplet guard — excellent

This regression is especially valuable:

```text
test_no_published_planet_triplet_claims_a_celestial_frame
```

Sweeping every planet in every located system and asserting that valued planet triplets are only `SystemFrame` closes the basis error at the architectural level.

It is stronger than checking one known system.

Keep it.

---

# 9. Phase provenance as a real row — PASS

Adding `NoteRow` for phase provenance is a good improvement.

A distance such as:

```text
Distance from host: 0.2057 AU
```

has materially different meaning depending on whether its phase is:

```text
CONSTRAINED
PARTIALLY_CONSTRAINED
ASSUMED
```

Putting that qualification into the inspector's row model is better than burying it in free text.

The shared `_assert_identical_rows` helper is also a good response: adding a third row type should not make the scientific-invariance regressions silently stop comparing part of the model.

---

# 10. C3 scientific architecture — PASS

The main design remains excellent:

```text
float64 scientific state
    ↓
coordinate/distance inspector
    ↓
read-only rows
```

not:

```text
render scene
    ↓
reverse-engineer science
```

Key accepted properties:

- Astropy owns celestial coordinate transforms;
- `CoordinateRow` requires frame + unit metadata structurally;
- direction-only Galactic coordinates can exist without distance;
- detached systems retain local data and no fake absolute values;
- host↔planet distance is derived from propagated float64 position;
- \(r=a(1-e\cos E)\) remains an independent regression, not a second production implementation;
- periapsis/apoapsis preserve input provenance;
- display exaggeration, LOD, camera placement, and orbit sampling cannot affect inspector science.

---

# 11. Verification result — PASS

Reported verification:

```text
architecture / golden-rule             31 passed
full suite                              808 passed
offline check (4 systems)              pass
verify_gl.py                            pass
GL backend                              29 passed
Explorer C1                             19 passed
Explorer C2                             27 passed
Explorer C3                             44 passed
all four demos                         pass
frame-count assertion                  11 frames >= 11
```

Progression:

```text
755 C2 baseline
763 depth-mask corrective commit
808 C3 commit
```

This is sufficient for a remote CI checkpoint.

---

# 12. `.claude/` — use local exclude, not repository `.gitignore`

Because `.claude/` is described as **local tooling configuration**, my preference is:

```text
.git/info/exclude
```

rather than committing:

```text
.claude/
```

to the repository `.gitignore`.

Why:

- it keeps machine/user-specific tooling out of `git status`;
- it does not impose a project-wide policy;
- it leaves open the possibility of intentionally sharing Claude/project configuration later.

Add locally:

```bash
echo .claude/ >> .git/info/exclude
```

On PowerShell:

```powershell
Add-Content .git/info/exclude ".claude/"
```

Only add `.claude/` to the tracked `.gitignore` if the project has deliberately decided that **no files in that directory should ever be versioned by any contributor**.

Do not amend either verified commit just for this local exclusion.

---

# 13. Push approval

The remote `3D-test` branch is still at the certified C2 commit, so the two local commits can now travel together as one normal fast-forward update.

Preflight:

```bash
git fetch origin
git merge-base --is-ancestor origin/3D-test HEAD
git status --short
```

After locally excluding `.claude/`, the working tree should be clean.

Then:

```bash
git push origin HEAD:3D-test
```

Do **not** force-push.

If the ancestry check fails or Git reports non-fast-forward, stop and reconcile rather than overriding the remote branch.

---

# 14. What to report after the push

Send:

```text
branch: 3D-test
remote HEAD: <full SHA of f470cfd>
parent corrective commit: <full SHA of b8b4457>
CI run: <run id>
CI status: green / failing
```

The remote audit will verify both commits independently where useful, especially:

- framebuffer state implementation;
- behavioral depth regression;
- C3 no-celestial-triplet guard;
- Astropy transform boundaries;
- provenance propagation;
- CI C3 step.

---

# 15. C3 milestone status

```text
repository recovery                    PASS
corrective rendering bisect point      PASS
real framebuffer depth writes          PASS
pixel-level depth behavior              PASS
C2 provenance follow-up                PASS
float64 C3 inspector                   PASS
coordinate frame metadata              PASS
detached semantics                     PASS
absolute planet ICRS withheld          PASS
planet→other-star exact distance        DEFERRED CORRECTLY
phase provenance row                   PASS
display-invariance tests               PASS
808 local tests                        PASS
remote CI                              NEXT
```

## Approved to push.

---

# 16. After C3 closes remotely

The next scientific prerequisite should be the **SystemFrame→ICRS basis definition** before implementing exact:

```text
planet absolute celestial position
selected exoplanet -> selected star distance
```

A correct implementation will need the host's local ICRS tangent triad and an explicit definition of how the orbital-reference axes map into it.

Do not smuggle that into a plotting/UI milestone.

Treat it as a small coordinate-physics vertical slice when C3 is remotely closed.
