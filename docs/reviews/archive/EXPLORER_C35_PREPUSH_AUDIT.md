# Explorer C3.5 Pre-Push Audit

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote tip verified:** `f470cfd4e42d2ed6dce02039226bbd22aa43672b`  
**Local C3.5 commit reported:** `8b55b6f`  
**Verdict:** **DO NOT PUSH YET.**

## What passes

The tangent-basis work is strong:

- float64 local basis;
- orthonormality and determinant +1 tests;
- direct LOS-sign regression;
- Astropy finite-difference checks for East/North;
- SystemFrame↔ICRS round trip;
- unknown-node publication gate;
- no host-to-star fallback disguised as planet-to-star distance;
- named C3.5 CI step;
- roadmap updated;
- 889 passed, 1 skipped locally.

## P0 — node convention is not yet proven compatible with the chosen +z sign

Your current basis is mathematically self-consistent:

```text
+x = North
+y = East
+z = star -> observer
```

because `North × East = -e_r`.

But standard astrometric conventions often identify the **ascending node with the receding node**. Common direct-imaging/astrometry tooling also uses position angle East of North while defining positive LOS motion as redshift/away from the observer.

With the production transform:

`Rz(Omega) Rx(i) Rz(omega)`

a body just after the `u=0` node moves toward positive local z. If your +z points toward the observer, that matrix's "ascending" crossing is approaching, not receding.

So the source convention must be separated from the internal Cartesian convention.

Recommended canonicalization:

```text
+X = East
+Y = North
+Z = away from observer
```

Then an astronomical position angle `Omega_PA` measured East of North maps to a mathematical azimuth:

`theta_math = pi/2 - Omega_PA`

before the internal `Rz()` rotation.

Equivalent conventions are possible, but the source→canonical conversion must be explicit and provenance-tagged.

Add something like:

```text
NodeConvention.PA_EAST_OF_NORTH_RECEDING
```

or normalize all ingested nodes into one canonical convention.

An unstated node convention must not unlock an absolute planet position.

## P0 — a numeric measured Omega may still be ambiguous by 180 degrees

`Omega.status == MEASURED` is not enough.

Relative astrometry commonly admits:

`omega' = omega + pi`

`Omega' = Omega - pi`

with the same projected orbit.

The ascending/descending sense is often resolved only by RV or equivalent LOS information.

Add a semantic state such as:

```text
NodeSense.RESOLVED
NodeSense.MODULO_180
NodeSense.UNKNOWN
```

Publication rule:

```text
angle known + convention known + node sense resolved
    -> unique physical 3D orientation may be published

angle known only modulo 180
    -> do not publish one unique absolute planet ICRS position
```

A normalized display solution remains `ASSUMED_FOR_VISUALIZATION`.

## P0/P1 — coordinate epoch and space motion are still missing

The remotely audited `SkyPosition` currently has RA, Dec, distance, and frame, but no:

```text
obstime / coordinate epoch
proper motion
radial velocity
```

So an "instantaneous absolute planet position" at orbital `time_jd` can otherwise mix:

`host position at catalog epoch + planet offset at requested epoch`.

For nearby/high-proper-motion stars, that can be a larger physical error than the planet's AU-scale offset.

Preferred current scope:

- ship the **local SystemFrame→ICRS tangent-vector transform**;
- continue withholding exact instantaneous planet absolute position and exact planet→selected-star distance until coordinate epoch/space motion are modeled.

Alternatively, add Astropy space-motion propagation in this slice and require all objects to be evaluated at one common epoch.

## Pole singularity

At Dec = ±90°, RA is not a unique physical direction. A closed-form tangent basis can still be algebraically orthonormal, but its azimuth is gauge-dependent.

Do not claim a unique physical orbital position angle at the exact celestial pole. Add an explicit singularity policy or mark orientation-dependent absolute output unavailable there.

## History rewrite

The remote branch is still at `f470cfd...`.

Do **not** force-push the current `8b55b6f` yet. Fix the scientific gates first so the shared branch is rewritten at most once.

### Preferred history path

If removing old co-author trailers is not mandatory, preserve public history and transplant the corrected C3.5 commit onto remote `f470cfd`, then normal fast-forward push.

### If trailer removal is mandatory

Because `3D-test` is shared:

1. confirm nobody has based unmerged work on the published C3 commits;
2. create a backup ref at the exact old remote tip;
3. fetch immediately before rewriting;
4. use force-with-lease pinned to the exact expected SHA;
5. never replace a failed lease with plain `--force`.

Backup:

```bash
git push origin f470cfd4e42d2ed6dce02039226bbd22aa43672b:refs/heads/archive/3D-test-pre-attribution-rewrite
```

Then, only after the science fixes:

```bash
git push --force-with-lease=refs/heads/3D-test:f470cfd4e42d2ed6dce02039226bbd22aa43672b origin HEAD:refs/heads/3D-test
```

If the lease fails, stop.

## Required extra regressions

```text
[ ] PA=0 deg points North
[ ] PA=90 deg points East
[ ] LOS sign matches the documented receding/approaching convention
[ ] source node-convention mismatch cannot silently pass through
[ ] Omega modulo 180 does not unlock unique absolute position
[ ] RV-resolved node sense can unlock it
[ ] Omega+180 / omega+180 projected degeneracy is reproduced
[ ] exact pole does not claim a unique physical PA basis
[ ] epoch mismatch blocks "instantaneous absolute" output
[ ] planet->selected-star distance requires a common coordinate epoch
```

## Immediate instruction

1. Do not push yet.
2. Canonicalize/provenance-tag node convention.
3. Add node-sense ambiguity state.
4. Keep absolute instantaneous position/distance withheld unless coordinate epochs are handled.
5. Handle exact-pole singularity.
6. Re-run full suite.
7. Send revised SHA + test count.
8. Then decide fast-forward vs backed-up force-with-lease.
