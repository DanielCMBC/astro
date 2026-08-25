# Explorer C3.5.1 Final Remote Audit and Explorer C3.6 Specification

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote HEAD:** `1508738d3d14888e3799012a9f57a4d3d6d47e1f`  
**CI run:** `32677618580`  
**Verdict:** **EXPLORER C3.5.1 CLOSED — PASS. START C3.6.**

## Remote verification

The public branch points to the exact C3.5.1 SHA. GitHub Actions is green on Python 3.11, Python 3.12 and software-Mesa OpenGL. The rendered-frames artifact belongs to the same head SHA.

## C3.5.1 scientific audit

The production paths now canonicalize catalogued longitude of ascending node through `resolve_node_azimuth()`:

- `OrbitalElements._display_angles()` for `position_at_*` / `state_at_*`
- `SystemSlice.state()`
- `orientation_guides()`

This fixes both published nodes and the common display-normalized `Omega_PA = 0°` case. In the canonical frame `+X=East`, `+Y=North`, so astronomical PA 0° means North and must become internal azimuth `pi/2`.

`physics/node_semantics.py` is now the shared semantic boundary. `coordinates/tangent.py` re-exports instead of duplicating the implementation.

`NodeSenseEvidence` is typed. `SYSTEMIC_RADIAL_VELOCITY` is explicitly non-resolving for a planet's node, while orbit-specific line-of-sight evidence can resolve it.

The bug-injection tests are strong because they fail behaviorally, by guide/propagator consistency, and structurally.

## One non-blocking hardening

Do not add the proposed rule that every module reaching the low-level orbital transform must import `resolve_node_azimuth`. Low-level numerical modules should remain convention-agnostic.

A better future structure is one provenance-aware orientation resolver, for example:

```python
@dataclass(frozen=True)
class ResolvedOrientation:
    inclination: float
    argument_of_periapsis: float
    node_azimuth: float
```

Then high-level `OrbitalElements` consumers obtain canonical angles once, while `physics/orientation.py` continues accepting plain mathematical angles.

Also harden `resolve_node_azimuth()` before any future provider ingests real node values: a scientific node with `NodeConvention.UNSPECIFIED` must not silently be treated as the standard PA convention. Display normalization may still explicitly assume PA=0°.

---

# Explorer C3.6 — astrometric epoch and space motion

C3.6 should replace the last deliberately closed gate in the absolute-position pipeline: coordinate epoch / stellar space motion.

Current `SkyPosition` still has RA, Dec, distance and frame, but no reference epoch, proper motion or radial velocity.

The temporary `epoch_resolved=True` test seam must not become a production bypass. Replace it with an actual astrometric state evaluated at a concrete time.

## Recommended model

Prefer a new type first:

```python
@dataclass(frozen=True)
class AstrometricState:
    position: SkyPosition
    source_catalog: str
    source_id: str | None
    release: str | None
    reference_epoch: Time | None
    pm_ra_cosdec: Parameter
    pm_dec: Parameter
    radial_velocity: Parameter
    reference: str | None
```

and a result type such as:

```python
@dataclass(frozen=True)
class PropagatedAstrometry:
    position: SkyPosition
    obstime: Time
    status: Status
    source_state: AstrometricState
    blockers: tuple[str, ...]
```

`absolute_planet_position()` and `planet_to_star_distance()` should consume real propagated states instead of a naked boolean.

## Gaia DR3 as preferred astrometric authority

The NASA Exoplanet Archive PS table exposes `gaia_dr3_id`. Use that deterministic identifier when available rather than cone-matching by name.

For Gaia DR3, preserve:

- `source_id`
- `ref_epoch`
- `ra`, `dec`, `parallax`
- `pmra`, `pmdec`
- `radial_velocity`
- `astrometric_params_solved`
- uncertainties for all of those values
- preferably the astrometric correlation coefficients too

Gaia DR3 uses reference epoch J2016.0, ICRS positions/proper motions, and TCB time coordinate. Gaia `pmra` is `mu_alpha* = d(alpha)/dt cos(delta)`, so it maps directly to Astropy `pm_ra_cosdec`; do not apply `cos(dec)` again.

## Online refresh / offline runtime

Architecture:

```text
online sync:
NASA host identity -> gaia_dr3_id -> Gaia TAP -> validate -> atomic local cache

offline runtime:
validated local cache -> AstrometricState -> Astropy propagation
```

Application startup and rendering must not depend on live Gaia/NASA access. A failed refresh must leave the previous validated cache usable.

The previously deferred `data/gaia.py` is now justified because C3.6 will actually consume it.

NASA PS also exposes `sy_pmra`, `sy_pmdec` and systemic `st_radv`, but do not use those to open the strict epoch gate unless their coordinate epoch is explicit. Do not infer J2000 merely from generic IPAC coordinate conventions.

## Important RV distinction

Systemic stellar RV is useful for **stellar 3D space motion**. It is still not evidence that resolves a planet's ascending node. Keep those type boundaries separate.

## Astropy owns propagation

Use `SkyCoord` with explicit:

```text
ra
dec
distance
pm_ra_cosdec
pm_dec
radial_velocity
obstime
```

and `apply_space_motion(new_obstime=...)`.

Do not hand-write proper-motion propagation, and do not silently turn missing motion components into zero.

## Knowledge tiers

Reference-epoch 3D position:
- RA, Dec, distance, reference epoch
- if requested time equals the reference epoch, no motion needs to be invented

Direction propagation:
- reference epoch + pm_ra_cosdec + pm_dec
- may support a shifted sky direction
- do not automatically call this a complete cross-epoch 3D position

Strict full cross-epoch 3D state:
- RA, Dec, distance, reference epoch, both proper-motion components, radial velocity
- propagate under an explicitly documented uniform rectilinear space-motion model

## Time semantics

Do not create a new astrometric API taking only `time_jd: float`. Use `astropy.time.Time` or a project object carrying the time scale explicitly.

Define one conversion point from the explorer's physical clock to the Astropy `Time` used for stellar propagation.

## Common-time rule

For absolute planet position at time `t`:

1. propagate host astrometry to `t`
2. propagate orbital phase/state to the same `t`
3. rebuild the host tangent basis from propagated RA/Dec
4. rotate the local planet offset through canonical node semantics
5. add in float64 ICRS

For planet-to-selected-star distance:

1. propagate host to `t`
2. propagate selected star to `t`
3. propagate planet orbit to `t`
4. form planet absolute ICRS vector
5. compute the norm in the same float64 frame

Never mix catalog epochs.

## First fixtures

Use:
- a synthetic high-proper-motion star
- a synthetic star with explicitly measured zero motion
- at least two Gaia-backed real exoplanet hosts
- the existing detached/unlocated regression case

## Required tests

```text
[ ] gaia_dr3_id maps deterministically to Gaia source_id
[ ] Gaia ref_epoch is stored, not inferred
[ ] Gaia pmra maps directly to pm_ra_cosdec
[ ] missing PM/RV remain UNKNOWN, never zero
[ ] explicit measured zero motion remains zero
[ ] reference-epoch position works without invented motion
[ ] missing epoch blocks cross-epoch publication
[ ] full state matches Astropy apply_space_motion
[ ] high-PM fixture moves substantially
[ ] forward/backward propagation round-trips within tolerance
[ ] RA wrap is handled
[ ] host tangent basis uses propagated RA/Dec
[ ] host, target star and planet use exactly the same target Time
[ ] epoch_resolved=True is removed as a production escape hatch
[ ] planet->star never falls back to host->star
[ ] node convention/sense gates remain independent
[ ] online sync writes cache atomically
[ ] failed refresh preserves old cache
[ ] normal runtime performs zero network access
[ ] offline check includes astrometric state
[ ] renderer imports no astrometric science code
[ ] all earlier C1-C3.5.1 tests remain green
```

## Scope boundary

C3.6 includes astrometric state, Gaia provider/cache, epoch, proper motion, radial velocity, common-time propagation, and replacing the epoch boolean with real state.

Keep C4, spectra UI, gravity and free-flight camera out of this slice.

## Immediate instruction

**Start C3.6 now.**

At the start of the slice:
1. add the `UNSPECIFIED NodeConvention` rendering hardening;
2. consider routing `SystemSlice.state()` through the existing provenance-aware orbital-elements propagation wrapper to remove another manual angle path;
3. implement `AstrometricState`;
4. use Gaia DR3 ID / Gaia DR3 as the preferred authoritative astrometric source;
5. cache validated astrometry for offline runtime;
6. replace the fake epoch boolean with actual propagated state at a concrete `Time`;
7. keep C4 untouched.

Do not push until the C3.6 slice has been locally audited.
