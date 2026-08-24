# Explorer C3.5 Final Remote Audit — Rewrite Verified, One Integration Fix Before C3.6

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote HEAD:** `d7082f8852508581f3617ab2bc5e5cef14dd8158`  
**CI run:** `32672800734`  
**Status:** **PASS WITH ONE REQUIRED FOLLOW-UP**

## 1. Rewrite and remote CI

The history rewrite succeeded and the active `3D-test` branch now points to `d7082f8852508581f3617ab2bc5e5cef14dd8158`.

The rewritten chain is:

```text
5129422  Explorer C2
d9224cf  Fix framebuffer depth writes for scientific overlays
9ccdd85  Explorer C3: coordinate and distance inspector
d7082f8  Explorer C3.5: SystemFrame to ICRS basis
```

GitHub Actions run `32672800734` is green on this exact head, including the named C3 and C3.5 test steps, full suite, offline check, software-Mesa OpenGL verification, demos, and rendered-frame artifact.

## 2. C3.5 tangent-basis mathematics — PASS

Canonical frame:

```text
+X = East
+Y = North
+Z = away from observer
```

Astronomical position angle is converted using:

\[
\theta_{\rm internal} = \frac{\pi}{2} - \Omega_{\rm PA}
\]

The tangent triad, determinant, LOS sign, modulo-180 node ambiguity, pole singularity, and epoch gate are all correctly represented and tested.

## 3. Required integration fix: production propagation still consumes raw Ω

The final remote audit found an important integration gap.

`physics/orientation.py` expects `longitude_of_ascending_node` to be the **mathematical azimuth measured from the internal +X axis**.

But `OrbitalElements._display_angles()` still passes:

```python
elements.longitude_of_ascending_node.value_in(u.rad, 0.0)
```

directly into the production transform.

Likewise `orientation_guides()` reads:

```python
node = display.longitude_of_ascending_node
node_rad = node.value_in(u.rad, 0.0)
```

and passes that raw value into the guide geometry.

Therefore `position_angle_to_azimuth()` is correctly implemented and tested, but it is not yet the production route by which a catalogued PA reaches orbit propagation.

A future real node stored as PA East of North would therefore still be interpreted as an internal mathematical azimuth.

## 4. Display-normalisation consequence

If the UI says:

```text
Omega_PA = 0 deg
```

that means North.

With internal `+X = East`, the correct internal azimuth is:

\[
\theta = \pi/2
\]

not zero.

So a normalized astronomical `Ω=0°` must also pass through the conversion. Otherwise the displayed orbit is normalized East while the text says North.

## 5. Recommended C3.5.1 patch

Make one normal fast-forward follow-up commit.

Prefer moving shared node semantics into a physics-level module such as:

```text
physics/node_semantics.py
```

or extending:

```text
physics/orbital_semantics.py
```

with:

```text
NodeConvention
NodeSense
NodeSenseEvidence
position_angle_to_azimuth
azimuth_to_position_angle
node_convention_of
node_sense_of
resolve_node_azimuth
```

Then:

1. `_display_angles()` must use the resolved internal azimuth.
2. `orientation_guides()` must use that same resolved azimuth.
3. UI annotations may display the raw published astronomical PA.
4. Raw PA must never be passed directly into `rotation_perifocal_to_inertial`.
5. A raw normalized `Omega_PA=0°` must become internal `theta=pi/2`.

## 6. Required end-to-end tests

Add at least:

```text
raw PA 0°   -> propagated node points North
raw PA 90°  -> propagated node points East
raw PA 180° -> propagated node points South
raw PA 270° -> propagated node points West

position_at_* uses canonicalized node
state_at_* uses canonicalized node
orientation guide node agrees with propagated orbit
periapsis guide agrees with propagated periapsis
normalized Omega_PA=0° is displayed North
raw catalog PA never reaches rotation_perifocal_to_inertial directly
```

## 7. Strengthen node-sense evidence before epoch publication opens

`node_sense_of()` currently requires a non-empty evidence string for `RESOLVED`, but any non-empty string will pass.

So a future bad ingestion value such as:

```text
node_sense_evidence = "systemic radial velocity"
```

could still unlock the gate even though the documentation correctly says this is insufficient.

Before the epoch gate can publish real positions, make evidence typed, for example:

```text
ORBITAL_RV_SOLUTION
DIRECT_COMPANION_LOS_VELOCITY
ASTROMETRY_PLUS_ORBITAL_RV
OTHER_ORBIT_SPECIFIC_LOS
SYSTEMIC_RADIAL_VELOCITY
```

with the last one explicitly non-resolving.

Keep a free-text/reference field alongside the evidence kind.

## 8. Milestone status

```text
history rewrite                         PASS
active attribution cleanup              PASS
remote CI                               PASS
canonical tangent triad                 PASS
PA conversion function                  PASS
LOS sign                                PASS
node modulo-180 gate                    PASS
pole gate                               PASS
epoch gate                              PASS
raw PA -> production orbit integration  REQUIRED FOLLOW-UP
typed node-sense evidence               REQUIRED BEFORE EPOCH GATE OPENS
```

Therefore:

**C3.5 = PASS WITH ONE REQUIRED FAST-FORWARD FOLLOW-UP.**

## 9. After C3.5.1: Explorer C3.6 — astrometric epoch and space motion

Current `SkyPosition` contains:

```text
RA
Dec
distance
frame
```

but not:

```text
obstime
pm_ra_cosdec
pm_dec
radial_velocity
```

The next coordinate-physics slice should add a provenance-aware astrometric state and let Astropy own space-motion propagation.

Recommended knowledge tiers:

```text
reference-epoch position:
    RA + Dec + distance + obstime

direction propagation:
    obstime + proper motions

full 3D cross-epoch state:
    distance + obstime + pm_ra_cosdec + pm_dec + radial_velocity
```

Do not invent J2000, zero proper motion, or zero radial velocity when data are absent.

If requested time equals the actual reference epoch, a position can be valid without motion propagation.

For arbitrary time, require the motion fields appropriate to the advertised result.

## 10. Common-time rule

For absolute planet position at time `t`:

```text
propagate host to t
propagate orbital phase to t
build tangent basis from propagated host RA/Dec at t
rotate planet local offset into ICRS
add host and planet offset in float64
```

For planet-to-selected-star distance:

```text
propagate host to t
propagate selected star to t
propagate planet to t
form planet absolute ICRS vector
compute distance in one common float64 frame
```

Never mix catalog epochs.

## 11. C3.6 tests

At minimum:

```text
missing obstime blocks cross-epoch publication
requested time == reference epoch works without inventing motion
missing proper motion is not treated as zero
missing radial velocity is not treated as zero for full 3D propagation
explicit zero motion remains unchanged
full 6D state matches Astropy apply_space_motion
forward/backward propagation round-trip
RA wrap handled
high-proper-motion synthetic fixture moves substantially
pm_ra_cosdec semantics correct
host tangent basis rebuilt from propagated RA/Dec
host and target evaluated at same time
planet absolute position uses same requested time
planet->star never falls back to host->star
detached systems remain unavailable
all coordinate math stays float64
renderer remains absent from the coordinate-physics layer
```

## 12. Loose review file

Move:

```text
C35_REWRITE_GATE_REMOTE_CONFIRMATION.md
```

into:

```text
docs/reviews/archive/
```

with the next normal follow-up commit.

Do not make a standalone push for it.

## Immediate instruction

1. Archive the loose rewrite-gate review with the next commit.
2. Make C3.5.1 as a normal fast-forward patch:
   - centralize node semantics;
   - wire PA→azimuth conversion into production propagation and C2 guides;
   - fix display-normalization semantics;
   - type node-sense evidence.
3. Run the full suite + Mesa CI-equivalent verification.
4. Push normally.
5. Send the SHA and CI result.
6. Only after that start C3.6 astrometric epoch and space motion.
