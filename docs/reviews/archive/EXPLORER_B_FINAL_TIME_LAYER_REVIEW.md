# Explorer B Final Time-Layer Review

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Working state:** local/uncommitted Explorer B follow-up  
**Remote state at review time:** still on the previous Explorer B commit  
**Verdict:** The HJD/JD split is correct. Keep the 8 s HJD bound. Two time-model issues should be corrected before the final Explorer B commit: the UTC→TDB term should not be a permanent hard-coded constant, and `mean_anomaly_at_epoch` needs an actual epoch date before it can drive propagation.

---

## 1. HJD / JD uncertainty split — approved

The new split is correct:

```python
BARYCENTRIC_MINUS_HELIOCENTRIC_MAX_SECONDS = 8.0
BARYCENTRIC_MINUS_GEOCENTRIC_MAX_SECONDS = 499.0
```

Current intended bounds:

| Scale | Residual bound |
|---|---:|
| `BJD_TDB` | 0 s |
| `BKJD` | 0 s after mission offset |
| `BTJD` | 0 s after mission offset |
| `HJD_UTC` | ~77.184 s |
| `JD_UTC` | ~568.184 s |
| `JD_UNSPECIFIED` | ~568.184 s conservative fallback |

This corrects the earlier conflation between:

- the heliocentric→barycentric correction, which is only several seconds;
- the geocentric→barycentric light-time correction, which is roughly one AU / c.

### Keep 8.0 s, not 5.3 s

Do **not** tighten the HJD bound to 5.3 s merely from a single displacement estimate.

The published Eastman/Siverd/Gaudi BJD guidance explicitly states that using the heliocenter instead of the Solar System barycenter can introduce errors as large as about **8 seconds**.

For a conservative scientific bound, `8.0` is the better number.

---

## 2. One more timing refinement: `69.184 s` is not a universal permanent TDB−UTC constant

The current code appears to treat:

```python
TDB_MINUS_UTC_SECONDS = 69.184
```

as a fixed quantity.

That is a good approximation for the current epoch, but scientifically it is not a universal constant.

Conceptually:

```text
TDB - UTC
=
(TT - UTC)
+
(TDB - TT)
```

where:

- `TT - UTC` depends on accumulated leap seconds;
- `TDB - TT` has a small periodic relativistic term at roughly the millisecond level.

Astropy already knows how to perform these scale conversions.

### Recommendation

Do not make the production time model permanently depend on a literal `69.184`.

Prefer a helper that evaluates the temporal-scale difference for the date being handled.

The exact implementation should use Astropy's high-precision `Time` representation rather than subtracting ordinary float JDs if sub-millisecond precision matters.

This avoids:

- becoming stale after a future leap second;
- pretending `TDB - TT` is exactly zero;
- encoding a 2026-era scale relationship as a physical constant.

This is a P1 precision/maintainability correction.

---

## 3. P0 model issue exposed by `Epoch.is_dated`: mean anomaly needs a date

This line in the latest output is important:

> `MEAN_ANOMALY_AT_EPOCH` stores an angle in radians — the old epochs list included it.

`Epoch.is_dated` correctly prevents that angle from being treated as a date.

However, the deeper model issue remains:

> A mean anomaly "at epoch" is only useful for time propagation if the corresponding epoch **date** is also known.

The correct equation is:

\[
M(t)=M_0+n(t-t_0)
\]

where:

- \(M_0\) is the mean anomaly at the reference epoch;
- \(t_0\) is the actual reference epoch date.

A bare:

```text
mean_anomaly_at_epoch = 1.14 rad
```

without:

```text
epoch_of_mean_anomaly = JD ...
```

cannot determine `M(t)` for an arbitrary requested date.

### Recommended model

Do not represent this as an `EpochKind` whose value is an angle.

Instead introduce a phase anchor such as:

```python
@dataclass(frozen=True)
class MeanAnomalyAnchor:
    anomaly: Parameter      # radians
    epoch: Epoch            # dated reference instant
```

or equivalent fields:

```text
mean_anomaly_at_epoch
mean_anomaly_epoch
```

Then:

```python
if anomaly.is_known and epoch.is_known:
    M = M0 + n * (t - t0)
else:
    absolute phase is not computable
```

### Required behavior

If:

```text
M0 known
t0 unknown
```

then preserve the published angle, but do not claim a current physical phase.

Suggested status:

```text
PARTIALLY_CONSTRAINED
```

or:

```text
REFERENCE_ANOMALY_UNDATED
```

### Required tests

```text
test_mean_anomaly_anchor_requires_a_dated_epoch_for_propagation
test_undated_mean_anomaly_does_not_claim_current_position
test_dated_mean_anomaly_propagates_with_mean_motion
test_mean_anomaly_anchor_round_trips_one_period
```

This should be treated as P0 even if none of the current reference systems exercises it.

---

## 4. `canonical_jd` — approved

`canonical_jd` is the correct production name.

It expresses:

> full Julian-day numbering after mission-offset normalization

without falsely claiming:

> exact BJD_TDB.

Keep:

```text
source_scale
scale_uncertainty
source_kind
source_name
```

beside the canonical number.

The legacy:

```text
as_bjd()
```

may remain temporarily as a deprecated alias.

Add the guard test already planned:

```text
test_no_new_production_callsite_uses_as_bjd
```

---

## 5. `physics.TimeController` rename — approved

The rename is correct:

```text
epoch_bjd       -> epoch_jd
simulated_bjd() -> simulated_jd()
now_bjd         -> now_jd
time_bjd        -> time_jd
```

Keeping deprecated aliases for:

```text
epoch_bjd
simulated_bjd()
```

is reasonable.

Renaming `now_bjd` outright is also reasonable because stale keyword callers should fail loudly.

Keep the compatibility regression:

```text
test_the_old_bjd_names_still_work_but_warn
```

---

## 6. Camera state — approved

The new mutually-exclusive representation is a significant improvement:

```text
camera_absolute_pc: ndarray | None
camera_local: FramedPosition | None
```

A detached camera being:

```text
FramedPosition([0, 0, 3], SystemFrame[AU])
```

is exactly the right semantic model.

Keep the invariant:

```python
(camera_absolute_pc is None) != (camera_local is None)
```

and retain tests around:

- detached local motion;
- located absolute navigation;
- detached→located focus;
- refusal to convert unlocated local coordinates into absolute space.

---

## 7. Epoch-selection policy — approved

The current policy is good:

```text
selected planet
    ↓
periastron epoch
    ↓
transit epoch
    ↓
fallback
```

with record order only as a deterministic final tie-break.

This is much better than silently taking the first epoch in catalog order.

---

## 8. Identity guarantee wording — approved

The corrected statement is accurate:

> `entity_id` is stable while the authoritative catalogue key remains unchanged.

Do not call it rename-proof.

Permanent cross-catalog identity belongs to the future synchronized database layer.

---

## 9. Review-file cleanup — do it before the final commit

Complete the documentation hygiene:

1. move durable decisions into:
   - `docs/explorer-b.md`
   - `docs/physics.md`
   - `docs/roadmap-status.md`;
2. delete the generated root review file:
   - `EXPLORER_B_REVIEW_TIME_SCALE_AND_NEXT_STEPS.md`;
3. optionally create `docs/reviews/archive/` if historical review snapshots are worth retaining.

The repository root should not become a pile of generated feedback documents.

---

## 10. Final pre-push checklist

```text
[ ] keep HJD bound at 8.0 s
[ ] keep JD geocentric bound around 499 s
[ ] stop treating 69.184 s as a permanent universal TDB−UTC constant
[ ] add a date-aware Astropy scale-offset path or bounded dynamic helper
[ ] model mean-anomaly phase anchor with both M0 and t0
[ ] ensure undated M0 cannot produce "current position"
[ ] add mean-anomaly-anchor regression tests
[ ] add no-new-as_bjd-callsite guard
[ ] archive/delete root Explorer B review MD
[ ] run complete suite
[ ] run headless GL CI-equivalent tests
[ ] commit
[ ] push
```

---

## 11. Milestone decision

Explorer B is now **very close to closure**.

The HJD/JD split is accepted.

The remaining substantive scientific issue is the latent:

```text
mean anomaly at epoch without epoch date
```

model.

Fix that before the final push.

The dynamic UTC→TDB treatment is also worth doing now while the time layer is already open.

After those are green, Explorer B can be closed and Explorer C can begin:

```text
habitable-zone rendering
orbital orientation overlays
distance/coordinate inspector
HR diagram
blackbody spectrum
atmospheric spectroscopy
```
