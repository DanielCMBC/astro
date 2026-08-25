# Explorer C3.6: astrometric epoch and space motion

C3.5 supplied the SystemFrame → ICRS rotation and then declined to publish
an absolute planet position anyway. Three gates blocked it: the node's
**convention**, the node's **sense**, and the **coordinate epoch**. The
first two are statements about what the catalogue measured. The third was
not — it was a placeholder:

```python
def absolute_position_blockers(host, node, *, epoch_resolved: bool = False):
    ...
    if not epoch_resolved:
        reasons.append(EPOCH_NOT_MODELLED)
```

`epoch_resolved` defaulted to `False`, was never `True` anywhere in `src/`,
and was `True` in about a dozen places in the tests. It was honest while it
stayed a placeholder and dangerous the moment it did not: **one `True`
written at one call site would have opened every downstream scientific gate
without a single number changing anywhere in the program.**

C3.6 deletes it and supplies the state it stood in for.

## Why the epoch mismatch is not a rounding error

The thing being added is an AU-scale planet offset. The thing being ignored
was proper motion × elapsed time, and for a nearby star it is much bigger.

HD 219134 moves 2.1 arcsec/yr and sits 6.5 pc away. Ten years off epoch
displaces it by roughly **135 AU**. Adding a 0.2 AU planet offset to a host
position that is 135 AU wrong is not a small inconsistency — the answer is
dominated entirely by the term nobody modelled, and it looks like an
ordinary coordinate.

## Three knowledge tiers, not one flag

`MotionKnowledge` exists because "we know where this star is" has three
genuinely different meanings, and collapsing them is how a *direction*
becomes a *position*.

| tier | what the catalogue gave | what may be published |
|---|---|---|
| `REFERENCE_EPOCH_ONLY` | RA, Dec, distance, reference epoch | a full 3D position **at that epoch**, with no motion invented |
| `DIRECTION_ONLY` | + both proper-motion components | an approximate sky direction; **not** a cross-epoch 3D position |
| `FULL_SPACE_MOTION` | + radial velocity | a 3D state at any date, under a stated space-motion model |

The middle row is the one that carries the risk, and it carries two
distinct ones.

**It is not a position.** Propagating the direction and keeping the
reference-epoch distance produces a complete-looking 3D position that nobody
measured, so `propagate_astrometry` drops the distance in that tier and says
why:

```
proper motion moves the sky direction only; combining it with a
reference-epoch distance would report a cross-epoch 3D position that
was never measured
```

**And the direction it does produce is model-dependent.** Astropy's
`apply_space_motion` assumes RV = 0 when none is supplied, and for a
finite-distance star the line-of-sight velocity changes the *apparent
angular* motion through perspective acceleration. So `DIRECTION_ONLY` names
what the catalogue has, not what is known at another date:

```
DIRECTION_ONLY:
proper-motion information exists, but complete 3D space motion does not.
Cross-epoch direction is model-dependent when RV is unavailable.
```

Every value it produces is stamped `ZERO_RV_APPROXIMATION` rather than
`SPACE_MOTION_MODEL`, and `DIRECTION_ONLY_IS_MODEL_DEPENDENT` appears in the
blockers. The size of the ambiguity is a real number, not a caveat:
`test_a_missing_radial_velocity_leaves_the_direction_undetermined` takes one
nearby fast star at RV = -100 km/s and at RV = +100 km/s — two values the
catalogue cannot distinguish — and over four centuries their sky directions
differ by more than an arcminute. Nothing in the propagated numbers would
have said so.

TRAPPIST-1 is a real example, not a contrived one: Gaia DR3 publishes a
parallax and both proper motions for it and **no radial velocity**.

### A realization must never become a measurement

The way that would happen is chaining. Propagate a direction-only star,
read back the differential Astropy produced from its assumed zero, build a
new state from it, and the missing measurement has become a known one — the
state then ranks as `FULL_SPACE_MOTION` and publishes.

Two rules prevent it:

* `PropagatedAstrometry.as_state()` carries through the **original**
  `radial_velocity`, which for this tier is `UNKNOWN`. It never reads the
  radial component off the propagated coordinate;
* the returned state records `realized_from`, and `propagate_astrometry`
  rebases onto `observed_root()` before computing anything. A second
  propagation therefore starts from the observation, not from the model
  output, so no chain of hops can apply the approximation twice while
  looking like one careful propagation. Rectilinear motion makes the rebased
  answer identical for a full solution, so nothing is lost by doing it
  unconditionally.

## Missing is not zero

Astropy's `apply_space_motion` substitutes zero for any differential the
coordinate does not carry. That is the correct library behaviour and the
wrong science, so nothing here hands it a partial motion and reads the
result as a 3D state:

* an absent `radial_velocity` becomes an `UNKNOWN` `Parameter`, never `0.0`;
* one proper-motion component without the other is rejected outright — half
  a motion is a motion at the wrong position angle, and it looks exactly
  like a motion;
* an `ASSUMED_FOR_VISUALIZATION` component does not count as measured.

The rule only works if both halves hold, so the converse is pinned too: a
motion that was **measured and found to be zero** stays zero, propagates to
itself, and remains publishable.

## Gaia DR3 as the astrometric authority

The NASA archive gives RA and Dec with no `ref_epoch` column, and its
`sy_pmra` / `sy_pmdec` / `st_radv` are compiled values whose epoch is not
stated either. An unlabelled position cannot open the epoch gate however
many motion columns sit beside it — so C3.6 goes to the catalogue that does
state one.

**The identifier, not a cone search.** `ps.gaia_dr3_id` is a cross-match the
archive already made. Re-deriving it by position would substitute our guess
for their curated answer, and would do so most confidently in the crowded
fields where it is most likely wrong. `parse_gaia_dr3_id` returns a
**string**, because a DR3 `source_id` is a 64-bit integer and many exceed
2⁵³: round-tripping one through a float changes which star it names, and
changes it into another valid-looking 19-digit identifier.

What is preserved per source: `source_id`, `ref_epoch`, `ra`, `dec`,
`parallax`, `pmra`, `pmdec`, `radial_velocity`,
`astrometric_params_solved`, every uncertainty, and the ten astrometric
correlation coefficients — the five parameters are jointly fitted, so the
covariance is not reconstructible from the diagonal errors, and re-querying
is the step that needs a network.

The refresh path fails closed on identity before it validates anything:
a returned `source_id` that was not requested is another star, and a
duplicate means `gaia_source` did not return what it claims to. Both raise
`GaiaIdentityError` and leave the previous cache untouched. **There is no
cone-search fallback** — if an exact NASA cross-match identifier fails to
resolve, the honest outcome is no astrometry for that host, not the
astrometry of whatever sits nearest a position whose epoch we could not
state. A requested id simply absent from the release is not an error; the
rest of the batch is still cached.

The offline read path fails closed too: the document declares a
`schema_version` and a `release`, and an unrecognised value for either is
refused rather than read optimistically. Column names are stable across Gaia
releases and their reference epochs are not, so a best-effort read would
propagate from the wrong year in silence.

Two conventions are handled explicitly because both fail silently:

* **`ref_epoch` is stored, not inferred.** Hard-coding J2016.0 works for DR3
  and keeps working, wrongly and quietly, into whatever release changes it.
  The documented value is used only to *reject* a row that disagrees.
  Assuming J2000 instead is sixteen years of motion — 33 arcsec for
  HD 219134;
* **`pmra` maps directly to `pm_ra_cosdec`.** Gaia's `pmra` is already
  μ<sub>α</sub>\* = α̇·cos δ. Applying `cos(dec)` again is exact at the
  equator and wrong by a factor of two at 60°, so a test written against one
  low-declination star would pass with the bug present. HD 219134 sits at
  +57°, where `cos(dec) = 0.54`.

## Online refresh, offline runtime

```
online sync   (scripts/sync_gaia_astrometry.py, run deliberately)
  NASA host identity -> gaia_dr3_id -> Gaia TAP -> validate -> atomic cache

offline runtime  (everything the application does)
  validated local cache -> AstrometricState -> Astropy propagation
```

Startup and rendering perform **no** network access. `requests` is imported
in exactly one function, `fetch_gaia_astrometry`, and the architecture test
checks that structurally rather than trusting the docstring. The runtime
test forbids socket construction outright, so a live request anywhere under
`build_slice` fails the suite rather than merely being slow.

A refresh that fails at any step — no connectivity, a TAP outage, an empty
response, a source that fails validation, a rename that hits a full disk —
leaves the previous validated cache exactly where it was. An explorer that
cannot show a position because a TAP service was down this morning is worse
than one showing yesterday's identical astrometry.

## The common-time rule, made structural

For an absolute planet position at time *t*:

1. propagate the host astrometry to *t*;
2. propagate the orbital phase to the same *t*;
3. rebuild the host tangent basis from the **propagated** RA/Dec;
4. rotate the local offset through canonical node semantics;
5. add in float64 ICRS.

Step 3 is where this is easy to get wrong: building the triad from the
catalogue RA/Dec while adding the offset to the propagated origin rotates
the offset away from the position it belongs to. So
`absolute_planet_position` *replaces* the host with the propagated position
when one is supplied — origin and basis together, from one object. There is
no code path that can take them from different epochs.

### The planet has a clock too

Steps 1 and 3 were structural from the start. Step 2 was not, and the gap
was invisible because the arithmetic never complains:

```python
absolute_planet_position(host, position_au, astrometry=propagated)
```

`position_au` is a bare `(3,)` array of AU. It has no epoch, so it cannot
disagree with `propagated.obstime` — a host propagated to 2035 and an orbit
propagated to 2025 add together perfectly and produce a coordinate about a
different night. The offset is the *small* term; getting its epoch wrong
does not make the answer slightly worse.

So the offset stops being a bare array.
`TimedOrbitalState` (`physics/timed_state.py`) carries the propagator's
float64 state, the `Time` it holds at, and the phase provenance. A raw
vector still draws — drawing needs no common epoch — but it cannot open a
scientific gate, because the thing the gate checks is not present on an
array:

```
the planet's orbital state carries no instant, so it cannot be shown to
hold at the same time as the host's propagated position
```

A dated state additionally has to *match* the host's instant
(`PLANET_TIME_MISMATCH`) and to have reached it from a published epoch
rather than an arbitrary zero (`PLANET_PHASE_NOT_CONSTRAINED`). The second
is not the same question as the first: a phase advanced from the Unix epoch
has an exact `obstime` that matches the host's exactly, and is still a
picture of the motion. A *transit-epoch* phase read through a normalised
argument of periastron does pass, because its timing is a real observation
and its orientation is already gated twice by the node checks — refusing it
here would double-count.

### Three clocks, not two

```
host astrometry obstime
    == target-star astrometry obstime
    == planet orbital-state obstime
```

All three are compared as **instants**, never as Julian day numbers: two
objects reached through different time scales sit at the same instant while
their JDs differ by up to about 89 seconds, so a comparison on the numbers
would reject a matched pair and accept a mismatched one that happened to
share a number. `same_instant` is the one rule, and
`INSTANT_MATCH_TOLERANCE_DAYS` is the one tolerance — defined in
`physics/epoch.py` and re-exported by `coordinates/astrometry.py`, because
two tolerances would eventually disagree about the same pair.

All three permutations of one-clock-out-of-step are refused, not just the
pair the checker happened to be handed.

### The misuse-proof entry point

`SystemSlice` owns the clock, so it owns the pairing:

```python
system.absolute_planet_position(record, time_jd)
system.planet_to_star_distance(record, time_jd, other_system)
```

Both derive every instant they need from one `time_jd` internally, so there
is no argument a caller can pair wrongly. `obstime()` is the single
conversion point from the explorer's Julian date to the Astropy `Time`, and
`timed_state()` is where the orbit picks up the same one.

## Time semantics

`astropy_time(jd, scale)` in `physics/epoch.py` is that conversion point. It
takes a full Julian date — a caller holding a mission-offset date must have
gone through `Epoch.canonical_jd` first — and the `TimeScale` it was
published in. An unstated scale is read as TDB, stated rather than hidden:
the residual is at most ~569 s, which is 1.8 × 10⁻⁵ yr and under a
microarcsecond of proper motion, while the same 569 s remains fatal to a
transit ephemeris and keeps travelling on the `TimeScale`.

`orbital_time_jd(time, scale)` is the conversion point in the other
direction, and it is the one C3.6 needs so the stellar side and the orbital
side land on the same *instant* rather than the same *number*. The orbital
propagator runs on the project's canonical full-JD axis, so the instant is
expressed in **the scale the orbital epoch was published in** — the
difference `phase_at` then takes, `t - t0`, is between two dates on one axis
rather than between a TDB date and a UTC one.

The two functions are exact inverses. What they exist to prevent is call
sites independently reaching for `time.jd`, `time.tdb.jd`, `time.tcb.jd` or
`time.utc.jd`: those differ by up to about 89 seconds, all of it invisible
in a rendered orbit and none of it invisible in a transit ephemeris.

A precise target time does **not** make an imprecise reference one known. If
the published epoch's scale is unstated it stays unstated, and
`TimeScale.uncertainty_seconds_at` keeps carrying the ~569 s it owes.

Nothing takes a bare `time_jd: float` across an astrometric boundary. A
float has no scale, and three consumers could read one three ways.

## The `UNSPECIFIED` node-convention hardening

Shipped in the same slice because C3.6 is where a provider starts ingesting
real astrometric values. A node with a published value and no recorded
convention is a number, not a direction on the sky. Feeding it to the
rotation would silently assert the standard convention — the failure mode
that is right most of the time, and therefore the worst one to have.

The refusal is in two places on purpose:

* `resolve_node_azimuth` uses the documented normalisation instead of the
  published value, so nothing can reach `R_z(Ω)` with an angle whose meaning
  is unknown;
* `OrbitalElements.for_display` relabels the element
  `ASSUMED_FOR_VISUALIZATION`, so the overlay draws it dashed and captions
  it as a normalisation. Leaving it `MEASURED` would put a normalised line
  of nodes on screen with a caption calling it an observation.

Display normalisation may still assume the convention explicitly: it wrote
the number itself, and `Ω_PA = 0` means North.

## What changed at the API boundary

| C3.5 | C3.6 |
|---|---|
| `absolute_position_blockers(host, node, *, epoch_resolved=False)` | `absolute_position_blockers(host, node, *, astrometry=None)` |
| `absolute_planet_position(..., epoch_resolved=False)` | `absolute_planet_position(..., astrometry=None)` |
| `planet_to_star_distance(..., epoch_resolved=False)` | `planet_to_star_distance(..., astrometry=None, other_astrometry=None)` |
| `EPOCH_NOT_MODELLED` | `ASTROMETRY_NOT_PROPAGATED` |
| planet offset: a bare `(3,)` AU array | `TimedOrbitalState` for anything published |
| orientation checked: node only | all three Euler angles, via `absolute_orientation_blockers` |

The default is still closed, and closing it still takes no argument. What
changed is that **opening it now requires a state that had to be
propagated**.

## The absolute-orientation gate

The node gates are necessary and **not sufficient**, and until the second
local audit that gap was open: a planet whose inclination and argument of
periapsis were normalised for display reached a published ICRS coordinate on
the strength of a tagged node alone. The number looked entirely ordinary.

A unique physical orientation needs all three Euler angles to be
observations, so `absolute_orientation_blockers(elements)` checks all three:

| blocker | fires when |
|---|---|
| `INCLINATION_UNRESOLVED` | the plane's tilt is a display normalisation, so there is no plane in space |
| `PERIAPSIS_DIRECTION_UNRESOLVED` | ω was never published, or was filled in by `for_display` |
| `PERIASTRON_CONVENTION_UNSTATED` | ω is a real number under an unstated convention, so periapsis may be the star's reflex direction and 180° away |
| `NODE_CONVENTION_UNSTATED` | the node is a number, not a direction on the sky |
| `NODE_SENSE_UNRESOLVED` | the node is known only modulo 180° |

Every failing gate is reported, so a row blocked for four reasons says four.

### Three concrete failure modes it closes

**A transit epoch with an assumed ω.** At transit the argument of latitude
is π/2, so ν_transit = π/2 − ω. For an eccentric orbit the mapping
ν → E → M is nonlinear in ω, so advancing M(t) = M_transit + n(t − t_transit)
does not give a unique physical position without the real ω. The *timing* is
a genuine observation and the time gate accepts it; the orientation gate is
a separate question and refuses.

**A periastron epoch with no ω.** A periastron epoch gives M = 0 exactly and
needs no ω at all, which is why the phase model correctly calls it
`CONSTRAINED`. An absolute 3D position still has to know which direction
periapsis points *within* the orbital plane, and that is a different
measurement.

**A missing inclination.** A perfect node and a perfect temporal phase do
not locate the orbital plane in 3D if `i` was merely normalised for display.

### Not merged into `PhaseStatus`

Phase and orientation are separate epistemic dimensions in this codebase and
stay that way. A transit epoch is a real observation of *when* and says
nothing about *which way*, so the two gates are checked side by side and
reported side by side. `TimedOrbitalState` carries both — the elements and
the phase — and asks each its own question.

That is also why `TimedOrbitalState.elements` is a required field rather
than an optional one: a state vector alone cannot be interrogated about how
well its orientation is known, and that question decides whether an absolute
position may be published at all.

### The circular-orbit case is conservatively blocked

For `e = 0` there is no periapsis, so ω is not a physical degree of freedom
and demanding it looks like an over-refusal. It is a deliberate one, and
`CIRCULAR_ORBIT_PERIAPSIS_NOTE` records why: the exemption is not a property
of the orientation alone.

* the position depends on the argument of latitude `u = ω + ν`;
* with a **transit or conjunction anchor**, ν_t = π/2 − ω at the anchor, so
  `u(t) = π/2 + n(t − t_t)` and ω cancels exactly. Such an orbit genuinely
  does not need it;
* with a **periastron anchor**, or a mean anomaly quoted at an epoch, the
  anchor is measured *from* periapsis — which does not exist at `e = 0` — so
  the epoch itself has no meaning and nothing is recovered.

So the correct rule needs the phase anchor kind, which would fold one
epistemic dimension into the other. No catalogue row in the snapshot can
reach this case — every real exoplanet is already blocked on the node — so
the conservative answer ships and the analysis stays written down for
whoever lifts it. A display-normalised `e = 0` never qualifies in any case:
it is `ASSUMED_FOR_VISUALIZATION`, and the gate reads status, not presence.

## What this does and does not unlock

Every Gaia-backed host in the snapshot now has **explicit astrometric epoch
knowledge**. That is not the same claim as "the epoch gate is open for every
Gaia-backed host", and the difference matters to a future caller who might
otherwise treat all cached rows as equally publishable:

| tier | eligible for |
|---|---|
| `REFERENCE_EPOCH_ONLY` | an exact position at the reference epoch only |
| `DIRECTION_ONLY` | an approximate direction; **no** strict cross-epoch 3D publication |
| `FULL_SPACE_MOTION` | strict cross-epoch 3D propagation |

Five of the six committed hosts reach `FULL_SPACE_MOTION`. TRAPPIST-1 reaches
`DIRECTION_ONLY` and is blocked, correctly, by a missing measurement rather
than by a missing feature.

HD 80606 b's absolute position is still withheld, and the second audit
changed how many reasons it gives. It has a published argument of periastron
under an unstated convention, which nothing checked before:

```
Absolute position: unknown [ICRS] - the argument of periastron was
published under an unstated convention, so periapsis may be the star's
reflex direction and 180 degrees from the planet's; also, the
ascending/descending sense of the node is not resolved, so the orientation
is known only modulo 180 degrees
```

Kepler-11's planets say something different again — no argument of periapsis
at all — and TRAPPIST-1 adds four more, including its missing radial
velocity. Each row now names the science it is actually short of.

That is the point of the slice. A gate that opened when the science
supported it is what makes the remaining refusals statements about the
catalogue rather than placeholders that never moved.

## Where it lives

| module | role |
|---|---|
| `coordinates/astrometry.py` | `AstrometricState`, `PropagatedAstrometry`, `MotionKnowledge`, `propagate_astrometry` |
| `physics/timed_state.py` | `TimedOrbitalState`, `same_instant` — the planet's half of the common-time rule, and its orientation |
| `physics/orbital_semantics.py` | `absolute_orientation_blockers` — all three Euler angles |
| `physics/node_semantics.py` | `node_publication_blockers` — the node's share of it |
| `data/gaia.py` | Gaia DR3 provider, identity guards, validation, atomic cache, offline `GaiaHostIndex` |
| `physics/epoch.py` | `astropy_time` / `orbital_time_jd` — the two clock conversion points, and the one tolerance |
| `scripts/sync_gaia_astrometry.py` | the online refresh; never called by the application |
| `data/reference_systems/gaia_dr3_astrometry.json` | the committed validated cache |
| `data/reference_systems/gaia_dr3_hosts.json` | the committed NASA host → Gaia source cross-match |
| `tests/regression/test_explorer_c36.py` | the slice's specification |
