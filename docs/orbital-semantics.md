# Orbital metadata semantics

The pass required by the review's section 14, before scaling to many
systems. The numerics were settled by the previous milestone; the remaining
risk was that a **numerically perfect transform can still be wrong** because
the catalogue value did not mean what the code assumed it meant.

## The argument of periastron

`pl_orblper` generally preserves the convention of the source publication.
Radial-velocity papers habitually report the argument of periastron of the
**star's reflex orbit**; transit and astrometry papers report the
**planet's**. They differ by exactly 180 degrees:

```
omega_planet = (omega_star + pi) mod 2 pi
```

On HD 80606 b, reading it the wrong way puts periapsis on the opposite side
of the star - a displacement of `2 a (1 - e)` = 0.0628 AU, asserted in
`test_a_stellar_reflex_reading_would_move_periapsis_across_the_star`.

### The model

`PeriastronConvention` records which orbit the value describes:

| Value | Meaning | `argument_of_periapsis_planet` |
|---|---|---|
| `PLANET` | explicitly the planet's orbit | the raw value, `MEASURED` |
| `STELLAR_REFLEX` | the host star's reflex orbit | raw + 180 deg, `DERIVED` |
| `AS_REPORTED` | taken from the catalogue, convention unstated | the raw value, `ASSUMED_FOR_VISUALIZATION` |
| `UNKNOWN` | no angle published | `UNKNOWN` |

The default for the NASA archive is **`AS_REPORTED`**, because that is what
a catalogue without a convention column actually tells us. Claiming
`PLANET` would be inventing metadata.

`OrbitalElements.argument_of_periastron` always holds the raw catalogue
value, unmodified. `argument_of_periapsis_planet` derives the planet-frame
angle from it, and the *status* of that derived parameter carries how much
was really known. The renderer sees the resolved value; the report shows
both:

```
  Arg. periapsis w:  -58.89 deg  (raw, as catalogued)
    convention:      as reported; convention not stated
    used for planet: 301.1 deg  [assumed for visualisation]
    caveat:          The source convention is unstated. If the publication
                     reported the stellar reflex orbit, periapsis is
                     oriented 180 degrees away from the truth.
```

## Orbit validity

`OrbitValidity` is a flag set, not a single state, because the four
questions are independent - and the normal condition for an exoplanet is a
well-determined shape, a partly known orientation and no usable phase:

| Flag | Requires |
|---|---|
| `GEOMETRY_VALID` | `a` and `e` are measured or derived |
| `PHASE_VALID` | an epoch **and** a period |
| `ORIENTATION_PARTIAL` | at least one of `i`, `omega`, `Omega` |
| `ORIENTATION_FULL` | all three **and** a stated `omega` convention |

That last condition matters: three known angles under an unstated
convention still leave periapsis ambiguous by 180 degrees, so HD 80606 b -
which has `i` and `omega` but no `Omega` and no stated convention - reports
`GEOMETRY_VALID | PHASE_VALID | ORIENTATION_PARTIAL`.

## Epochs and time systems

An epoch is not just a number of days. `TimeScale` records which Julian-date
flavour it is:

| Scale | Offset to JD | Ambiguity if assumed BJD_TDB |
|---|---|---|
| `BJD_TDB` | 0 | 0 s |
| `HJD_UTC` | 0 | up to ~77 s |
| `JD_UTC` | 0 | up to ~568 s |
| `BKJD` (Kepler) | +2454833.0 | 0 s |
| `BTJD` (TESS) | +2457000.0 | 0 s |
| `JD_UNSPECIFIED` | 0 | up to ~568 s |

The ambiguities are of very different sizes and must be kept apart — one
constant for "not barycentric" would be wrong for both cases it covered:

* **heliocentric vs barycentric** is only ~8 s. It is the Sun's own motion
  about the solar-system barycentre, up to 1.6 million km, which is about
  five light-seconds. An HJD has *already* removed the Earth's orbital
  light time, so this small residual is all it still owes;
* **geocentric vs barycentric** is ~499 s — one AU of light travel, the
  familiar "8 minutes". This is the Earth's orbital displacement, and it is
  what a plain, unlabelled JD carries. It is roughly sixty times the
  heliocentric term, so `HJD_UTC` and `JD_UTC` cannot share a bound;
* **TDB vs UTC** is `(TT - UTC) + (TDB - TT)`, currently about 69.184 s;
* **mission offsets** are 13 and 18 *years*. Ignoring one is not a rounding
  error, which is why they get a type rather than a comment.

`JD_UNSPECIFIED` takes the `JD_UTC` bound, not the `HJD_UTC` one: a plain
geocentric JD is a live reading of an unlabelled archive date, so the
conservative choice is the larger of the two.

### TDB - UTC is a function of the date, not a constant

The bounds in the table are quoted for the present era. The leap-second
term is not a constant of nature: it steps whenever IERS announces one, and
`TDB - TT` carries a periodic relativistic term of order a millisecond that
is never exactly zero. A literal `69.184` in the propagation path would
freeze a 2026 relationship into the physics and would silently go stale.

So `tdb_minus_utc_seconds(jd)` asks astropy, which owns both the
leap-second table and the `TDB - TT` series. The difference is taken
between the two scales' Julian-day *numbering* — via the `jd1`/`jd2` pair,
so the sub-millisecond term survives — because the two are the same instant
and subtracting them as times would correctly give zero.

`Epoch.scale_uncertainty_seconds` evaluates it at the epoch's own date, so a
1990 epoch is charged the twelve fewer leap seconds that actually applied
then. `TDB_MINUS_UTC_FALLBACK_SECONDS` is used only when there is no date at
all, or when astropy refuses one; a bad date degrades the precision of a
stated uncertainty rather than raising inside a render loop.

The archive publishes no machine-readable scale for `pl_orbtper` or
`pl_tranmid`, so the honest default is `JD_UNSPECIFIED`. Rather than argue
about whether it matters, `Epoch.phase_uncertainty_fraction` states it: for
HD 80606 b it is 6e-8 of a revolution, i.e. irrelevant. On an ultra-short
period planet it would still be small, but it would be stated the same way.

### A mean anomaly needs the date it was quoted at

`M0` alone does not place a planet. The propagation law is

```text
M(t) = M0 + n (t - t0)
```

so without `t0` the published angle constrains the orbit at one unstated
moment and at no other. The catalogue routinely publishes one without the
other, so the two travel together as a `MeanAnomalyAnchor`:

```python
anchor.is_known   # M0 was published
anchor.is_dated   # M0 and t0 were both published - the only usable state
```

`mean_anomaly_at_epoch` is an angle in radians and never occupies an epoch
slot; `epoch_mean_anomaly` is its reference date and does. `Epoch.is_dated`
tests the *unit*, so an angle miscast as an epoch is still refused a Julian
date.

An undated `M0` reports `PhaseKnowledge.REFERENCE_ANOMALY_UNDATED` and
`PhaseProvenance.MEAN_ANOMALY_UNDATED`, yields no mean anomaly, and is
still listed in the orbit-validity block — the measurement is real, it just
cannot be moved to another date. The previous model returned `M0` verbatim
for every requested instant, which is a correct position at exactly the
moment nobody published and wrong at all the others, while reporting itself
as observationally anchored.

`EpochKind.TRANSIT.needs_argument_of_periapsis` is `True`, because the true
anomaly at mid-transit is `pi/2 - omega` - so a transit epoch inherits the
convention ambiguity, while a periastron epoch does not.

## References

Planetary and stellar parameters can come from different papers even within
one default solution, so `pl_refname` and `st_refname` are kept separately.
The archive wraps them in an HTML anchor; `clean_reference` extracts the
citation and `reference_url` the ADS link, and the raw markup never reaches
a caller.

## Physical radius vs display radius

Review section 6. `display_radius` is strictly a rendering parameter and must
never enter gravitational, transit-depth, density, collision,
orbital-distance or stellar-radius calculations.

This is enforced rather than documented: `physics`, `data`, `spectroscopy`,
`classification` and `coordinates` may not contain the symbols
`display_radius`, `radius_display` or `DisplayScale` at all, and a test
asserts that changing the exaggeration leaves every orbital position bit-for-bit
identical.

## What this pass changed, and what it did not

It changed no numbers. HD 80606 b's rendered orbit is identical, because
`AS_REPORTED` resolves to the same direction as the raw value - the point
was never to move the planet, but to stop the code from *claiming* it knew
where periapsis was.

| Review section 14 task | Where |
|---|---|
| preserve raw omega | `OrbitalElements.argument_of_periastron` |
| store periastron convention | `PeriastronConvention` |
| preserve publication/reference | `clean_reference`, `reference_url`, `OrbitalElements.reference` |
| add `pl_orbtper` | `epoch_periastron`, `EpochKind.PERIASTRON` |
| preserve time-system metadata | `TimeScale`, `Epoch` |
| geometry-valid vs phase-valid | `OrbitValidity` |
| partial vs full orientation | `OrbitValidity`, gated on a stated convention |
| test the 180-degree conversion | `test_stellar_reflex_converts_to_planet_by_exactly_pi` |
| preserve unknown Omega | unchanged from the previous milestone; still tested |
| keep assumptions explicit | `describe_orbit`, dashed orbit rendering |
