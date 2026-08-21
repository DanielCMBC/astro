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
| `HJD_UTC` | 0 | up to 549 s |
| `BKJD` (Kepler) | +2454833.0 | 0 s |
| `BTJD` (TESS) | +2457000.0 | 0 s |
| `JD_UNSPECIFIED` | 0 | up to 549 s |

The two ambiguities are of very different sizes and worth separating:

* **barycentric vs heliocentric** is up to ~480 s, because the Sun moves up
  to 1.6 million km from the solar-system barycentre;
* **TDB vs UTC** is the accumulated leap seconds, 69.184 s;
* **mission offsets** are 13 and 18 *years*. Ignoring one is not a rounding
  error, which is why they get a type rather than a comment.

The archive publishes no machine-readable scale for `pl_orbtper` or
`pl_tranmid`, so the honest default is `JD_UNSPECIFIED`. Rather than argue
about whether it matters, `Epoch.phase_uncertainty_fraction` states it: for
HD 80606 b it is 6e-8 of a revolution, i.e. irrelevant. On an ultra-short
period planet it would still be small, but it would be stated the same way.

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
