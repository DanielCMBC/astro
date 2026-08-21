# Physics

Every constant comes from `astropy.constants` and every quantity carries an
`astropy.units` unit until the rendering boundary (roadmap sections 3.8 and
3.9). `physics/constants.py` is the only module allowed to define a
constant, and even there Wien's `b` is derived from `h`, `c` and `k_B`
rather than quoted.

## Kepler's equation

`physics/kepler.py` solves

```
M = E - e sin E
```

with a **safeguarded Newton-Raphson** iteration, vectorised over NumPy
arrays.

The 3D prototype used the first-order approximation `E ~ M + e sin M`. That
is accurate for near-circular orbits and badly wrong otherwise: at `e = 0.9`
it is off by more than 0.3 radians. `tests/physics/test_kepler.py` asserts
both that the real solver converges and that the approximation does not.

The solver is safeguarded because plain Newton can overshoot near periapsis
of a very eccentric orbit. A bracket `[0, pi]` is maintained on the folded
half-orbit; any Newton step leaving the bracket is replaced by a bisection.
Because `f(E) = E - e sin E - M` is monotonically increasing for `e < 1`,
the bracket always contains the root and always halves, so convergence is
guaranteed. Measured residuals stay below `1e-13` radians for every
eccentricity up to `0.9999`.

Hyperbolic and parabolic orbits raise `ValueError`. They are a different
equation, and the original code's `np.clip(e, 0.0, 0.98)` silently changed
the orbit instead.

## Orbital elements

`OrbitalElements` stores each element as a `Parameter`, so the following are
distinguishable states rather than the same number:

* published eccentricity of 0.0 (`MEASURED`);
* no published eccentricity (`UNKNOWN`);
* a circle drawn because something had to be drawn
  (`ASSUMED_FOR_VISUALIZATION`, produced only by `for_display()`).

Three knowledge states are tracked separately, because exoplanets routinely
have a well-measured shape, a partly measured orientation and no usable
phase at all:

* `shape_known` - `a` and `e` are available;
* `orientation_known` - `i`, `omega` **and** `Omega` are all measured. This
  is almost always `False`: the longitude of the ascending node is not
  observable from transits or radial velocity, and the NASA archive does not
  publish it;
* `phase_knowledge` - one of `ORBIT_SHAPE_KNOWN`, `ORBIT_PHASE_CONSTRAINED`,
  `CURRENT_POSITION_COMPUTABLE`, `DISPLAY_PHASE_ASSUMED`.

### Geometry

The 3D transform lives in its own module, `physics/orientation.py`, which
imports nothing but NumPy - no units, no provenance - so it can be tested
purely numerically.

Perifocal position (reference section 4):

```
x = a (cos E - e)
y = a sqrt(1 - e^2) sin E
z = 0
```

rotated into the reference frame by `R = Rz(Omega) Rx(i) Rz(omega)`
(reference section 8). The rotations apply right to left: `omega` acts
first, inside the orbital plane; then the plane is tilted by `i`; then the
whole configuration is spun about the reference pole by `Omega`. Any other
order gives a different, wrong orbit, and a test asserts that swapping two
of them changes the result.

`tests/physics/test_orientation.py` checks each angle separately and then
cross-checks the composed matrix against the **expanded scalar equations**
of reference section 11, written out independently in the test file:

| Case | Assertion |
|---|---|
| `i = 0` | `max\|z\| < 1e-12` for any `omega`, `Omega` |
| `i = 90 deg` | orbit lies in XZ; the normal is `[0, -1, 0]` |
| any `i` | the normal tilts from the pole by exactly `i` |
| `omega = 90 deg` | periapsis moves from `+x` to `+y` |
| any `omega` | the node-to-periapsis angle equals `omega` |
| `Omega = 90 deg` | the normal rotates about `z`, its tilt unchanged |
| any `Omega` | the orbit's ascending `z = 0` crossing is at `Omega` |
| combined | matches the expanded equations to 1e-13 |

Rotations also preserve orbital radius, keep the orbit planar, and leave the
normal independent of `omega` - all asserted, because the prototype's orbits
were `[x, y, 0.0]` and every system was coplanar.

### Velocity and the conserved quantities

`physics/state_vectors.py` adds velocity (reference section 15):

```
v_p = (n a / (1 - e cos E)) * [-sin E, sqrt(1-e^2) cos E, 0],  n = sqrt(mu/a^3)
```

rotated by the same matrix as the position. `mu = G(M* + Mp)` comes from
`gravitational_parameter`, which returns `None` when the stellar mass is
unpublished - so an orbit with no host mass simply has no velocity, rather
than a guessed one.

Two conservation tests do the work a shape test cannot:

**Kepler's second law.** 100 equal time steps around the orbit must sweep
equal areas. Equal steps in mean anomaly *are* equal steps in time, so this
tests the propagator the animation actually uses. Areas are computed from
the 3D cross product, which stays correct for an inclined orbit where a
projected area would shrink with the tilt.

The chord between two positions cuts the corner of a curved sector, so the
polygonal estimate under-reads by `O(dnu^2)` - at `e = 0.99` a hundred-step
polygon under-reads the periapsis sector by nearly 40%. Each interval is
therefore subdivided, and a separate test asserts the residual falls by ~16x
when the substeps quadruple. That second-order convergence is what
distinguishes "the polygon is coarse" from "the orbit is wrong".

**Specific orbital energy** (reference section 17):

```
epsilon = v^2/2 - mu/r  ==  -mu/(2a)
```

This tests position and velocity *together*, at many phases, for
eccentricities up to 0.99. On HD 80606 b the relative error is 1.9e-15. It
is also the state an eventual N-body handoff needs.

## Kepler's third law

```
a = ( G (M* + Mp) P^2 / 4 pi^2 )^(1/3)
P = 2 pi sqrt( a^3 / (G (M* + Mp)) )
```

Used three ways:

1. to derive a missing semimajor axis (marked `DERIVED`, never `MEASURED`);
2. to derive a missing period;
3. as a consistency check. `kepler_third_law_residual` returns
   `(a_published - a_derived) / a_derived`, shown in the overview panel as an
   educational diagnostic (roadmap section 8.7).

Fractional uncertainty propagates as `da/a = (2/3) dP/P + (1/3) dM/M`.

## Stellar physics

```
L = 4 pi R^2 sigma T^4          -> luminosity_from_radius_and_teff (DERIVED)
M_bol = 4.74 - 2.5 log10(L/Lsun)
T_eq = ((1-A) L / (16 pi f sigma a^2))^(1/4)
```

`equilibrium_temperature` records its albedo and redistribution assumptions
in the parameter's `note`, because both are modelling choices.

The habitable zone uses the Kopparapu et al. (2013) runaway-greenhouse and
maximum-greenhouse coefficients, and returns `UNKNOWN` outside the fitted
range of 2600-7200 K rather than extrapolating a polynomial into a regime it
was never fitted for.

### The two diagrams

The original program plotted effective temperature against stellar *radius*
and called it an HR diagram. Both plots are kept, each with its correct name
(roadmap section 3.6):

* `DiagramKind.HR_DIAGRAM` - luminosity against `T_eff`;
* `DiagramKind.TEMPERATURE_RADIUS` - radius against `T_eff`.

## Radiation

```
B_lambda(T) = 2 h c^2 / lambda^5 * 1 / (exp(hc / lambda k T) - 1)
lambda_max  = b / T
```

`np.expm1` is used for the denominator so the long-wavelength Rayleigh-Jeans
tail does not lose precision to cancellation. The steradian is made explicit
in the returned unit so radiance cannot be silently mixed with flux.

Every curve is tagged `SpectrumModel.IDEAL_BLACKBODY` and carries the caveat
that a real stellar spectrum contains absorption lines and
atmosphere-dependent structure (roadmap section 3.7). A test integrates
`B_lambda` over wavelength and the hemisphere and checks it reproduces
`sigma T^4` to one part in a thousand.
