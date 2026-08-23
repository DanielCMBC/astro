# Explorer C: scientific overlays

Explorer A could fly and select; Explorer B could say what was selected and
when. Explorer C draws things that are not bodies — a region the physics
layer computed, and the orientation of an orbit in space.

Both raise the same question in different forms: **can the 3D view show a
scientific result without owning it, and without overstating it?**

| | overlay | the risk it carries |
|---|---|---|
| C1 | habitable zone | a band that looks like a claim about habitability |
| C2 | orbital orientation | a line that looks like a measured direction on the sky |

---

# C1 — the habitable-zone overlay

## One model, two consumers

`habitable_zone_au` lives in `physics/stellar.py` and is reached only
through `StarRecord.habitable_zone`. The info panel and the 3D overlay both
read *that object*. Neither evaluates the Kopparapu polynomial itself, so
there is no second implementation to drift out of step, and
`test_the_overlay_and_the_panel_agree` pins that the numbers on screen and
the numbers in the panel are the same numbers.

```
stellar physics  ->  StarRecord.habitable_zone  ->  scene builder
                                                        |
                                              RenderZone (finished rings)
                                                        |
                                                     OpenGL
```

`RenderZone` carries two rings of geometry, a fill colour and an edge
colour. It has no luminosity, no effective temperature, no Kopparapu
coefficients, no AU boundary, no model name and no provenance — so the
renderer has nothing it could recompute a boundary from. The band between
the rings is triangulated on the CPU for the same reason: the GPU is handed
finished vertices and has no say in where a boundary sits.

## It is a cross-section, and it says so

The physical habitable zone is a range of *radial distances* from the star,
so the region is a spherical shell. The overlay is a flat annulus on the
system reference plane — a **cross-section** of that shell:

> Habitable-zone cross-section 0.965–1.7 AU: radial irradiation boundaries
> shown in the system reference plane. The physical region is a spherical
> shell around the star.

A translucent sphere would be more literal and less readable. The
cross-section is kept, and the annotation carries what the picture leaves
out. The annulus is **never** rotated into a selected planet's orbital
plane: it represents distance from the star, not a property of any plane,
and a test asserts that selecting a planet does not move it.

The two boundaries are drawn as lines in `edge_color`, not left as the edge
of a fading wash. The boundaries are where the model's statement actually
lies — the fill is only what lies between them — so a band with no visible
limit would understate how sharp the statement is.

## Unknown is drawn as nothing

The Kopparapu coefficients are fitted for 2600 K ≤ T_eff ≤ 7200 K. Outside
that, `habitable_zone_au` returns UNKNOWN bounds rather than extrapolating,
and an unknown zone becomes no geometry at all:

```
model input outside the fitted range
        -> HZ boundaries UNKNOWN
        -> no RenderZone
        -> an annotation saying why
```

Not clamped to the model limit, not substituted with a Solar value, not
drawn as a generic ring. This is not a rare path: **TRAPPIST-1**, whose
planets are routinely described as being in the habitable zone, is cooler
than the fitted floor, so this project draws no zone for it and says so.

## What the zone does not say

Every scene that draws one states it:

> The habitable zone is a stellar-irradiation model, not a claim about
> habitability: it says where an Earth-like atmosphere could support liquid
> surface water, not that any planet drawn inside it does.

The model establishes nothing about atmosphere, surface liquid water,
pressure, albedo, stellar activity tolerance, volatile retention,
geological state or biosphere, so the overlay must never imply those.

## Reading a planet against the zone

Wording matters here more than it looks. **Kepler-11 g** lies starward of
the 1.007 AU inner HZ boundary, so it is outside this irradiation-defined
habitable zone on the hot side — and all six Kepler-11 planets do, which is
exactly the kind of thing the overlay exists to make visible.

Saying instead that a planet is "inside the inner edge" invites the reading
"inside the habitable zone", which is the opposite of what it means. Prefer
*starward of* or *interior to* the inner boundary.

## Boundary naming

`HabitableZone` names its model — Kopparapu et al. (2013), runaway
greenhouse and maximum greenhouse — and its validity range explicitly, so
the current generic `inner` and `outer` fields are unambiguous. Before a
second prescription is ever supported, the boundary kinds must become
explicit (`RUNAWAY_GREENHOUSE`, `MAXIMUM_GREENHOUSE`, `RECENT_VENUS`,
`EARLY_MARS`) rather than silently changing what `inner` means.

---

# C2 — the orbital-orientation overlay

## The problem is not geometry

Drawing a plane, a normal, a node line and a periapsis arrow is
straightforward. The difficulty is that **a line drawn from an element
nobody measured looks exactly like a line drawn from one somebody did.**

Almost no exoplanet has a measured longitude of the ascending node: it is
not observable from transits or radial velocity. Every planet in the
committed snapshot has `Omega = UNKNOWN`, normalised to zero for display.
That normalisation is correct and necessary. It is also the most
misreadable object in the overlay, because a line of nodes at `Omega = 0`
is a direction on the sky — and there is no such direction to look at.

## What is drawn

For the **selected** planet, not for every planet at once:

| guide | defined by | drawn as |
|---|---|---|
| system reference plane | the frame | a ring, always |
| orbital plane | `i`, `Omega` | a ring |
| orbit normal | `i`, `Omega` | an arrow |
| line of nodes | `Omega` | a line through the star |
| periapsis direction | `omega`, `i`, `Omega` | an arrow ending *on* the orbit |
| inclination indicator | `i`, `Omega` | an arc swept about the nodes |

Six sets of planes and node lines at once would be unreadable, and the
question the overlay answers — *how is this orbit oriented* — is asked about
one planet at a time. Selection drives it: `Explorer.scene()` passes the
current selection's entity id, and changing the selection changes the
guides without touching a single orbit point.

## Provenance decides what may be drawn

```
MEASURED                   -> solid
DERIVED                    -> solid, labelled derived
ASSUMED_FOR_VISUALIZATION  -> dashed, and said in words
UNKNOWN                    -> not drawn at all, unless normalisation is asked for
```

`show_normalised` is off by default. With it on, a normalised guide is
drawn **dashed**, never solid, and always with:

> Ascending node: unknown. Display normalisation Omega = 0 deg. The
> absolute rotation of this orbit about the line of sight is unconstrained.

Two distinctions inside that table are worth stating plainly:

* **DERIVED is solid.** An argument of periastron converted from the host
  star's reflex orbit by 180° is a real orientation reached by a stated
  transform from a stated convention. It is labelled derived — but it is
  not a guess, and dashing it would say it was.
* **A measured element drawn at a normalised azimuth is dashed, not
  withheld.** A transiting planet's inclination is a real measurement even
  though the direction it is tilted *towards* is not, so the plane is drawn
  — dashed, with the caveat that it may be rotated about the line of sight
  from the true one.

Provenance is never carried by colour alone. Colour separates one guide
from another; the **stroke** carries the science, and the annotations carry
it in words. A greyscale print or a colour-blind viewer loses nothing.

## `AS_REPORTED` is the normal case

The NASA archive preserves each source publication's convention for
`pl_orblper` and carries no machine-readable column saying which it is.
Radial-velocity papers habitually report the *star's* reflex orbit; transit
papers report the *planet's*. The two differ by exactly 180°.

So HD 80606 b — a real number, an unstated convention — draws its periapsis
arrow dashed, with:

> The source convention is unstated. If the publication reported the stellar
> reflex orbit, periapsis is oriented 180 degrees away from the truth.

And three published angles under `AS_REPORTED` still cannot reach
`ORIENTATION_FULL`: periapsis remains ambiguous by half a turn however many
numbers were printed.

## One reading of the angles, not two

Every guide is built from `physics/orientation.py` — the same
`R = R_z(Omega) R_x(i) R_z(omega)` the propagator uses:

```
r = R_z(Omega) R_x(i) R_z(omega) r_perifocal
```

A guide built from a second, independent reading of those three angles
would look perfectly plausible while disagreeing with the orbit it
annotates, and would disagree *silently* — both would be smooth curves in
roughly the right place. So the visualization layer never reinterprets a
catalogue convention: `orientation_guides()` reads the resolved elements and
calls the production transform, and
`test_the_guides_use_the_production_orbital_transform` pins the result
against `rotation_perifocal_to_inertial` directly. The periapsis arrow ends
at the position the propagator gives for `E = 0`, on the orbit.

## The render contract

`RenderGuide` is the C2 primitive: an identifier, a polyline, a stroke
style, a colour and a label.

```python
RenderGuide(
    identifier="planet:ascending-node",
    points_local=...,   # (N, 3) float32, already in the frame's units
    style=GuideStyle.DASHED,
    color=...,
    label="line of nodes",
)
```

There is no inclination, no `omega`, no `Omega`, no eccentricity, no status
and no convention — for the same reason `RenderZone` carries no luminosity.
`GuideStyle` has exactly two values, `SOLID` and `DASHED`, both purely
visual: *why* a guide is dashed is words, and words belong to the UI. Every
guide in a scene is one batched indexed `LINES` draw, sharing the orbit
program.

## Seeing all three cases

```bash
python -m astro_explorer.app.orientation_demo --out renders
```

renders `MEASURED`, `DERIVED` and `ASSUMED` side by side, and CI renders
them on software Mesa on every push — because the difference between them
is a dash pattern, which is exactly the kind of thing that can be right in a
scene description and lost on the way to the GPU.

The first two use **constructed element sets**, labelled as such on the
frame: no exoplanet in the snapshot has a measured longitude of the
ascending node, so there is no real system that can demonstrate what a fully
determined orientation looks like. The third is HD 80606 b, and it is what
almost every planet in the archive looks like.

---

# C3 — the coordinate and distance inspector

C1 and C2 asked whether the scene could *draw* derived science honestly. C3
asks whether the explorer can put a **number** next to it.

That is the easier thing to get quietly wrong. A drawn overlay is obviously
a picture. A figure reading `0.2057 AU` is read as a measurement, and
nothing about its appearance says whether it came from the float64
propagator or from a float32 vertex that had already been scaled so the
planet would be visible beside its star.

So C3 has exactly one rule:

> **Every scientific distance and coordinate comes from float64 scientific
> state, and never from display geometry.**

## What the renderer does to a number

Each of these is correct for drawing and fatal for measuring:

| the renderer | what it does to a position |
|---|---|
| float32 vertices | loses an AU-sized offset at parsec scale |
| radius exaggeration | changes apparent size by large factors |
| level of detail | swaps the mesh under the body |
| camera transforms | moves everything, every frame |

A distance read back out of the scene would therefore change when the viewer
zoomed — which is precisely the sort of quantity that looks authoritative
and is not.

`coordinates/inspector.py` imports no rendering module, and
`test_explorer_c3.py` builds the scene both ways and asserts the inspector
did not move:

```
exaggerate=False / exaggerate=True   -> identical inspector values
LOD 0 .. 5                            -> identical inspector values
camera anywhere                       -> identical separations
```

The scene really does differ in each case — the tests assert that too, so
an invariance check cannot pass by comparing two identical scenes.

## Three row types, and why the second one is strict

```python
InspectorRow("Distance from Sun", parameter)          # a scalar
CoordinateRow("Cartesian position", xyz, u.pc, ICRS)  # a triplet
NoteRow("Phase provenance", "constrained by ...")     # a qualifier
```

`CoordinateRow` **cannot be constructed without naming its frame**. A bare
`x/y/z` is meaningless: the same planet is at three entirely different
coordinates in ICRS, in Galactic and in its own system frame. Omitting the
frame is a `ValueError`, not a display quirk noticed later.

Non-finite components are rejected for the same reason. An unknown position
is `None` and reads as UNKNOWN; it is never NaN dressed as a number.

`NoteRow` carries the **phase provenance**, and it sits in the row list
rather than in prose on purpose. `Distance from host: 0.2057 AU` means
something quite different depending on whether the planet's position at this
instant is fixed by a published epoch or is being advanced from an arbitrary
zero — in the second case the number describes the motion, not tonight. A
panel that rendered the distances must not be able to drop the sentence that
says how to read them.

## The frame layer is Astropy's

There is no second hand-written RA/Dec engine. `SkyCoord` does the ICRS ↔
Galactic work, and the round trip is asserted to machine precision.

Galactic `l` and `b` are available even when the parallax is unusable: a
direction needs no distance. It is the *radial* coordinate that is missing,
not the pointing.

## Distances come from the propagated state

```python
r = np.linalg.norm(planet_position_au)   # the float64 physics vector
```

The identity `r = a(1 - e cos E)` holds, and the tests check that it holds —
but the inspector does not use it. Recomputing would create a second opinion
about the orbit, free to drift from the one actually drawn. **One
propagation, one answer.**

HD 80606 b is the reason this matters: at `e = 0.93` the instantaneous
distance sweeps a factor of ~28 across one period, so a UI that showed the
semimajor axis as "the distance" would be wrong by that factor for most of
the orbit.

## An assumption does not become a measurement

`a(1-e)` is exact — which is the trap. The arithmetic being sound says
nothing about whether `a` was measured:

| inputs | periapsis status |
|---|---|
| `a`, `e` measured | `DERIVED` |
| either assumed for visualisation | `ASSUMED_FOR_VISUALIZATION` |
| either unknown | `UNKNOWN` |

A value invented so a picture could be drawn must not emerge as a quotable
orbital distance.

## A detached system keeps its orbit and claims no address

TRAPPIST-1 has no usable distance in the committed snapshot. It therefore
reports:

```
Distance from Sun     UNKNOWN     (never 0 pc — 0 pc is the Sun)
Cartesian position    UNKNOWN
System-frame position known
Distance from host    known
Absolute position     UNKNOWN
```

The orbit is fine. It is the *address* that is missing. This is the
detached-frame rule from Explorer B, applied to the sky.

## The coordinate C3 refused to publish

The obvious next row is the planet's absolute position:

```
r_planet = r_host + r_planet/local
```

C3 did **not** publish it. The host's Cartesian position is in the **ICRS**
basis; the planet's local vector is in the **system frame**, whose axes come
from the orbital transform. No rotation between those bases existed, so that
sum added components measured along different axes — a **basis error, not an
uncertainty**, which does not become correct when Ω happens to be measured.

At 66 pc an AU-scale offset is numerically tiny, which is exactly the trap:

> A small error is not a correct coordinate. Scale must not be allowed to
> hide a basis mistake.

C3.5 supplies the missing rotation. See below.

## Worked systems

| system | what it exercises |
|---|---|
| HD 80606 b | `e = 0.93`; instantaneous distance against peri/apo |
| Kepler-11 | six planets sharing one host origin |
| TRAPPIST-1 | detached: local coordinates, no absolute distance |
| HD 219134 | mixed data, and a quoted distance uncertainty |

---

# C3.5 — the SystemFrame → ICRS basis

A small coordinate-physics slice, not UI polish, and a prerequisite for C4.
It is almost entirely a module about **conventions**: the arithmetic is four
lines, and the risk is that those four lines encode a *different* convention
from the one the catalogue used — producing orbits that are mirrored,
rotated ninety degrees, or reflected through the plane of the sky, all of
which look completely plausible in a render.

## The tangent triad

At a host with ICRS coordinates (α, δ):

```
e_r     = ( cos δ cos α,  cos δ sin α,  sin δ)    Sun -> star
e_east  = (-sin α,        cos α,        0     )
e_north = (-sin δ cos α, -sin δ sin α,  cos δ )
```

`e_east` and `e_north` are exactly the normalised derivatives of `e_r` with
respect to α and δ — which is how the tests check them: by
finite-differencing an **Astropy** position, not by restating these same
formulas. Restating them would only prove the file was copied correctly.

## The canonical frame, and why +Z points away

```
+X  ->  e_east
+Y  ->  e_north
+Z  ->  e_east × e_north  =  e_r      (away from the observer)
```

The sign of `+Z` is **not** a matter of taste. The standard definition of
the *ascending* node is the crossing where the body moves **away** from the
observer. Under `R_z R_x(i) R_z(ω)`, a body just past the node has
`z = sin(u) sin(i) > 0` for any inclination, prograde or retrograde — it
moves toward `+z`. For that crossing to be the ascending one, `+z` must
point away from us.

> An earlier draft assigned `+x = North, +y = East`. That is perfectly
> self-consistent and makes `R_z(Ω)` carry North toward East — but it forces
> `+z = North × East = −e_r`, *toward* the observer, which would have made
> this codebase's "ascending" node the **approaching** one: a silent 180°
> disagreement with every catalogue it reads.

`test_a_body_just_past_the_ascending_node_is_receding` pins this against the
production propagator, and `LINE_OF_SIGHT` states it in words. No round-trip
test can catch a flipped `+Z` — it round-trips perfectly and mirrors every
orbit. Only a direct assertion catches it.

## A position angle is not a mathematical azimuth

Because `+X` is East and `+Y` is North, `R_z(θ)` carries East toward North.
The catalogued longitude of the ascending node is a **position angle**,
measured from North toward East — the opposite sense, from the other axis:

```
θ = π/2 − Ω_PA
```

`position_angle_to_azimuth()` is the only place that conversion may happen.
Keeping the **source** convention separate from the **internal Cartesian**
convention — rather than feeding a catalogue angle straight into a rotation
matrix — is the entire point of the module.

## A valid rotation does not license publishing

Having `R` is necessary and nowhere near sufficient. Four further gates
apply, and `absolute_position_blockers()` reports **all** that fail, because
a row blocked for three reasons should say three:

| gate | why |
|---|---|
| host located | otherwise there is no origin |
| orbit propagates | otherwise there is no offset |
| node **convention** recorded | an angle with no stated convention is a number, not a direction |
| node **sense** resolved | a measured number is only known modulo 180° |
| not at a celestial pole | RA is not a unique physical direction there |
| coordinate **epoch** handled | see below |

### Node sense is the subtle one

`(ω, Ω)` and `(ω + π, Ω − π)` produce the **same projected orbit** — they
differ only in which node is receding. Relative astrometry therefore
determines the node only modulo 180°, and only radial-velocity or equivalent
line-of-sight information breaks the tie. So `Status.MEASURED` is *not*
enough: `node_sense_of()` defaults a valued node to `MODULO_180`, and only
an explicitly recorded `RESOLVED` unlocks publication.
`test_the_projected_degeneracy_is_real` reproduces the degeneracy directly.

`RESOLVED` also has to **name its evidence**. The claim is that some
observation identified *this orbit's* receding node — a radial-velocity
orbit, an eclipse timing. A generic **systemic** stellar radial velocity is
the thing most likely to be reached for and cannot do it: it describes the
whole system's motion relative to the Sun and says nothing about which of a
planet's two nodes recedes. Since that distinction is invisible in the
number, an unevidenced `RESOLVED` tag falls back to `MODULO_180`.

### The epoch gate currently blocks everything

`SkyPosition` carries no obstime, proper motion or radial velocity. So a
host position is at its catalogue epoch while the planet offset is at the
requested time. For a nearby, high-proper-motion star that mismatch is a
**larger physical error than the AU-scale offset it would be added to** — so
it blocks rather than being waved through as small.

The result: this slice ships the transform, and no real object publishes an
absolute position.
`test_no_real_system_publishes_an_absolute_position` sweeps the whole
snapshot to prove it, and
`test_no_published_planet_triplet_claims_a_celestial_frame` from C3 still
holds unchanged.

A normalised realisation is available via `normalised=True`, stamped
`ASSUMED_FOR_VISUALIZATION` and carrying every unresolved reason. It exists
so a scene can be drawn; it is never a catalogue coordinate, and it cannot
leak into a published distance.

## Planet → selected star distance

Restored behind exactly the same gates:

```
D = |r_other − r_planet|      in one float64 ICRS frame
```

It deliberately does **not** fall back to the host-to-star separation. That
fallback would be plausible — the two differ by an AU at parsec range — and
it would answer a question nobody asked, with nothing on the number saying
it was about the star instead of the planet.

## What comes after

Modelling coordinate epoch and space motion (obstime, proper motion, radial
velocity, evaluated at one common epoch via Astropy) is what would open the
epoch gate. That is its own slice, not a detail to smuggle into a plotting
milestone.
