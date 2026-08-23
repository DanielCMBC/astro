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
