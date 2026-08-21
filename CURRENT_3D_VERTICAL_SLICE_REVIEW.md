# Current 3D Exoplanet Vertical Slice — Review and Next Steps

**Repository:** `DanielCMBC/astro`  
**Active development branch:** `3D-test`  
**Current milestone:** One-star / one-planet scientifically correct 3D vertical slice  
**Validation object:** HD 80606 b

---

## 1. Current implementation output

Frame mixing is prevented by type.

Example:

```text
system.at([0.05,0,0]) + universe.at([66.47,0,0])
```

raises:

```text
FrameMismatchError
```

Conversions route through absolute parsec coordinates in `float64`.

The rendering conversion refuses to emit a degenerate `float32` position when precision would be lost.

Inside `SystemFrame`, the orbital vector reaches the GPU unchanged, so there is no remaining hidden unit-conversion step inside the rendering layer.

This is the desired scientific/rendering separation.

---

## 2. Verification on HD 80606 b

The current implementation was verified using HD 80606 b.

The OpenGL renderer currently runs with:

```text
OpenGL 3.3 core
offscreen rendering
VAO
VBO
EBO
instanced draws
no fixed-function OpenGL calls
```

The rendered orbit includes 24 equal-time markers.

These markers bunch strongly near apoapsis, demonstrating the expected non-uniform angular motion under Kepler's second law.

The measured arc lengths vary by approximately:

```text
9.5×
```

while equal-time swept area remains constant to approximately:

```text
5.3 × 10^-5
```

This is strong evidence that the animation is propagating the orbit by physical time rather than by equal-angle stepping.

---

## 3. HD 80606 b numerical checks

The reported periapsis distance is:

\[
|r|_{T_{\rm peri}} = 0.03137865\ {\rm AU}
\]

which is consistent with:

\[
r_{\rm peri}=a(1-e)
\]

The reported periapsis velocity is:

\[
v_{\rm peri}=239.925\ {\rm km\,s^{-1}}
\]

which is consistent with:

\[
v_{\rm peri}
=
\sqrt{
\mu
\frac{1+e}{a(1-e)}
}
\]

The orbital-plane normal tilt is:

\[
89.24^\circ
\]

and corresponds to the published inclination:

\[
i = 89.24^\circ
\]

---

## 4. Frame-system assessment

The current frame architecture is a strong design choice.

The implementation distinguishes frames by type rather than convention.

A position in `SystemFrame` cannot be silently added to a position in `UniverseFrame`.

This prevents accidental mixing of:

```text
pc
AU
km
```

inside a common coordinate namespace.

The intended hierarchy remains:

```text
UniverseFrame
    unit = pc
    CPU precision = float64

SystemFrame
    origin = host star
    unit = AU
    CPU physics = float64
    GPU local position = float32

PlanetFrame
    origin = planet
    unit = km or planetary radii
```

The conversion to GPU render coordinates should always happen at the final rendering boundary.

---

## 5. `position_local` naming decision

Keep:

```python
position_local
```

rather than:

```python
position_au
```

`position_local` is the better abstraction because the renderer should remain independent of the scientific unit system.

For example:

```text
SystemFrame
    local coordinates may represent AU

PlanetFrame
    local coordinates may represent km
```

The renderer only needs to know that the position is valid within the active render frame.

---

## 6. Display scaling

The current display model renders the planet approximately:

```text
3.4× larger than its host star
```

for visibility.

This is acceptable as long as the distinction between physical and display radii remains explicit.

Keep these concepts permanently separate:

```text
physical_radius
display_radius
```

The display radius must never enter:

- gravitational calculations;
- transit-depth calculations;
- density calculations;
- collision calculations;
- orbital distances;
- stellar-radius calculations;
- scientific plots;
- physical scale calculations.

It is strictly a rendering parameter.

The UI should disclose when objects are not drawn to physical scale.

---

## 7. NASA `ps` duplicate handling

HD 80606 b has multiple rows in the NASA Exoplanet Archive `ps` table.

The current snapshot reports:

```text
8 rows
```

for HD 80606 b.

The application now selects the row with:

```text
default_flag = 1
```

rather than performing:

```python
drop_duplicates(subset=["pl_name"])
```

after retrieval.

The corrected behavior should remain:

```text
NASA PS
    multiple published solutions may exist

SolutionPolicy
    default_flag = 1

Application scientific state
    deliberate reference solution
```

The current HD 80606 b snapshot resolves to the designated default solution.

---

## 8. Current milestone status

The one-star / one-planet vertical slice should be considered a **pass**, with one orbital-semantics item still needing attention before scaling to many systems.

Current status:

```text
[PASS] frame mismatch protection
[PASS] float64 astronomical coordinate conversion
[PASS] local float32 rendering coordinates
[PASS] modern OpenGL 3.3 core
[PASS] VAO/VBO/EBO pipeline
[PASS] instanced rendering
[PASS] no fixed-function OpenGL
[PASS] physical Kepler propagation
[PASS] Kepler II equal-area behavior
[PASS] high-eccentricity orbit validation
[PASS] periapsis distance
[PASS] periapsis velocity
[PASS] inclination transform
[PASS] NASA default-solution policy
[PASS] scientific/rendering separation
```

---

## 9. Important remaining issue: argument of periastron

The next important scientific issue is the interpretation of:

```text
pl_orblper
```

or the argument of periastron/periapsis.

The NASA Exoplanet Archive may preserve the convention used by the source publication.

Depending on the paper, the reported value may describe:

```text
the planet's orbit
```

or:

```text
the host star's reflex orbit
```

These two conventions differ by:

\[
180^\circ
\]

or:

\[
\pi\ {\rm radians}
\]

Therefore a numerically correct 3D transform may still orient periapsis incorrectly by \(180^\circ\) if the catalog value is interpreted without tracking its convention.

---

## 10. Required periastron convention model

Add an explicit semantic field such as:

```text
PeriastronConvention
```

with values similar to:

```text
PLANET
STELLAR_REFLEX
AS_REPORTED
UNKNOWN
```

The raw catalog value should remain unchanged.

Example:

```python
omega_raw = value_from_archive
omega_convention = "AS_REPORTED"
```

If a publication clearly reports the stellar reflex orbit and the application needs the planet's argument of periapsis:

\[
\omega_{\rm planet}
=
(\omega_{\rm star}+180^\circ)
\bmod 360^\circ
\]

or equivalently:

\[
\omega_{\rm planet}
=
(\omega_{\rm star}+\pi)
\bmod 2\pi
\]

This conversion must be explicit and tested.

---

## 11. Orbital phase / epoch support

The program should next include the orbital epoch needed to distinguish a correct orbital geometry from a correct instantaneous planetary position.

A useful Archive field is:

```text
pl_orbtper
```

representing the time of periastron.

The orbital-state model should distinguish at least:

```text
GEOMETRY_VALID
PHASE_VALID
ORIENTATION_PARTIAL
ORIENTATION_FULL
```

### `GEOMETRY_VALID`

Enough parameters exist to draw the shape of the orbit.

### `PHASE_VALID`

Enough epoch information exists to place the planet at a physically meaningful location at time \(t\).

### `ORIENTATION_PARTIAL`

Some 3D orientation elements are known, but one or more are observationally unconstrained.

### `ORIENTATION_FULL`

The available orbital elements fully specify the 3D orientation under the chosen convention.

---

## 12. Unknown longitude of ascending node

For many exoplanets:

\[
\Omega
\]

is unknown.

The scientific model should preserve:

```python
longitude_ascending_node = None
```

rather than:

```python
longitude_ascending_node = 0.0
```

For visualization, the renderer may normalize:

\[
\Omega_{\rm display}=0
\]

but this should be tagged:

```text
ASSUMED_FOR_VISUALIZATION
```

The UI should state:

```text
Longitude of ascending node: unknown
Display normalization: 0°
```

---

## 13. Recommended regression test for periastron convention

Add a test proving that a stellar-reflex argument of periastron converts to the planetary convention by exactly:

\[
\pi
\]

or:

\[
180^\circ
\]

Example conceptual assertion:

```python
planet_omega = convert_stellar_reflex_to_planet(star_omega)

assert angular_difference(
    planet_omega,
    star_omega
) == pytest.approx(np.pi)
```

with proper angular wrapping.

---

## 14. Recommended next development step

Before scaling to thousands of systems, complete a short orbital-semantics pass.

Tasks:

- [ ] preserve raw \(\omega\);
- [ ] store periastron convention;
- [ ] preserve publication/reference;
- [ ] add `pl_orbtper`;
- [ ] preserve time-system/epoch metadata;
- [ ] distinguish geometry-valid from phase-valid;
- [ ] distinguish partial from full orientation;
- [ ] test the \(180^\circ\) stellar-reflex conversion;
- [ ] preserve unknown \(\Omega\);
- [ ] keep visualization assumptions explicit.

After this is complete, the project is ready to move toward multi-planet rendering.

---

## 15. Next approved milestone

### Multi-planet `SystemFrame` rendering

Recommended features:

```text
multiple planets
batched orbit geometry
per-system local coordinates
per-system LOD
labels
system information panel
physical time controls
display-scale disclosure
provenance-aware orbital fields
```

The renderer should continue receiving only prepared render states.

Example:

```python
RenderPlanet(
    position_local=(x, y, z),
    display_radius=display_radius,
    material_id=material_id,
)
```

It should not receive raw orbital elements and solve the orbit itself.

---

## 16. Current milestone verdict

**PASS — proceed after one orbital-semantics correction pass.**

The current implementation demonstrates that the project has moved beyond the original prototype's largest architectural risks:

- incorrect unit mixing;
- arbitrary orbital scaling;
- fixed-function OpenGL;
- renderer-owned physics;
- equal-angle animation;
- fake duplicate selection;
- hidden scientific assumptions.

The remaining concern is no longer basic orbital mathematics.

It is **orbital metadata semantics**, especially the meaning and provenance of \(\omega\), epoch information, and unknown orientation parameters.

Once those are explicit, the project can safely expand from a scientifically validated one-planet vertical slice into a multi-planet system renderer.
