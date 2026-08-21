# Orbital Mechanics Formula Reference for the 3D Exoplanet Project

This document formats the core orbital-mechanics equations needed for the next `3D-test` milestone.

---

## 1. Mean Motion

\[
n = \frac{2\pi}{P}
\]

---

## 2. Mean Anomaly as a Function of Time

\[
M(t)=M_0+n(t-t_0)
\]

---

## 3. Kepler's Equation

\[
M = E - e\sin E
\]

Newton-Raphson update:

\[
E_{k+1}
=
E_k
-
\frac{E_k-e\sin E_k-M}
{1-e\cos E_k}
\]

---

## 4. Position in the Orbital Plane

\[
x_p = a(\cos E-e)
\]

\[
y_p = a\sqrt{1-e^2}\sin E
\]

\[
z_p = 0
\]

Vector form:

\[
\mathbf r_p =
\begin{bmatrix}
a(\cos E-e) \\
a\sqrt{1-e^2}\sin E \\
0
\end{bmatrix}
\]

---

## 5. Instantaneous Orbital Distance

Using true anomaly \(\nu\):

\[
r =
\frac{a(1-e^2)}
{1+e\cos\nu}
\]

Equivalent form:

\[
r = a(1-e\cos E)
\]

---

## 6. Periapsis and Apoapsis

\[
r_{\rm peri}=a(1-e)
\]

\[
r_{\rm apo}=a(1+e)
\]

---

## 7. True Anomaly from Eccentric Anomaly

\[
\nu =
2\arctan2
\left(
\sqrt{1+e}\sin\frac{E}{2},
\sqrt{1-e}\cos\frac{E}{2}
\right)
\]

---

# 8. Full 3D Orbital Orientation

\[
\mathbf r =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf r_p
\]

where:

- \(i\) = inclination
- \(\omega\) = argument of periapsis
- \(\Omega\) = longitude of ascending node

---

## 9. Rotation About the Z Axis

\[
R_z(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

---

## 10. Rotation About the X Axis

\[
R_x(i)=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos i & -\sin i \\
0 & \sin i & \cos i
\end{bmatrix}
\]

---

## 11. Expanded 3D Position Equations

Let:

\[
x_p=a(\cos E-e)
\]

\[
y_p=a\sqrt{1-e^2}\sin E
\]

Then:

\[
x =
(\cos\Omega\cos\omega-\sin\Omega\sin\omega\cos i)x_p
+
(-\cos\Omega\sin\omega-\sin\Omega\cos\omega\cos i)y_p
\]

\[
y =
(\sin\Omega\cos\omega+\cos\Omega\sin\omega\cos i)x_p
+
(-\sin\Omega\sin\omega+\cos\Omega\cos\omega\cos i)y_p
\]

\[
z =
(\sin\omega\sin i)x_p
+
(\cos\omega\sin i)y_p
\]

---

# 12. Kepler's Third Law

\[
P^2 =
\frac{4\pi^2a^3}
{G(M_\star+M_p)}
\]

Solving for semimajor axis:

\[
a =
\left[
\frac{G(M_\star+M_p)P^2}
{4\pi^2}
\right]^{1/3}
\]

---

# 13. Kepler's Second Law

\[
\frac{dA}{dt}
=
\frac{h}{2}
=
\text{constant}
\]

For equal time intervals:

\[
\Delta t_1 = \Delta t_2 = \cdots
\]

the swept areas should satisfy:

\[
\Delta A_1
\approx
\Delta A_2
\approx
\Delta A_3
\approx
\cdots
\]

within numerical tolerance.

---

# 14. Specific Angular Momentum

\[
\mathbf h
=
\mathbf r \times \mathbf v
\]

\[
h=
\sqrt{\mu a(1-e^2)}
\]

where:

\[
\mu=G(M_\star+M_p)
\]

---

# 15. Orbital Velocity in the Perifocal Frame

\[
\mathbf v_p
=
\frac{na}{1-e\cos E}
\begin{bmatrix}
-\sin E \\
\sqrt{1-e^2}\cos E \\
0
\end{bmatrix}
\]

with:

\[
n=\sqrt{\frac{\mu}{a^3}}
\]

and:

\[
\mathbf v =
R_z(\Omega)
R_x(i)
R_z(\omega)
\mathbf v_p
\]

---

# 16. Vis-Viva Equation

\[
v^2
=
\mu
\left(
\frac{2}{r}
-
\frac{1}{a}
\right)
\]

or:

\[
v=
\sqrt{
\mu
\left(
\frac{2}{r}
-
\frac{1}{a}
\right)
}
\]

---

# 17. Specific Orbital Energy

\[
\epsilon
=
\frac{v^2}{2}
-
\frac{\mu}{r}
\]

For a bound ellipse:

\[
\epsilon
=
-\frac{\mu}{2a}
\]

So a regression test can verify:

\[
\frac{v^2}{2}
-
\frac{\mu}{r}
\approx
-\frac{\mu}{2a}
\]

---

# 18. Coordinate Distance Between Two Objects

\[
D =
\left|
\mathbf r_2-\mathbf r_1
\right|
\]

or explicitly:

\[
D =
\sqrt{
(x_2-x_1)^2+
(y_2-y_1)^2+
(z_2-z_1)^2
}
\]

---

# 19. Spherical to Cartesian Coordinates

Given right ascension \(\alpha\), declination \(\delta\), and distance \(d\):

\[
x=d\cos\alpha\cos\delta
\]

\[
y=d\sin\alpha\cos\delta
\]

\[
z=d\sin\delta
\]

For production code, prefer Astropy coordinate frames.

---

# 20. Important Unit Conversion

\[
1\,{\rm pc}
=
206264.806\,{\rm AU}
\]

Therefore:

\[
1\,{\rm AU}
\approx
4.8481368\times10^{-6}\,{\rm pc}
\]

For production rendering, prefer hierarchical frames rather than direct scaling:

```text
UniverseFrame
    unit = pc

SystemFrame
    unit = AU

PlanetFrame
    unit = km or planetary radii
```

---

# 21. Unknown Orbital Orientation

If \(\Omega\) is observationally unavailable:

```text
longitude_of_ascending_node = UNKNOWN
```

Do not store:

```text
Ω = 0°
```

as though it were measured.

For rendering only, a display normalization may use:

\[
\Omega_{\rm display}=0
\]

with the status:

```text
ASSUMED_FOR_VISUALIZATION
```

---

# 22. Suggested Unit Tests

## Kepler solver

Verify:

\[
|E-e\sin E-M| < \varepsilon
\]

for a range including:

```text
e = 0
e = 0.1
e = 0.5
e = 0.9
e = 0.99
e = 0.9999
```

## Kepler's second law

For equal time intervals:

\[
\Delta A_i \approx \Delta A_j
\]

within tolerance.

## Energy conservation

At many orbital phases:

\[
\frac{v^2}{2}
-
\frac{\mu}{r}
\approx
-\frac{\mu}{2a}
\]

## 3D rotation sanity checks

### Zero inclination

\[
i=0
\]

Orbit remains in the XY plane.

### Polar orientation

\[
i=90^\circ
\]

Orbit rotates into a perpendicular plane.

### Argument of periapsis rotation

\[
\omega=90^\circ
\]

Periapsis rotates correctly.

### Ascending-node rotation

\[
\Omega=90^\circ
\]

The orbital orientation rotates correctly around the reference Z axis.

---

# 23. Scientific-to-Rendering Boundary

The scientific layer should produce:

\[
\mathbf r(t)
\]

and optionally:

\[
\mathbf v(t)
\]

Conceptually:

```python
RenderPlanet(
    position_local=(x, y, z),
    display_radius=display_radius,
    material_id=material_id,
)
```

The renderer should not solve:

\[
M=E-e\sin E
\]

and should not derive:

\[
a,\ e,\ P,\ i,\ \omega,\ \Omega
\]

from raw catalog values.

---

# 24. Recommended Next Milestone

Implement and test:

\[
\text{catalog data}
\rightarrow
\text{validated orbital elements}
\rightarrow
M(t)
\rightarrow
E
\rightarrow
\mathbf r_p
\rightarrow
R_z(\Omega)R_x(i)R_z(\omega)
\rightarrow
\mathbf r
\rightarrow
\text{RenderState}
\rightarrow
\text{OpenGL}
\]

The target should be one scientifically correct host star and one exoplanet before expanding to larger systems.
