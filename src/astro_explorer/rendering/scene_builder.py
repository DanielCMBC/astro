"""Turns scientific records into a :class:`SceneDescription`.

This is the single crossing point of roadmap section 6's flow diagram: it
reads from the science layer and writes only render primitives.  Every
decision that requires knowing whether a value was measured, derived or
assumed is taken *here*, and what reaches the renderer is a position, a
radius and a material.

The builder refuses to place a planet whose semimajor axis is unknown.  It
would rather draw a star on its own, and say why in an annotation, than
invent an orbit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np

from ..assets.procedural import planet_material, star_display_color
from ..coordinates.floating_origin import Scale, SceneGraph
from ..coordinates.system_frame import SystemFrame
from ..physics.orbital_elements import position_at_mean_anomaly
from ..physics.orientation import (
    inclination_arc,
    node_line,
    orbit_normal,
    orbit_plane_ring,
    periapsis_direction,
    reference_plane_ring,
)
from ..provenance import Status
from .materials import material_for
from .renderer import (
    GuideStyle,
    RenderGuide,
    RenderOrbit,
    RenderPlanet,
    RenderStar,
    RenderZone,
    SceneDescription,
)

__all__ = [
    "build_system_scene",
    "build_frame_scene",
    "orbit_path",
    "habitable_zone_overlay",
    "orientation_guides",
    "OrientationOverlay",
    "display_radius_au",
    "DisplayScale",
    "HABITABLE_ZONE_DISCLAIMER",
    "ORIENTATION_DISCLAIMER",
]

#: Bodies drawn to scale in an AU-wide view would be invisible, so radii are
#: exaggerated by a fixed, disclosed factor rather than an arbitrary one.
STAR_RADIUS_EXAGGERATION = 12.0
PLANET_RADIUS_EXAGGERATION = 400.0

_SOLAR_RADIUS_AU = float((1.0 * u.R_sun).to_value(u.au))
_EARTH_RADIUS_AU = float((1.0 * u.R_earth).to_value(u.au))


def display_radius_au(radius_solar=None, radius_earth=None, *, exaggerate: bool = True) -> float:
    """Display radius in AU, with the exaggeration factor applied.

    Returns a small positive fallback when the radius is unknown, so a body
    with no published radius is still visible - but the caller is expected
    to annotate that the size is not to scale.
    """
    if radius_solar is not None:
        value = radius_solar * _SOLAR_RADIUS_AU
        factor = STAR_RADIUS_EXAGGERATION if exaggerate else 1.0
    elif radius_earth is not None:
        value = radius_earth * _EARTH_RADIUS_AU
        factor = PLANET_RADIUS_EXAGGERATION if exaggerate else 1.0
    else:
        return 0.004
    return float(max(value * factor, 1e-6))


@dataclass
class DisplayScale:
    """How much body radii are exaggerated, and why.

    Drawing an AU-wide orbit to scale makes every body sub-pixel, so some
    exaggeration is unavoidable. Two constraints keep it from becoming a
    lie:

    * the star is never drawn wider than a fraction of the closest approach,
      so a planet cannot appear to orbit inside its host;
    * a planet is never drawn larger than its star, which is the error the
      naive per-body factors produced - a Jupiter-radius planet came out
      3.4 times the width of a solar-radius host.

    The effective factors are reported so the viewer knows what was done.
    """

    star_factor: float = 1.0
    planet_factor: float = 1.0
    star_radius_au: float = 0.0

    #: Star diameter as a fraction of the closest periapsis in the system.
    MAX_STAR_FRACTION_OF_PERIAPSIS = 0.35
    #: A planet may not exceed this fraction of the drawn stellar radius.
    MAX_PLANET_FRACTION_OF_STAR = 0.5

    @classmethod
    def for_system(
        cls,
        star_radius_au: float | None,
        min_periapsis_au: float | None,
        *,
        exaggerate: bool = True,
    ) -> "DisplayScale":
        if not exaggerate:
            return cls(1.0, 1.0, star_radius_au or 0.0)

        star_radius = star_radius_au or (1.0 * _SOLAR_RADIUS_AU)
        star_factor = STAR_RADIUS_EXAGGERATION
        if min_periapsis_au and min_periapsis_au > 0.0 and star_radius > 0.0:
            cap = cls.MAX_STAR_FRACTION_OF_PERIAPSIS * min_periapsis_au / star_radius
            star_factor = min(star_factor, max(cap, 1.0))

        drawn_star = star_radius * star_factor
        return cls(star_factor=star_factor, planet_factor=PLANET_RADIUS_EXAGGERATION,
                   star_radius_au=drawn_star)

    def star_radius(self, radius_solar: float | None) -> float:
        if radius_solar is None:
            return max(self.star_radius_au, 1e-6) or 0.004
        return float(max(radius_solar * _SOLAR_RADIUS_AU * self.star_factor, 1e-6))

    def planet_radius(self, radius_earth: float | None) -> float:
        """Planet radius, capped so it can never exceed its host star."""
        if radius_earth is None:
            base = 0.25 * self.star_radius_au
        else:
            base = radius_earth * _EARTH_RADIUS_AU * self.planet_factor
        if self.star_radius_au > 0.0:
            base = min(base, self.MAX_PLANET_FRACTION_OF_STAR * self.star_radius_au)
        return float(max(base, 1e-7))

    def effective_planet_factor(self, radius_earth: float | None) -> float:
        if not radius_earth:
            return float("nan")
        true_radius = radius_earth * _EARTH_RADIUS_AU
        return self.planet_radius(radius_earth) / true_radius if true_radius else float("nan")

    def describe(self, radius_earth: float | None = None) -> str:
        if self.star_factor == 1.0 and self.planet_factor == 1.0:
            return "Bodies and orbits are drawn to a common scale."
        planet_factor = self.effective_planet_factor(radius_earth)
        planet_text = (
            "{0:.3g}x".format(planet_factor) if np.isfinite(planet_factor) else "nominal"
        )
        return (
            "Body radii are exaggerated relative to orbital distances: star {0:.3g}x, "
            "planet {1}. Relative body sizes are therefore not to scale.".format(
                self.star_factor, planet_text
            )
        )


def orbit_path(elements, samples: int = 512) -> np.ndarray | None:
    """Sample a full orbit in the reference frame, in AU.

    Sampling is uniform in eccentric anomaly rather than in mean anomaly, so
    a highly eccentric orbit gets enough points near periapsis where the
    curvature is greatest.
    """
    display = elements.for_display()
    if not display.semimajor_axis.is_known:
        return None

    from ..physics.orbital_elements import position_at_eccentric_anomaly

    ecc_anomaly = np.linspace(0.0, 2.0 * np.pi, samples)
    return position_at_eccentric_anomaly(display, ecc_anomaly)


#: Said in words wherever the zone is drawn. The Kopparapu bounds are a
#: statement about stellar irradiation - where liquid water on an
#: Earth-like planet with an Earth-like atmosphere would be
#: thermodynamically possible - and nothing at all about whether a planet
#: there is habitable, has water, or has an atmosphere.
HABITABLE_ZONE_DISCLAIMER = (
    "The habitable zone is a stellar-irradiation model, not a claim about "
    "habitability: it says where an Earth-like atmosphere could support "
    "liquid surface water, not that any planet drawn inside it does."
)

#: Fill and edge colours for the habitable-zone band. Purely display
#: choices: neither value is read by anything that computes a boundary, and
#: changing them cannot move the zone.
HABITABLE_ZONE_FILL = (0.36, 0.78, 0.52, 0.13)
HABITABLE_ZONE_EDGE = (0.45, 0.90, 0.62, 0.50)


def habitable_zone_overlay(
    zone, frame, *, samples: int = 240, identifier: str = "habitable-zone"
) -> RenderZone | None:
    """The habitable zone as a flat band, or None when there is none.

    ``zone`` is the :class:`~astro_explorer.physics.stellar.HabitableZone`
    the star record already computed - the same object the info panel
    reports. Nothing here re-derives a boundary: the Kopparapu polynomial
    lives in the physics layer and is evaluated once, upstream, so the
    overlay and the panel cannot drift apart or disagree.

    Returns None when either edge is unknown, which is what the physics
    layer reports for a star with no luminosity, no effective temperature,
    or a temperature outside the range the coefficients were fitted for. A
    missing zone is drawn as nothing, never as a default ring.

    The band is a flat annulus on the frame's reference plane: a
    *cross-section* of the zone, not the zone itself. The physical region is
    a spherical shell around the star - a range of radial distances, not a
    region of one plane - and the caller says so in an annotation rather
    than leaving the flat band to imply otherwise.
    """
    if zone is None or not zone.is_known:
        return None

    inner_au = zone.inner.value_in(u.au)
    outer_au = zone.outer.value_in(u.au)
    if inner_au is None or outer_au is None:
        return None
    if not (np.isfinite(inner_au) and np.isfinite(outer_au)):
        return None
    if inner_au <= 0.0 or outer_au <= inner_au:
        return None

    angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    unit = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)

    return RenderZone(
        identifier=identifier,
        inner_points_local=frame.place_planet(unit * inner_au).to_render(),
        outer_points_local=frame.place_planet(unit * outer_au).to_render(),
        color=HABITABLE_ZONE_FILL,
        edge_color=HABITABLE_ZONE_EDGE,
        label="habitable zone",
    )


# ==========================================================================
# Orientation guides (Explorer C2)
# ==========================================================================

#: Said in words wherever a guide is drawn from a normalised element. The
#: honest problem C2 exists to solve is not drawing lines; it is that a line
#: drawn from an angle nobody measured looks exactly like one drawn from an
#: angle somebody did.
ORIENTATION_DISCLAIMER = (
    "A dashed guide is a display normalisation, not an observation: the "
    "element behind it was never published, or the convention it was "
    "published under was never stated."
)

#: Guide colours. Colour is deliberately *not* how provenance is carried -
#: a colour-blind viewer, a greyscale print or a screenshot would all lose
#: it. Stroke style carries it and the annotations say it in words; colour
#: only separates one guide from another.
GUIDE_REFERENCE_PLANE = (0.62, 0.62, 0.64, 0.40)
GUIDE_ORBIT_PLANE = (0.58, 0.78, 0.96, 0.60)
GUIDE_ORBIT_NORMAL = (0.58, 0.78, 0.96, 0.85)
GUIDE_NODE_LINE = (0.96, 0.82, 0.45, 0.85)
GUIDE_PERIAPSIS = (0.98, 0.55, 0.42, 0.90)
GUIDE_INCLINATION = (0.70, 0.92, 0.72, 0.80)

#: Below this the inclination arc is a degenerate point rather than an arc,
#: and a coplanar orbit is better said than drawn.
COPLANAR_TOLERANCE_RAD = 1.0e-4


def _arrow_polyline(origin, tip, *, head_fraction: float = 0.14) -> np.ndarray:
    """An arrow as a single polyline: shaft, then back along each barb.

    One primitive rather than three keeps a guide batchable as one indexed
    line strip, and the doubled-back segments cost two extra vertices.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    shaft = tip - origin
    length = float(np.linalg.norm(shaft))
    if length <= 0.0:
        return np.stack([origin, tip])

    direction = shaft / length
    # Any vector not parallel to the shaft gives a barb plane; the pole
    # serves except for an arrow that is itself the pole.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ reference)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    perpendicular = np.cross(direction, reference)
    perpendicular /= np.linalg.norm(perpendicular)

    head = length * head_fraction
    back = tip - head * direction
    spread = 0.45 * head * perpendicular
    return np.stack([origin, tip, back + spread, tip, back - spread])


def _closed(points) -> np.ndarray:
    """Repeat the first point so a ring is drawn closed."""
    array = np.asarray(points, dtype=np.float64)
    return np.vstack([array, array[:1]])


def _guide_style(*parameters) -> GuideStyle:
    """Solid unless something behind the guide was assumed.

    DERIVED is drawn solid: an argument of periastron converted from the
    host star's reflex orbit by 180 degrees is a real orientation, reached
    by a stated transform from a stated convention. It is still labelled
    derived - but it is not a guess, and dashing it would say it was.
    """
    return (
        GuideStyle.DASHED
        if any(p.status is Status.ASSUMED_FOR_VISUALIZATION for p in parameters)
        else GuideStyle.SOLID
    )


def _provenance_word(parameter) -> str:
    if parameter.status is Status.MEASURED:
        return "measured"
    if parameter.status is Status.DERIVED:
        return "derived"
    if parameter.status is Status.ASSUMED_FOR_VISUALIZATION:
        return "assumed for display"
    return "unknown"


@dataclass
class OrientationOverlay:
    """Finished orientation guides plus the words that qualify them.

    The two travel together on purpose. A guide without its annotation is
    the failure this overlay exists to prevent: once it is a line on a
    screen, a normalised node is indistinguishable from a measured one, and
    only the text says which it was.
    """

    guides: list = field(default_factory=list)
    annotations: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.guides)

    def guide(self, name: str):
        """One guide by the short name it was built under, or None."""
        return next(
            (g for g in self.guides if g.identifier.rsplit(":", 1)[-1] == name), None
        )


def orientation_guides(
    elements,
    frame,
    *,
    radius: float | None = None,
    samples: int = 240,
    identifier: str = "orbit",
    show_normalised: bool = False,
    label: str = "",
) -> OrientationOverlay:
    """Orientation overlays for one orbit, with their provenance in words.

    Drawn for a *selected* orbit rather than for every orbit at once:

    * the system reference plane, and the orbital plane against it;
    * the orbit normal;
    * the line of nodes;
    * the periapsis direction;
    * an inclination indicator swept about the nodes.

    Every one of them is built from
    :mod:`astro_explorer.physics.orientation` - the same rotation the
    propagator uses - so a guide cannot express a different reading of
    ``Omega``, ``i`` and ``omega`` from the orbit it is drawn against.

    What may be drawn at all is decided here, from provenance:

    ==========================  =============================================
    the defining element is     the guide is
    ==========================  =============================================
    MEASURED                    drawn solid
    DERIVED                     drawn solid, and labelled derived
    ASSUMED_FOR_VISUALIZATION   drawn dashed, and labelled assumed
    UNKNOWN                     **not drawn**, unless ``show_normalised``
    ==========================  =============================================

    ``show_normalised`` is the switch for "show me where the display put the
    things nobody measured". It is off by default, because the guide it
    controls - a line of nodes for an orbit whose node was never observed -
    is the most misreadable object in this overlay: it looks like a
    direction on the sky, and there is no such direction to look at.

    A guide whose own element is known but whose *placement* rests on a
    normalised one is a different case, and is drawn dashed rather than
    withheld: the inclination of a transiting planet is a real measurement
    even though the azimuth it is drawn at is not.
    """
    overlay = OrientationOverlay()
    display = elements.for_display()

    axis = display.semimajor_axis.value_in(u.au)
    if axis is None or not np.isfinite(axis) or axis <= 0.0:
        overlay.annotations.append(
            "{0}: no semimajor axis published or derivable, so there is no "
            "orbit to orient.".format(label or elements.name or "this orbit")
        )
        return overlay

    eccentricity = display.eccentricity.value_in(u.dimensionless_unscaled, 0.0)
    if radius is None:
        radius = axis * (1.0 + eccentricity)

    inclination = display.inclination
    node = display.longitude_of_ascending_node
    periastron = display.argument_of_periastron

    i_rad = inclination.value_in(u.rad, 0.0)
    node_rad = node.value_in(u.rad, 0.0)
    omega_rad = periastron.value_in(u.rad, 0.0)

    i_known = elements.inclination.is_known
    node_known = elements.longitude_of_ascending_node.is_known
    omega_known = elements.argument_of_periastron.is_known

    def add(name, points, style, color, text):
        overlay.guides.append(
            RenderGuide(
                identifier="{0}:{1}".format(identifier, name),
                points_local=frame.place_planet(points).to_render(),
                style=style,
                color=color,
                label=text,
            )
        )

    # -- the reference plane ---------------------------------------------
    # Not a claim about this system: it is the plane the catalogue's angles
    # are measured against, which for a transiting exoplanet is the sky.
    add(
        "reference-plane",
        _closed(reference_plane_ring(radius, samples)),
        GuideStyle.SOLID,
        GUIDE_REFERENCE_PLANE,
        "system reference plane",
    )
    overlay.annotations.append(
        "System reference plane: the plane inclination is measured against - "
        "the plane of the sky for a transiting orbit, so i = 90 deg is "
        "edge-on. Which direction within it is +x is a display convention, "
        "not a measured direction on the sky."
    )

    # -- the orbital plane and its normal --------------------------------
    if i_known or show_normalised:
        style = _guide_style(inclination, node)
        add(
            "orbit-plane",
            _closed(orbit_plane_ring(i_rad, node_rad, radius, samples)),
            style,
            GUIDE_ORBIT_PLANE,
            "orbital plane",
        )
        add(
            "orbit-normal",
            _arrow_polyline(np.zeros(3), orbit_normal(i_rad, node_rad) * radius * 0.65),
            style,
            GUIDE_ORBIT_NORMAL,
            "orbit normal",
        )
        overlay.annotations.append(
            "Inclination: {0} ({1}); the orbital plane and its normal are "
            "drawn from it.".format(
                inclination.to(u.deg).format(with_status=False),
                _provenance_word(elements.inclination if i_known else inclination),
            )
        )
        if not node_known:
            overlay.annotations.append(
                "The tilt of that plane is measured; the direction it is "
                "tilted towards is not. The plane is therefore drawn dashed, "
                "and may be rotated about the line of sight from the true one."
            )

    # -- the inclination indicator ---------------------------------------
    if (i_known or show_normalised) and abs(i_rad) > COPLANAR_TOLERANCE_RAD:
        add(
            "inclination",
            inclination_arc(i_rad, node_rad, radius * 0.45, max(8, samples // 6)),
            _guide_style(inclination, node),
            GUIDE_INCLINATION,
            "inclination {0}".format(inclination.to(u.deg).format(with_status=False)),
        )
    elif i_known:
        overlay.annotations.append(
            "Inclination is zero to within the published precision: the orbit "
            "lies in the reference plane, so there is no arc to draw."
        )

    # -- the line of nodes -----------------------------------------------
    if node_known or show_normalised:
        add(
            "ascending-node",
            node_line(node_rad, radius),
            _guide_style(node),
            GUIDE_NODE_LINE,
            "line of nodes",
        )
    if node_known:
        overlay.annotations.append(
            "Ascending node: {0} (measured).".format(
                node.to(u.deg).format(with_status=False)
            )
        )
    else:
        overlay.annotations.append(
            "Ascending node: unknown. Display normalisation Omega = 0 deg. "
            "The absolute rotation of this orbit about the line of sight is "
            "unconstrained{0}.".format(
                ", and the normalised line of nodes is drawn dashed"
                if show_normalised
                else ", so no line of nodes is drawn"
            )
        )

    # -- the periapsis direction -----------------------------------------
    resolved = elements.argument_of_periapsis_planet
    if omega_known or show_normalised:
        tip = periapsis_direction(i_rad, omega_rad, node_rad) * axis * (
            1.0 - eccentricity
        )
        add(
            "periapsis",
            _arrow_polyline(np.zeros(3), tip),
            _guide_style(periastron, inclination, node),
            GUIDE_PERIAPSIS,
            "periapsis",
        )
    if omega_known:
        overlay.annotations.append(
            "Argument of periastron: {0} as catalogued ({1}); drawn as the "
            "planet's periapsis at {2} ({3}). The arrow ends at the "
            "periapsis distance, on the orbit.".format(
                elements.argument_of_periastron.to(u.deg).format(with_status=False),
                elements.periastron_convention.label,
                resolved.to(u.deg).format(with_status=False),
                _provenance_word(resolved),
            )
        )
        if elements.periastron_convention_is_assumed:
            overlay.annotations.append(
                "Periapsis direction is assumed, not measured: "
                + elements.periastron_convention.caveat
            )
    else:
        overlay.annotations.append(
            "Argument of periastron: unknown, so periapsis has no direction to "
            "point at{0}.".format(
                "; the normalised arrow is drawn dashed at omega = 0 deg"
                if show_normalised
                else " and no periapsis arrow is drawn"
            )
        )

    if any(guide.style is GuideStyle.DASHED for guide in overlay.guides):
        overlay.annotations.append(ORIENTATION_DISCLAIMER)

    return overlay

def _orbit_is_assumed(elements) -> bool:
    """True when any element used to draw the path was substituted."""
    display = elements.for_display()
    return any(
        parameter.status is Status.ASSUMED_FOR_VISUALIZATION
        for parameter in (
            display.eccentricity,
            display.inclination,
            display.argument_of_periastron,
            display.longitude_of_ascending_node,
        )
    ) or display.semimajor_axis.status is Status.DERIVED


def build_system_scene(
    planets,
    *,
    mean_anomalies=None,
    scene_graph: SceneGraph | None = None,
    host_position_pc=(0.0, 0.0, 0.0),
    draw_orbits: bool = True,
) -> SceneDescription:
    """Build a host-system scene from :class:`PlanetRecord` objects.

    Parameters
    ----------
    planets:
        Records sharing one host star.
    mean_anomalies:
        Mapping of planet name to mean anomaly in radians.  A planet absent
        from the mapping (because its phase is not computable) has its orbit
        drawn but no body placed on it.
    """
    scene = SceneDescription(unit_label="AU")
    if not planets:
        scene.annotations.append("No planets to display.")
        return scene

    graph = scene_graph or SceneGraph()
    if graph.origin.scale is not Scale.SYSTEM:
        graph.enter_system(host_position_pc)

    host = planets[0].host
    mean_anomalies = mean_anomalies or {}

    # -- host star -------------------------------------------------------
    color = star_display_color(host.effective_temperature)
    star_radius = display_radius_au(radius_solar=host.radius.value_in(u.R_sun))
    scene.stars.append(
        RenderStar(
            identifier=str(host.entity_id) if host.entity_id else host.name,
            position_local=graph.host_render_position(host_position_pc),
            radius_display=star_radius,
            color=color.stylized if color else (1.0, 0.95, 0.85),
            temperature_k=color.temperature_k if color else None,
            label=host.name,
        )
    )
    if color is None:
        scene.annotations.append(
            "Host star colour is a placeholder: no effective temperature published."
        )
    if not host.radius.is_known:
        scene.annotations.append("Host star radius unknown; drawn at a nominal size.")

    scene.annotations.append(
        "Body sizes are exaggerated {0:g}x (stars) and {1:g}x (planets) "
        "relative to orbital distances.".format(
            STAR_RADIUS_EXAGGERATION, PLANET_RADIUS_EXAGGERATION
        )
    )

    # -- planets ---------------------------------------------------------
    for record in planets:
        elements = record.elements
        if not elements.semimajor_axis.is_known:
            scene.annotations.append(
                "{0}: no semimajor axis published or derivable; not drawn.".format(record.name)
            )
            continue

        display = elements.for_display()
        assumed = _orbit_is_assumed(elements)

        if draw_orbits:
            path = orbit_path(display)
            if path is not None:
                points = np.array(
                    [graph.planet_render_position(host_position_pc, point) for point in path]
                )
                scene.orbits.append(
                    RenderOrbit(
                        identifier="{0}:orbit".format(
                            record.entity_id or record.name
                        ),
                        points_local=points,
                        dashed=assumed,
                    )
                )

        anomaly = mean_anomalies.get(record.name)
        if anomaly is None:
            scene.annotations.append(
                "{0}: orbital phase not constrained; the path is shown without a "
                "current position.".format(record.name)
            )
            continue

        offset_au = position_at_mean_anomaly(display, float(anomaly))
        material = planet_material(
            record.radius_earth, record.equilibrium_temperature, record.bulk_density
        )
        definition = material_for(material.material_class)

        scene.planets.append(
            RenderPlanet(
                identifier=str(record.entity_id) if record.entity_id else record.name,
                position_local=graph.planet_render_position(host_position_pc, offset_au),
                radius_display=display_radius_au(
                    radius_earth=record.radius_earth.value_in(u.R_earth)
                ),
                material_id=definition.material_id,
                base_color=material.base_color,
                emissive=material.emissive,
                banding=material.banding,
                roughness=material.roughness,
                label=record.name,
            )
        )

        if assumed:
            scene.annotations.append(
                "{0}: orbit drawn using assumed orientation or eccentricity.".format(record.name)
            )

    return scene


# ==========================================================================
# Frame-aware scene construction (the vertical slice)
# ==========================================================================


def build_frame_scene(
    frame: SystemFrame,
    star,
    planets,
    *,
    mean_anomalies=None,
    draw_orbits: bool = True,
    orbit_samples: int = 720,
    draw_habitable_zone: bool = True,
    exaggerate: bool = True,
    orientation_for: str | None = None,
    show_normalised_orientation: bool = False,
) -> SceneDescription:
    """Build a scene entirely inside one :class:`SystemFrame`.

    This is the crossing point of the vertical slice. Everything above it
    works in AU, float64, with provenance; everything below it receives
    float32 positions, a radius and a material id, and nothing else.

    The host star is at the frame origin by construction, and a planet's
    position is the orbital vector the physics layer produced, labelled as a
    frame position and narrowed to float32 exactly once. No scale factor is
    applied anywhere in this function - that absence is the fix for the
    prototype's AU-to-parsec bug.

    Parameters
    ----------
    frame:
        The host-system frame; its origin is the star.
    star:
        A :class:`~astro_explorer.data.schema.StarRecord`.
    planets:
        :class:`~astro_explorer.data.schema.PlanetRecord` objects orbiting it.
    mean_anomalies:
        Planet name to mean anomaly in radians. A planet missing from the
        mapping has its orbit drawn but no body placed on it, because its
        phase is not constrained.
    orientation_for:
        Name or entity id of the planet whose orientation guides to draw
        (Explorer C2). Guides are per-selection rather than per-system: six
        sets of planes and node lines at once would be unreadable, and the
        question they answer - "how is *this* orbit oriented" - is asked
        about one planet at a time. ``None`` draws none.
    show_normalised_orientation:
        Whether guides for elements nobody published may be drawn, dashed,
        at their display normalisation. Off by default.
    """
    scene = SceneDescription(unit_label=frame.unit_label)
    mean_anomalies = mean_anomalies or {}

    # The size model needs to know the tightest orbit in the system before it
    # can decide how much the star may be inflated.
    periapses = [
        p.elements.periapsis.value_in(u.au)
        for p in planets
        if p.elements.periapsis.is_known
    ]
    scale = DisplayScale.for_system(
        star.radius.value_in(u.R_sun, 0.0) * _SOLAR_RADIUS_AU or None,
        min(periapses) if periapses else None,
        exaggerate=exaggerate,
    )

    # -- host star -------------------------------------------------------
    color = star_display_color(star.effective_temperature)
    star_position = frame.star_position()
    scene.stars.append(
        RenderStar(
            # The stable key is the identity picking and selection use; the
            # display name is only ever a label, so a catalogue renaming
            # cannot invalidate a selection.
            identifier=str(star.entity_id) if star.entity_id else star.name,
            position_local=star_position.to_render(),
            radius_display=scale.star_radius(star.radius.value_in(u.R_sun)),
            color=color.stylized if color else (1.0, 0.95, 0.85),
            temperature_k=color.temperature_k if color else None,
            label=star.name,
        )
    )
    if color is None:
        scene.annotations.append(
            "Host star colour is a placeholder: no effective temperature published."
        )
    if not star.radius.is_known:
        scene.annotations.append("Host star radius unknown; drawn at a nominal size.")

    largest = max(
        (p.radius_earth.value_in(u.R_earth) for p in planets if p.radius_earth.is_known),
        default=None,
    )
    scene.annotations.append(scale.describe(largest))

    # -- habitable zone --------------------------------------------------
    if draw_habitable_zone:
        zone = star.habitable_zone
        overlay = habitable_zone_overlay(zone, frame)
        if overlay is not None:
            scene.zones.append(overlay)
            scene.annotations.append(
                "Habitable-zone cross-section {0:.3g}-{1:.3g} AU ({2}): radial "
                "irradiation boundaries shown in the system reference plane. "
                "The physical region is a spherical shell around the star. "
                "{3}".format(
                    zone.inner.value_in(u.au),
                    zone.outer.value_in(u.au),
                    zone.model,
                    HABITABLE_ZONE_DISCLAIMER,
                )
            )
        else:
            scene.annotations.append(
                "No habitable zone drawn: {0} publishes no luminosity and "
                "effective temperature the model accepts, so its boundaries "
                "are unknown rather than defaulted.".format(star.name or "this star")
            )

    # -- planets ---------------------------------------------------------
    for record in planets:
        elements = record.elements
        if not elements.semimajor_axis.is_known:
            scene.annotations.append(
                "{0}: no semimajor axis published or derivable; not drawn.".format(record.name)
            )
            continue

        display = elements.for_display()
        assumed = _orbit_is_assumed(elements)

        if draw_orbits:
            path_au = orbit_path(display, samples=orbit_samples)
            if path_au is not None:
                points = frame.place_planet(path_au).to_render()
                scene.orbits.append(
                    RenderOrbit(
                        identifier="{0}:orbit".format(
                            record.entity_id or record.name
                        ),
                        points_local=points,
                        dashed=assumed,
                    )
                )

        anomaly = mean_anomalies.get(record.name)
        if anomaly is None:
            scene.annotations.append(
                "{0}: orbital phase not constrained; the path is shown without a "
                "current position.".format(record.name)
            )
            continue

        offset_au = position_at_mean_anomaly(display, float(anomaly))
        material = planet_material(
            record.radius_earth, record.equilibrium_temperature, record.bulk_density
        )
        definition = material_for(material.material_class)

        scene.planets.append(
            RenderPlanet(
                identifier=str(record.entity_id) if record.entity_id else record.name,
                position_local=frame.place_planet(offset_au).to_render(),
                radius_display=scale.planet_radius(
                    record.radius_earth.value_in(u.R_earth)
                ),
                material_id=definition.material_id,
                base_color=material.base_color,
                emissive=material.emissive,
                banding=material.banding,
                roughness=material.roughness,
                label=record.name,
            )
        )

        if assumed:
            scene.annotations.append(
                "{0}: orbit drawn using assumed orientation or eccentricity.".format(record.name)
            )

    # -- orientation guides for the selected orbit -----------------------
    if orientation_for is not None:
        selected = next(
            (
                record
                for record in planets
                if orientation_for in (record.name, str(record.entity_id or ""))
            ),
            None,
        )
        if selected is not None:
            overlay = orientation_guides(
                selected.elements,
                frame,
                identifier=str(selected.entity_id or selected.name),
                show_normalised=show_normalised_orientation,
                label=selected.name,
            )
            scene.guides.extend(overlay.guides)
            scene.annotations.append(
                "Orientation guides: {0}.".format(selected.name)
            )
            scene.annotations.extend(overlay.annotations)

    return scene
