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

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from ..assets.procedural import planet_material, star_display_color
from ..coordinates.floating_origin import Scale, SceneGraph
from ..coordinates.system_frame import SystemFrame
from ..physics.orbital_elements import position_at_mean_anomaly
from ..provenance import Status
from .materials import material_for
from .renderer import RenderOrbit, RenderPlanet, RenderStar, SceneDescription

__all__ = [
    "build_system_scene",
    "build_frame_scene",
    "orbit_path",
    "display_radius_au",
    "DisplayScale",
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
            identifier=host.name,
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
                        identifier="{0}:orbit".format(record.name),
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
                identifier=record.name,
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
    exaggerate: bool = True,
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
            identifier=star.name,
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
                        identifier="{0}:orbit".format(record.name),
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
                identifier=record.name,
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

    return scene
