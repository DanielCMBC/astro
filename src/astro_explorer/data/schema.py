"""Catalogue rows to provenance-aware records.

This module is where roadmap sections 3.3 and 3.4 are enforced:

* a missing semimajor axis is derived from the period and the stellar mass
  and tagged DERIVED, or left UNKNOWN.  It never becomes 1 AU;
* a missing eccentricity stays UNKNOWN.  The circular-orbit assumption is
  applied only by :meth:`OrbitalElements.for_display`, which tags it
  ASSUMED_FOR_VISUALIZATION.

Every field records which catalogue column it came from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

import astropy.units as u
import numpy as np

from ..coordinates.frames import SkyPosition, sky_position
from ..physics.ephemeris import kepler_third_law_residual, semimajor_axis_from_period
from ..physics.epoch import TimeScale
from ..physics.orbital_elements import OrbitalElements
from ..physics.orbital_semantics import PeriastronConvention
from ..physics.stellar import (
    equilibrium_temperature,
    habitable_zone_au,
    insolation_earth_units,
    luminosity_from_radius_and_teff,
)
from ..provenance import Parameter, measured, unknown
from ..text import clean_text
from .nasa_archive import SolutionPolicy

__all__ = [
    "StarRecord",
    "PlanetRecord",
    "parse_float",
    "clean_text",
    "clean_reference",
    "reference_url",
    "build_planet_record",
]


#: The archive wraps references in an HTML anchor.  Both the display name
#: and the ADS link are worth keeping; neither is worth showing raw.
_REF_TAG = re.compile(r"<[^>]*>")
_REF_HREF = re.compile(r"href\s*=\s*([^\s>]+)")


def clean_reference(raw: Any) -> str | None:
    """Human-readable citation from the archive's ``*_refname`` markup."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not clean_text(text):
        return None
    stripped = _REF_TAG.sub("", text).strip()
    return stripped or None


def reference_url(raw: Any) -> str | None:
    """The ADS (or other) URL embedded in a ``*_refname`` value."""
    if raw is None:
        return None
    match = _REF_HREF.search(str(raw))
    if not match:
        return None
    url = match.group(1).strip("\"'")
    return url or None


def parse_float(value: Any, default: float = math.nan) -> float:
    """Tolerant numeric coercion for catalogue cells.

    Handles the archive's ``null`` strings and limit markers (``<``, ``>``).
    Unlike the original helper this never invents a substitute value; the
    caller decides what an unavailable number means.
    """
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(float(value)) else default
    text = str(value).replace("<", "").replace(">", "").strip()
    if not text or text.lower() in ("null", "nan", "none", "--"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _param_from_row(
    row: Mapping[str, Any],
    column: str,
    unit: u.UnitBase,
    *,
    table: str,
    reference: str | None = None,
    retrieved: date | None = None,
    error_suffixes: tuple[str, str] = ("err1", "err2"),
) -> Parameter:
    """One catalogue column, with its asymmetric error columns if present."""
    value = parse_float(row.get(column))
    if not math.isfinite(value):
        return unknown(unit, provenance="{0}.{1}".format(table, column))

    plus = parse_float(row.get(column + error_suffixes[0]))
    minus = parse_float(row.get(column + error_suffixes[1]))
    return measured(
        value,
        unit,
        error_plus=None if not math.isfinite(plus) else plus,
        error_minus=None if not math.isfinite(minus) else minus,
        provenance="{0}.{1}".format(table, column),
        reference=reference,
        retrieved=retrieved,
    )


@dataclass(frozen=True)
class StarRecord:
    """A host star with provenance-aware parameters."""

    name: str
    effective_temperature: Parameter = unknown(u.K)
    radius: Parameter = unknown(u.R_sun)
    mass: Parameter = unknown(u.M_sun)
    luminosity: Parameter = unknown(u.L_sun)
    metallicity: Parameter = unknown()
    age: Parameter = unknown(u.Gyr)
    spectral_type: str = ""
    position: SkyPosition | None = None
    source_table: str = ""
    reference: str | None = None

    @property
    def habitable_zone(self):
        return habitable_zone_au(self.luminosity, self.effective_temperature)

    def describe(self) -> list[str]:
        lines = [
            "Host star:         {0}".format(self.name),
            "Spectral type:     {0}".format(self.spectral_type or "unknown"),
            "Effective temp.:   {0}".format(self.effective_temperature.format()),
            "Radius:            {0}".format(self.radius.format()),
            "Mass:              {0}".format(self.mass.format()),
            "Luminosity:        {0}".format(self.luminosity.format()),
            "Metallicity:       {0} [Fe/H]".format(self.metallicity.format()),
            "Age:               {0}".format(self.age.format()),
        ]
        if self.position is not None:
            lines.append("Distance:          {0}".format(self.position.distance.format()))
        return lines


@dataclass(frozen=True)
class PlanetRecord:
    """A planet, its host, its orbit and the provenance of all of it."""

    name: str
    host: StarRecord
    elements: OrbitalElements
    radius_earth: Parameter = unknown(u.R_earth)
    mass_earth: Parameter = unknown(u.M_earth)
    density: Parameter = unknown(u.g / u.cm**3)
    equilibrium_temperature_published: Parameter = unknown(u.K)
    insolation_published: Parameter = unknown()
    mass_provenance: str = ""
    discovery_method: str = ""
    discovery_year: Parameter = unknown(u.yr)
    discovery_facility: str = ""
    solution_policy: SolutionPolicy = SolutionPolicy.COMPOSITE
    source_table: str = ""
    reference: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived quantities ----------------------------------------------
    @property
    def semimajor_axis(self) -> Parameter:
        """Published axis if there is one, otherwise the Kepler-3 derivation."""
        return self.elements.semimajor_axis

    @property
    def equilibrium_temperature(self) -> Parameter:
        """Published Teq, falling back to the derived energy-balance value."""
        if self.equilibrium_temperature_published.is_known:
            return self.equilibrium_temperature_published
        return equilibrium_temperature(self.host.luminosity, self.semimajor_axis)

    @property
    def insolation(self) -> Parameter:
        if self.insolation_published.is_known:
            return self.insolation_published
        return insolation_earth_units(self.host.luminosity, self.semimajor_axis)

    @property
    def surface_gravity(self) -> Parameter:
        """``g = G M / R^2`` in m/s^2, when both mass and radius are known."""
        from ..physics.constants import G, M_EARTH, R_EARTH
        from ..provenance import derived

        mass = self.mass_earth.value_in(u.M_earth)
        radius = self.radius_earth.value_in(u.R_earth)
        if mass is None or radius is None or radius <= 0:
            return unknown(u.m / u.s**2, provenance="G M / R^2")
        gravity = (G * mass * M_EARTH / (radius * R_EARTH) ** 2).to(u.m / u.s**2)
        return derived(float(gravity.value), u.m / u.s**2, provenance="G M / R^2")

    @property
    def bulk_density(self) -> Parameter:
        """Published density, otherwise derived from mass and radius."""
        from ..physics.constants import M_EARTH, R_EARTH
        from ..provenance import derived

        if self.density.is_known:
            return self.density
        mass = self.mass_earth.value_in(u.M_earth)
        radius = self.radius_earth.value_in(u.R_earth)
        if mass is None or radius is None or radius <= 0:
            return unknown(u.g / u.cm**3, provenance="3M / 4 pi R^3")
        volume = (4.0 / 3.0) * np.pi * (radius * R_EARTH) ** 3
        return derived(
            float(((mass * M_EARTH) / volume).to_value(u.g / u.cm**3)),
            u.g / u.cm**3,
            provenance="3M / (4 pi R^3)",
        )

    @property
    def kepler_residual(self) -> float | None:
        """Fractional a-vs-P disagreement (roadmap section 8.7)."""
        published = self.extra.get("published_semimajor_axis")
        if published is None or not published.is_known:
            return None
        return kepler_third_law_residual(
            self.elements.period, published, self.host.mass, self._planet_mass_solar()
        )

    def _planet_mass_solar(self) -> float:
        mass = self.mass_earth.value_in(u.M_sun)
        return 0.0 if mass is None else mass

    # -- presentation ----------------------------------------------------
    def describe(self) -> list[str]:
        """Information-panel lines that never hide a substitution."""
        lines = [
            "Planet:            {0}".format(self.name),
            "Host star:         {0}".format(self.host.name),
            "Discovery:         {0} ({1})".format(
                self.discovery_method or "unknown",
                int(round(self.discovery_year.value)) if self.discovery_year.is_known else "unknown",
            ),
            "Facility:          {0}".format(self.discovery_facility or "unknown"),
            "",
            "Orbital period:    {0}".format(self.elements.period.format()),
            "Semi-major axis:   {0}".format(self.semimajor_axis.format()),
            "Eccentricity:      {0}".format(self.elements.eccentricity.format()),
            "Periapsis:         {0}".format(self.elements.periapsis.format()),
            "Apoapsis:          {0}".format(self.elements.apoapsis.format()),
        ]
        lines.extend("  " + line for line in self.elements.describe_orientation())

        residual = self.kepler_residual
        if residual is not None:
            lines.append(
                "Kepler 3 residual: {0:+.2%} (published a vs a derived from P and M*)".format(residual)
            )

        lines += [
            "",
            "Radius:            {0}".format(self.radius_earth.format()),
            "Mass:              {0}".format(self.mass_earth.format()),
            "Bulk density:      {0}".format(self.bulk_density.format()),
            "Surface gravity:   {0}".format(self.surface_gravity.format()),
            "Equilibrium temp.: {0}".format(self.equilibrium_temperature.format()),
            "Insolation:        {0}".format(self.insolation.format()),
            "",
        ]
        lines.extend(self.host.describe())
        lines += [
            "",
            "Data source:       {0}".format(self.solution_policy.label),
            "                   {0}".format(self.solution_policy.caveat),
        ]
        return lines


def build_planet_record(
    row: Mapping[str, Any],
    *,
    policy: SolutionPolicy = SolutionPolicy.COMPOSITE,
    retrieved: date | None = None,
) -> PlanetRecord:
    """Convert one catalogue row into a :class:`PlanetRecord`.

    The semimajor axis policy (roadmap section 3.3) is applied here: use the
    published value when there is one; otherwise derive it from the period
    and the stellar mass and label it DERIVED; otherwise leave it UNKNOWN.
    """
    table = policy.table
    # Planetary and stellar parameters can come from different papers even
    # within one default solution, so each group keeps its own citation.
    reference = clean_reference(row.get("pl_refname"))
    stellar_reference = clean_reference(row.get("st_refname")) or reference
    planet_url = reference_url(row.get("pl_refname"))

    def column(name: str, unit: u.UnitBase) -> Parameter:
        return _param_from_row(row, name, unit, table=table, reference=reference, retrieved=retrieved)

    def stellar_column(name: str, unit: u.UnitBase) -> Parameter:
        return _param_from_row(
            row, name, unit, table=table, reference=stellar_reference, retrieved=retrieved
        )

    host_name = clean_text(row.get("hostname"))
    teff = stellar_column("st_teff", u.K)
    st_radius = stellar_column("st_rad", u.R_sun)
    st_mass = stellar_column("st_mass", u.M_sun)

    # st_lum is log10(L/L_sun) in the archive.
    log_lum = parse_float(row.get("st_lum"))
    if math.isfinite(log_lum):
        luminosity = measured(
            10.0**log_lum,
            u.L_sun,
            provenance="{0}.st_lum (10^log L)".format(table),
            reference=stellar_reference,
            retrieved=retrieved,
        )
    else:
        luminosity = luminosity_from_radius_and_teff(st_radius, teff)

    star = StarRecord(
        name=host_name,
        effective_temperature=teff,
        radius=st_radius,
        mass=st_mass,
        luminosity=luminosity,
        metallicity=stellar_column("st_met", u.dimensionless_unscaled),
        age=stellar_column("st_age", u.Gyr),
        spectral_type=clean_text(row.get("st_spectype")),
        position=sky_position(
            host_name,
            parse_float(row.get("ra")),
            parse_float(row.get("dec")),
            parallax_mas=_param_from_row(row, "sy_plx", u.mas, table=table, retrieved=retrieved),
            catalog_distance_pc=_param_from_row(
                row, "sy_dist", u.pc, table=table, retrieved=retrieved
            ),
        ),
        source_table=table,
        reference=stellar_reference,
    )

    period = column("pl_orbper", u.day)
    mass_earth = column("pl_bmasse", u.M_earth)
    planet_mass_solar = mass_earth.value_in(u.M_sun) or 0.0

    published_axis = column("pl_orbsmax", u.au)
    if published_axis.is_known:
        axis = published_axis
    else:
        # Roadmap 3.3: derive, or stay unknown. Never 1 AU.
        axis = semimajor_axis_from_period(period, st_mass, planet_mass_solar)

    # Archive angles are in degrees; convert once, here.
    inclination = column("pl_orbincl", u.deg)
    arg_periastron = column("pl_orblper", u.deg)

    elements = OrbitalElements(
        name=clean_text(row.get("pl_name")),
        semimajor_axis=axis,
        # Roadmap 3.4: an absent eccentricity stays UNKNOWN here.
        eccentricity=column("pl_orbeccen", u.dimensionless_unscaled),
        period=period,
        inclination=inclination.to(u.rad) if inclination.is_known else unknown(u.rad),
        argument_of_periastron=(
            arg_periastron.to(u.rad) if arg_periastron.is_known else unknown(u.rad)
        ),
        # The archive publishes no longitude of the ascending node for
        # exoplanets; it is not observable from transits or radial velocity.
        longitude_of_ascending_node=unknown(
            u.rad, provenance="not published", note="not observable for most exoplanets"
        ),
        epoch_periastron=column("pl_orbtper", u.day),
        epoch_transit=column("pl_tranmid", u.day),
        # Review section 10: the archive preserves the source publication's
        # convention and carries no column saying which it is.  Claiming
        # PLANET here would be inventing metadata, so the honest value is
        # AS_REPORTED and the ambiguity travels with the element.
        periastron_convention=(
            PeriastronConvention.AS_REPORTED
            if arg_periastron.is_known
            else PeriastronConvention.UNKNOWN
        ),
        # pl_orbtper and pl_tranmid are Julian days; the archive does not
        # publish a machine-readable time scale for them.
        epoch_scale=TimeScale.JD_UNSPECIFIED,
        reference=reference,
    )

    return PlanetRecord(
        name=clean_text(row.get("pl_name")),
        host=star,
        elements=elements,
        radius_earth=column("pl_rade", u.R_earth),
        mass_earth=mass_earth,
        density=column("pl_dens", u.g / u.cm**3),
        equilibrium_temperature_published=column("pl_eqt", u.K),
        insolation_published=column("pl_insol", u.dimensionless_unscaled),
        mass_provenance=clean_text(row.get("pl_bmassprov")),
        discovery_method=clean_text(row.get("discoverymethod")),
        discovery_year=column("disc_year", u.yr),
        discovery_facility=clean_text(row.get("disc_facility")),
        solution_policy=policy,
        source_table=table,
        reference=reference,
        extra={
            "published_semimajor_axis": published_axis,
            "reference_url": planet_url,
            "stellar_reference": stellar_reference,
        },
    )
