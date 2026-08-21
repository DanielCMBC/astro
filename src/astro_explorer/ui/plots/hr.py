"""The two stellar diagrams, correctly named (roadmap sections 3.6 and 14.1).

The original program plotted effective temperature against stellar radius
and labelled it an HR diagram.  Both plots are kept; each is drawn by its own
function with its own axes and title, so neither can be mistaken for the
other.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from ...physics.stellar import DiagramKind, luminosity_from_radius_and_teff

__all__ = ["draw_hr_diagram", "draw_temperature_radius_diagram", "population_arrays"]

#: Main-sequence reference points (Teff in K, L/L_sun, R/R_sun), used to draw
#: a guide line so a single selected star has context.
_MAIN_SEQUENCE = (
    (2800.0, 0.0018, 0.18),
    (3300.0, 0.015, 0.32),
    (3800.0, 0.06, 0.50),
    (4400.0, 0.16, 0.70),
    (5200.0, 0.42, 0.85),
    (5772.0, 1.00, 1.00),
    (6200.0, 1.75, 1.15),
    (7000.0, 4.5, 1.40),
    (8500.0, 18.0, 1.80),
    (10000.0, 55.0, 2.40),
)


def population_arrays(catalog):
    """Teff, luminosity and radius arrays for the host-star population.

    Luminosity comes from ``st_lum`` (log10 L/L_sun) when the archive
    publishes it, and is otherwise derived from radius and temperature.
    Hosts are de-duplicated so a system with eight planets is one point.
    """
    if catalog is None or catalog.empty:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    columns = [c for c in ("hostname", "st_teff", "st_rad", "st_lum") if c in catalog.columns]
    frame = catalog[columns].drop_duplicates(subset=["hostname"])

    teff = frame["st_teff"].to_numpy(dtype=float) if "st_teff" in frame else np.array([])
    radius = frame["st_rad"].to_numpy(dtype=float) if "st_rad" in frame else np.full(teff.shape, np.nan)

    if "st_lum" in frame:
        log_lum = frame["st_lum"].to_numpy(dtype=float)
        luminosity = np.where(np.isfinite(log_lum), np.power(10.0, log_lum), np.nan)
    else:
        luminosity = np.full(teff.shape, np.nan)

    # Fill the gaps with the Stefan-Boltzmann derivation.
    needs_derivation = ~np.isfinite(luminosity) & np.isfinite(teff) & np.isfinite(radius)
    if np.any(needs_derivation):
        from ...physics.constants import SOLAR_EFFECTIVE_TEMPERATURE

        solar_teff = float(SOLAR_EFFECTIVE_TEMPERATURE.to_value(u.K))
        luminosity = np.where(
            needs_derivation,
            np.power(radius, 2.0) * np.power(teff / solar_teff, 4.0),
            luminosity,
        )

    return teff, luminosity, radius


#: Temperatures outside this range are almost always bad catalogue entries,
#: and a handful of them otherwise squash the main sequence into a corner.
TEFF_DISPLAY_RANGE = (2000.0, 45000.0)


def _style_axes(axes, kind: DiagramKind, teff=None) -> None:
    axes.set_title(kind.title)
    axes.set_xlabel(kind.x_label)
    axes.set_ylabel(kind.y_label)
    axes.set_yscale("log")
    # Log temperature is the classical HR convention and keeps the M dwarfs
    # legible next to the occasional 40,000 K host.
    axes.set_xscale("log")

    low, high = TEFF_DISPLAY_RANGE
    if teff is not None:
        finite = teff[np.isfinite(teff)]
        if finite.size:
            low = max(low, float(np.nanpercentile(finite, 0.2)) * 0.85)
            high = min(high, float(np.nanpercentile(finite, 99.8)) * 1.15)
    axes.set_xlim(high, low)  # hot stars on the left, as convention requires

    ticks = [t for t in (2500, 3500, 5000, 7000, 10000, 20000, 40000) if low <= t <= high]
    if ticks:
        axes.set_xticks(ticks)
        axes.set_xticklabels([str(t) for t in ticks])
    axes.minorticks_off()
    axes.grid(True, linestyle=":", alpha=0.4)


def draw_hr_diagram(axes, catalog, record=None, *, show_main_sequence: bool = True):
    """Classical HR diagram: luminosity against effective temperature."""
    axes.clear()
    teff, luminosity, _radius = population_arrays(catalog)

    valid = np.isfinite(teff) & np.isfinite(luminosity) & (luminosity > 0)
    if np.any(valid):
        axes.scatter(
            teff[valid],
            luminosity[valid],
            s=4,
            color="gray",
            alpha=0.3,
            label="Known exoplanet hosts ({0})".format(int(valid.sum())),
        )

    if show_main_sequence:
        axes.plot(
            [point[0] for point in _MAIN_SEQUENCE],
            [point[1] for point in _MAIN_SEQUENCE],
            color="steelblue",
            linewidth=1.0,
            alpha=0.7,
            label="Main sequence (reference)",
        )

    if record is not None:
        star_teff = record.host.effective_temperature.value_in(u.K)
        star_lum = record.host.luminosity.value_in(u.L_sun)
        if star_teff is not None and star_lum is not None and star_lum > 0:
            axes.plot(
                [star_teff],
                [star_lum],
                "o",
                markersize=11,
                color="crimson",
                markeredgecolor="black",
                label="{0}: {1:.0f} K, {2:.3g} L_sun{3}".format(
                    record.host.name,
                    star_teff,
                    star_lum,
                    " (derived)" if record.host.luminosity.status.value == "DERIVED" else "",
                ),
            )

    _style_axes(axes, DiagramKind.HR_DIAGRAM, teff)
    axes.legend(loc="best", fontsize=8)
    return axes


def draw_temperature_radius_diagram(axes, catalog, record=None):
    """The original program's plot, under its correct name."""
    axes.clear()
    teff, _luminosity, radius = population_arrays(catalog)

    valid = np.isfinite(teff) & np.isfinite(radius) & (radius > 0)
    if np.any(valid):
        axes.scatter(
            teff[valid],
            radius[valid],
            s=4,
            color="gray",
            alpha=0.3,
            label="Known exoplanet hosts ({0})".format(int(valid.sum())),
        )

    axes.plot(
        [point[0] for point in _MAIN_SEQUENCE],
        [point[2] for point in _MAIN_SEQUENCE],
        color="steelblue",
        linewidth=1.0,
        alpha=0.7,
        label="Main sequence (reference)",
    )

    if record is not None:
        star_teff = record.host.effective_temperature.value_in(u.K)
        star_radius = record.host.radius.value_in(u.R_sun)
        if star_teff is not None and star_radius is not None and star_radius > 0:
            axes.plot(
                [star_teff],
                [star_radius],
                "o",
                markersize=11,
                color="crimson",
                markeredgecolor="black",
                label="{0}: {1:.0f} K, {2:.3g} R_sun".format(
                    record.host.name, star_teff, star_radius
                ),
            )

    _style_axes(axes, DiagramKind.TEMPERATURE_RADIUS, teff)
    axes.legend(loc="best", fontsize=8)
    return axes
