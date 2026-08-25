"""The two stellar diagrams, correctly named (roadmap sections 3.6 and 14.1).

The original program plotted effective temperature against stellar radius
and labelled it an HR diagram.  Both plots are kept; each is drawn by its own
function with its own axes and title, so neither can be mistaken for the
other.

Explorer C4a moved the science out of this module. The selected host's place
on the HR diagram now comes from
:func:`~astro_explorer.physics.hr_diagram.hr_placement`, which returns a
:class:`~astro_explorer.physics.hr_diagram.HRPlacement` carrying the star's
own :class:`~astro_explorer.provenance.Parameter` objects - the same ones the
info panel shows. This module draws it.

That is the same golden rule the renderer follows, applied to a plot for the
same reason: a figure that recomputes a scientific value is a second opinion
that can drift from the first, and the reader has no way to tell which one
they are looking at.

The background population still comes from the catalogue frame in bulk -
thousands of rows cannot each afford a ``Parameter`` - but it is a
:class:`~astro_explorer.physics.hr_diagram.HRPopulation` rather than three
bare numeric arrays, so each point keeps the provenance of its luminosity.
The archive publishes ``st_lum`` for some hosts and a radius and temperature
for the rest, and drawing both as one cloud would assert that every dot is
the same kind of thing. They are drawn with different *markers*, because a
scientific distinction carried by colour alone is lost to a greyscale print
or a colour-blind reader.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from ...physics.hr_diagram import (
    MAIN_SEQUENCE_GUIDE,
    MAIN_SEQUENCE_GUIDE_LABEL,
    HRPlacement,
    HRPopulation,
    hr_placement,
    hr_population,
)
from ...physics.stellar import DiagramKind
from ...provenance import Status

__all__ = [
    "draw_hr_diagram",
    "draw_temperature_radius_diagram",
    "population_arrays",
    "population_for",
    "placement_for",
]

def population_for(catalog) -> HRPopulation:
    """The host-star background, with each point's luminosity provenance.

    This module's share of the work is dataframe plumbing: pull the columns
    out, de-duplicate so a system with eight planets is one point, and hand
    plain arrays to
    :func:`~astro_explorer.physics.hr_diagram.hr_population`, which decides
    what each luminosity *is*. The tiering is a scientific judgement and
    lives with the science.
    """
    if catalog is None or catalog.empty:
        empty = np.array([], dtype=float)
        return hr_population(np.array([], dtype=object), empty, empty, empty)

    columns = [c for c in ("hostname", "st_teff", "st_rad", "st_lum") if c in catalog.columns]
    frame = catalog[columns].drop_duplicates(subset=["hostname"])

    def column(name, dtype=float):
        if name in frame:
            return frame[name].to_numpy(dtype=dtype)
        return np.full(len(frame), np.nan if dtype is float else None, dtype=dtype)

    return hr_population(
        column("hostname", object),
        column("st_teff"),
        column("st_rad"),
        column("st_lum"),
    )


def population_arrays(catalog):
    """Teff, luminosity and radius arrays for the host-star population.

    Kept for callers that only want the numbers. Anything that *draws* the
    population should use :func:`population_for` instead: these three arrays
    have had the per-point provenance flattened out of them, so a figure
    built from them cannot tell a published luminosity from a derived one.
    """
    population = population_for(catalog)
    return (
        population.effective_temperature_k,
        population.luminosity_solar,
        population.radius_solar,
    )


def placement_for(record) -> HRPlacement | None:
    """The selected host's HR placement, from its own parameters.

    One line, and it is the whole crossing point: the plot asks the physics
    layer where the star goes and is told, rather than reading two numbers
    off the record and deciding for itself.
    """
    if record is None:
        return None
    star = record.host
    return hr_placement(star.name, star.effective_temperature, star.luminosity)


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


#: How each luminosity provenance is drawn in the background scatter.
#:
#: The two differ by **marker shape**, not by colour alone: colour is the
#: first thing lost to a colour-blind reader, a greyscale print or a
#: projector, and the distinction being carried here is a scientific one
#: rather than decoration. The same reason the orientation overlay uses a
#: dash pattern rather than a hue.
POPULATION_STYLES = {
    Status.MEASURED: {
        "marker": "o",
        "label": "hosts with a published luminosity",
        "alpha": 0.35,
        "s": 5,
    },
    Status.DERIVED: {
        "marker": "x",
        "label": "hosts with a derived luminosity (radius and Teff)",
        "alpha": 0.30,
        "s": 7,
        "linewidths": 0.6,
    },
}


def draw_hr_diagram(axes, catalog, record=None, *, show_main_sequence: bool = True):
    """Classical HR diagram: luminosity against effective temperature."""
    axes.clear()
    population = population_for(catalog)
    teff = population.effective_temperature_k

    # Published and derived luminosities are drawn separately. Merging them
    # into one cloud would say every dot is the same kind of thing, which is
    # the claim the selected marker itself was making before C4a.
    for status, style in POPULATION_STYLES.items():
        mask = population.with_status(status)
        if not np.any(mask):
            continue
        options = dict(style)
        label = options.pop("label")
        axes.scatter(
            population.effective_temperature_k[mask],
            population.luminosity_solar[mask],
            color="gray",
            label="{0} ({1})".format(label, int(mask.sum())),
            **options,
        )

    if show_main_sequence:
        axes.plot(
            [point[0] for point in MAIN_SEQUENCE_GUIDE],
            [point[1] for point in MAIN_SEQUENCE_GUIDE],
            color="steelblue",
            linewidth=1.0,
            alpha=0.7,
            linestyle="--",
            label=MAIN_SEQUENCE_GUIDE_LABEL,
        )

    placement = placement_for(record)
    if placement is not None:
        if placement.is_plottable:
            axes.plot(
                [placement.teff_k],
                [placement.luminosity_solar],
                "o",
                markersize=11,
                # A placement built on an assumption is drawn hollow, the
                # same distinction the orientation overlay makes with a
                # dashed guide: a filled marker is a measurement.
                color="crimson" if placement.is_scientific else "none",
                markeredgecolor="black" if placement.is_scientific else "crimson",
                markeredgewidth=1.0 if placement.is_scientific else 1.8,
                label=placement.label(),
            )
        else:
            # The old code omitted an unplaceable star in silence, which
            # reads as "not interesting" rather than "not measured". The
            # reasons go on the figure instead.
            axes.annotate(
                "{0}\n{1}".format(placement.name, "\n".join(placement.blockers)),
                xy=(0.03, 0.03),
                xycoords="axes fraction",
                fontsize=7,
                color="crimson",
                wrap=True,
                verticalalignment="bottom",
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
        [point[0] for point in MAIN_SEQUENCE_GUIDE],
        [point[2] for point in MAIN_SEQUENCE_GUIDE],
        color="steelblue",
        linewidth=1.0,
        alpha=0.7,
        linestyle="--",
        label=MAIN_SEQUENCE_GUIDE_LABEL,
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
