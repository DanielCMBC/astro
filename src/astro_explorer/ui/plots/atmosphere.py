"""Atmospheric spectra plot (roadmap sections 3.1 and 13).

Each spectrum is drawn as its own series with its own legend entry naming
the instrument, facility and publication.  Nothing is merged, so a Hubble
WFC3 measurement and a JWST NIRSpec measurement of the same planet remain
visibly distinct observations.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

__all__ = ["draw_spectra", "draw_orbit"]

#: Distinct colours for overlaid spectra.
_PALETTE = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#B279A2", "#9D755D", "#72B7B2", "#EECA3B",
)


def draw_spectra(axes, collection, *, bands=(), max_legend_entries: int = 8):
    """Plot every spectrum in a collection as a separate series."""
    axes.clear()

    if collection is None or collection.is_empty:
        planet = getattr(collection, "planet", "this planet")
        axes.text(
            0.5,
            0.5,
            "No local atmospheric spectra found for {0}.".format(planet),
            ha="center",
            va="center",
            color="crimson",
            transform=axes.transAxes,
        )
        axes.set_title("Atmospheric spectra")
        errors = getattr(collection, "errors", None)
        if errors:
            axes.text(
                0.5, 0.35, "\n".join(errors[:3]), ha="center", va="top",
                transform=axes.transAxes, fontsize=7, color="0.4",
            )
        return axes

    all_wavelengths = []
    for index, spectrum in enumerate(collection):
        color = _PALETTE[index % len(_PALETTE)]
        wavelength = spectrum.wavelength.to_value(u.micron)
        values = spectrum.value.value
        all_wavelengths.append(wavelength)

        # The bandwidth is the x error bar, not the signal. Confusing the two
        # was the original parser bug.
        xerr = None
        if spectrum.bandwidth is not None:
            half_width = 0.5 * spectrum.bandwidth.to_value(u.micron)
            if np.any(np.isfinite(half_width)):
                xerr = np.where(np.isfinite(half_width), half_width, 0.0)

        yerr = spectrum.yerr_array()

        axes.errorbar(
            wavelength,
            values,
            yerr=yerr,
            xerr=xerr,
            fmt="o",
            markersize=3.5,
            linewidth=1.0,
            capsize=2,
            alpha=0.85,
            color=color,
            label=spectrum.label,
        )

    combined = np.concatenate(all_wavelengths)
    low, high = float(np.nanmin(combined)), float(np.nanmax(combined))
    span = max(high - low, 1e-3)
    axes.set_xlim(low - 0.05 * span, high + 0.05 * span)

    if bands:
        from ...spectroscopy.normalization import band_overlays

        for band, positions in band_overlays(bands, (low, high)):
            for position in positions:
                axes.axvline(position, color=band.color, alpha=0.22, linewidth=1.0, zorder=0)
            axes.plot([], [], color=band.color, alpha=0.5, label="{0} bands".format(band.molecule))

    first = next(iter(collection))
    axes.set_title(
        "{0} - {1} spectra ({2} measurement{3})".format(
            collection.planet,
            first.spectrum_type or "atmospheric",
            len(collection),
            "" if len(collection) == 1 else "s",
        )
    )
    axes.set_xlabel("Wavelength (microns)")
    axes.set_ylabel("Transit depth ({0})".format(first.value.unit.to_string()))
    axes.grid(True, linestyle=":", alpha=0.5)
    if len(collection) <= max_legend_entries:
        axes.legend(loc="best", fontsize=7)
    axes.text(
        0.99,
        0.01,
        "Measurements are shown separately; instruments and publications are never merged.",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    return axes


def draw_orbit(axes, record, mean_anomaly=None, *, time_label: str = ""):
    """Top-down orbit view with an honest treatment of unknown elements."""
    from ...physics.orbital_elements import (
        position_at_eccentric_anomaly,
        position_at_mean_anomaly,
    )

    axes.clear()
    # The orbit view is the one dark panel; carry the dark ground onto the
    # figure too, or the title and tick labels sit unreadable on white.
    axes.set_facecolor("#05070d")
    if axes.figure is not None:
        axes.figure.set_facecolor("#05070d")

    elements = record.elements
    if not elements.semimajor_axis.is_known:
        axes.text(
            0.5,
            0.5,
            "No semimajor axis published, and none derivable from the period\n"
            "and stellar mass. The orbit is not drawn rather than assumed.",
            ha="center",
            va="center",
            color="#ff8080",
            transform=axes.transAxes,
        )
        axes.set_title(
            "Orbit unavailable for {0}".format(record.name), color="0.85", fontsize=10
        )
        # An empty 0-1 grid would imply there is something to read off it.
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_color("0.25")
        return axes

    display = elements.for_display()
    path = position_at_eccentric_anomaly(display, np.linspace(0.0, 2.0 * np.pi, 720))

    assumed = display.eccentricity.is_assumed or display.semimajor_axis.status.value == "DERIVED"
    axes.plot(
        path[:, 0],
        path[:, 1],
        linestyle="--" if assumed else "-",
        color="#9fb4d8",
        alpha=0.75,
        linewidth=1.2,
    )

    axes.plot(0, 0, "o", color="#ffd257", markersize=13, label=record.host.name)

    # Periapsis marker makes the orientation of an eccentric orbit readable.
    periapsis = position_at_eccentric_anomaly(display, 0.0)
    axes.plot(
        [periapsis[0]], [periapsis[1]], "^", color="#7fd4c1", markersize=6, label="Periapsis"
    )

    subtitle = []
    if mean_anomaly is not None:
        position = position_at_mean_anomaly(display, float(mean_anomaly))
        radius = float(np.linalg.norm(position))
        axes.plot(
            [position[0]], [position[1]], "o", color="#ff6b6b", markersize=9, label=record.name
        )
        subtitle.append("r = {0:.4g} AU".format(radius))
    else:
        subtitle.append("orbital phase not constrained; no current position shown")

    if time_label:
        subtitle.append(time_label)

    limit = float(np.max(np.abs(path[:, :2]))) * 1.2 or 0.02
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.set_xlabel("AU", color="0.75")
    axes.set_ylabel("AU", color="0.75")
    axes.tick_params(colors="0.75")
    for spine in axes.spines.values():
        spine.set_color("0.3")

    axes.set_title(
        "{0} - {1}".format(record.name, " | ".join(subtitle)), color="0.85", fontsize=10
    )
    axes.legend(loc="upper right", fontsize=7, facecolor="#11151f", labelcolor="0.8")

    notes = []
    if display.eccentricity.is_assumed:
        notes.append("Eccentricity unknown; drawn as a circle.")
    if display.semimajor_axis.status.value == "DERIVED":
        notes.append("Semimajor axis derived from period and stellar mass.")
    if notes:
        axes.text(
            0.02,
            0.02,
            "\n".join(notes),
            transform=axes.transAxes,
            fontsize=7,
            color="#ffb86b",
            va="bottom",
        )
    return axes
