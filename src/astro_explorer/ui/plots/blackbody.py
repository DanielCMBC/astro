"""Blackbody plot, labelled as an approximation (roadmap sections 3.7, 14.2)."""

from __future__ import annotations

import astropy.units as u
import numpy as np

from ...physics.radiation import blackbody_curve

__all__ = ["draw_blackbody"]

#: Rough visible-band edges, drawn as a shaded strip for orientation.
_VISIBLE_NM = (380.0, 750.0)


def draw_blackbody(axes, record=None, *, temperature_k: float | None = None):
    """Plot the Planck curve for a star, with Wien's peak marked.

    The title and an on-axes note both say "ideal blackbody", because a real
    stellar spectrum has absorption lines and atmosphere-dependent structure
    this curve cannot show.
    """
    axes.clear()

    if temperature_k is None and record is not None:
        temperature_k = record.host.effective_temperature.value_in(u.K)

    if temperature_k is None or not np.isfinite(temperature_k) or temperature_k <= 0:
        axes.text(
            0.5,
            0.5,
            "No effective temperature published for this host star.",
            ha="center",
            va="center",
            color="crimson",
            transform=axes.transAxes,
        )
        axes.set_title("Blackbody spectrum")
        return axes

    curve = blackbody_curve(temperature_k * u.K)
    wavelength_nm = curve.wavelength.to_value(u.nm)

    axes.axvspan(*_VISIBLE_NM, color="0.85", alpha=0.35, zorder=0, label="Visible band")
    axes.plot(wavelength_nm, curve.normalized, color="orange", linewidth=2.0, zorder=3)
    axes.fill_between(wavelength_nm, curve.normalized, color="orange", alpha=0.25, zorder=2)

    peak_nm = curve.peak_wavelength.to_value(u.nm)
    axes.axvline(
        peak_nm,
        color="crimson",
        linestyle="--",
        zorder=4,
        label="Wien peak: {0:.1f} nm".format(peak_nm),
    )

    name = record.host.name if record is not None else "star"
    axes.set_title(
        "{0} - {1} (T_eff = {2:.0f} K)".format(name, curve.label, temperature_k)
    )
    axes.set_xlabel("Wavelength (nm)")
    axes.set_ylabel("Relative spectral radiance")
    axes.set_xlim(wavelength_nm.min(), wavelength_nm.max())
    axes.set_ylim(0.0, 1.08)
    axes.grid(True, linestyle=":", alpha=0.5)
    axes.legend(loc="upper right", fontsize=8)

    axes.text(
        0.99,
        0.02,
        curve.caveat,
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
        wrap=True,
    )
    return axes
