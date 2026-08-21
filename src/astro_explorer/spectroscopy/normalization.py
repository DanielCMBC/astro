"""Spectrum normalisation and molecular band overlays.

Roadmap section 13.  Normalisation is offered as an explicit, reversible
*display* operation: two spectra of the same planet taken with different
instruments have different absolute offsets, and subtracting a baseline to
compare their shapes is legitimate as long as the plot says so.

The overlay data comes from ``atmospheric_signatures.json``, which lists
approximate band centres.  These are indicative wavelengths for orientation,
not detections; :func:`band_overlays` labels them accordingly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import astropy.units as u
import numpy as np

from .models import Spectrum

__all__ = [
    "Normalization",
    "normalize",
    "MolecularBand",
    "load_signatures",
    "band_overlays",
]


class Normalization(str, Enum):
    """How a spectrum's y axis has been transformed for display."""

    NONE = "NONE"
    """Raw published values."""

    MEAN_SUBTRACTED = "MEAN_SUBTRACTED"
    """Mean removed; compares shape, discards absolute depth."""

    MEDIAN_SCALED = "MEDIAN_SCALED"
    """Divided by the median; dimensionless relative variation."""

    @property
    def label(self) -> str:
        return {
            Normalization.NONE: "published values",
            Normalization.MEAN_SUBTRACTED: "mean-subtracted (shape only)",
            Normalization.MEDIAN_SCALED: "median-scaled (relative)",
        }[self]

    @property
    def is_display_only(self) -> bool:
        return self is not Normalization.NONE


def normalize(spectrum: Spectrum, mode: Normalization) -> Spectrum:
    """Return a copy of ``spectrum`` with its values transformed.

    Uncertainties are transformed consistently: a shift leaves them alone, a
    scaling divides them by the same factor.  The mode is recorded in the
    returned spectrum's ``note`` so a plot legend cannot lose it.
    """
    if mode is Normalization.NONE:
        return spectrum

    values = spectrum.value.value
    unit = spectrum.value.unit

    if mode is Normalization.MEAN_SUBTRACTED:
        offset = float(np.nanmean(values))
        new_values = (values - offset) * unit
        err_plus, err_minus = spectrum.error_plus, spectrum.error_minus
    else:
        median = float(np.nanmedian(values))
        if not np.isfinite(median) or median == 0.0:
            return spectrum
        new_values = (values / median) * u.dimensionless_unscaled
        err_plus = (
            None
            if spectrum.error_plus is None
            else (spectrum.error_plus.value / median) * u.dimensionless_unscaled
        )
        err_minus = (
            None
            if spectrum.error_minus is None
            else (spectrum.error_minus.value / median) * u.dimensionless_unscaled
        )

    note = "; ".join(part for part in (spectrum.note, mode.label) if part)
    return Spectrum(
        planet=spectrum.planet,
        spectrum_type=spectrum.spectrum_type,
        facility=spectrum.facility,
        instrument=spectrum.instrument,
        wavelength=spectrum.wavelength,
        value=new_values,
        bandwidth=spectrum.bandwidth,
        error_plus=err_plus,
        error_minus=err_minus,
        reference=spectrum.reference,
        source_file=spectrum.source_file,
        note=note,
        metadata=dict(spectrum.metadata),
    )


@dataclass(frozen=True)
class MolecularBand:
    """An indicative absorption band centre for a molecule."""

    molecule: str
    wavelengths: u.Quantity
    color: str = "gray"

    @property
    def caveat(self) -> str:
        return (
            "Indicative band centres for orientation only; overlaying them is "
            "not a detection."
        )


def load_signatures(path) -> list[MolecularBand]:
    """Read ``atmospheric_signatures.json`` into :class:`MolecularBand` objects."""
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    bands: list[MolecularBand] = []
    for molecule, entry in raw.items():
        wavelengths = entry.get("wavelengths") or []
        if not wavelengths:
            continue
        bands.append(
            MolecularBand(
                molecule=molecule,
                wavelengths=np.asarray(wavelengths, dtype=np.float64) * u.micron,
                color=entry.get("color", "gray"),
            )
        )
    return bands


def band_overlays(
    bands: list[MolecularBand],
    wavelength_range: tuple[float, float],
    *,
    unit: u.UnitBase = u.micron,
) -> list[tuple[MolecularBand, np.ndarray]]:
    """Bands that actually fall inside the plotted wavelength range.

    Returns ``(band, wavelengths_in_unit)`` pairs, skipping molecules with no
    line inside the range so the legend stays honest about what is visible.
    """
    low, high = min(wavelength_range), max(wavelength_range)
    visible: list[tuple[MolecularBand, np.ndarray]] = []
    for band in bands:
        values = band.wavelengths.to_value(unit)
        inside = values[(values >= low) & (values <= high)]
        if inside.size:
            visible.append((band, inside))
    return visible
