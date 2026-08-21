"""The :class:`Spectrum` object (roadmap section 13).

An atmospheric spectrum is not "all points for a planet".  It is one
measurement, by one instrument, on one facility, published in one paper.
Merging two of them because they share a planet name destroys the science.
:class:`Spectrum` therefore keeps its metadata and refuses to concatenate
with a spectrum from a different origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import numpy as np

__all__ = ["Spectrum", "SpectrumCollection"]


@dataclass(frozen=True)
class Spectrum:
    """One atmospheric spectrum with its full provenance.

    Attributes
    ----------
    wavelength:
        Band centres, as a :class:`~astropy.units.Quantity`.
    bandwidth:
        Band widths; the x error bar, not the y value.  Confusing these two
        was the original parser bug (roadmap section 3.1).
    value:
        The measured quantity, usually transit depth.
    error_plus, error_minus:
        Asymmetric uncertainties, stored as non-negative magnitudes.
    """

    planet: str
    spectrum_type: str
    facility: str
    instrument: str
    wavelength: u.Quantity
    value: u.Quantity
    bandwidth: u.Quantity | None = None
    error_plus: u.Quantity | None = None
    error_minus: u.Quantity | None = None
    reference: str = ""
    source_file: str = ""
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.size(self.wavelength))

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def origin_key(self) -> tuple[str, str, str, str, str]:
        """Identity of the measurement, used to keep spectra separate."""
        return (
            self.planet,
            self.spectrum_type,
            self.facility,
            self.instrument,
            self.reference,
        )

    @property
    def label(self) -> str:
        """Legend label naming the instrument and publication."""
        parts = [p for p in (self.instrument, self.facility) if p and p.lower() != "none"]
        origin = " / ".join(parts) if parts else "unknown instrument"
        if self.reference and self.reference.lower() != "none":
            return "{0} ({1})".format(self.reference, origin)
        return "{0} - {1}".format(origin, self.source_file or "local file")

    @property
    def has_errors(self) -> bool:
        return self.error_plus is not None or self.error_minus is not None

    def yerr_array(self) -> np.ndarray | None:
        """``(2, N)`` array for matplotlib ``errorbar``, lower row first."""
        if not self.has_errors:
            return None
        unit = self.value.unit
        zeros = np.zeros(len(self))
        minus = zeros if self.error_minus is None else np.abs(self.error_minus.to_value(unit))
        plus = zeros if self.error_plus is None else np.abs(self.error_plus.to_value(unit))
        return np.vstack(
            [
                np.where(np.isfinite(minus), minus, 0.0),
                np.where(np.isfinite(plus), plus, 0.0),
            ]
        )

    def sorted_by_wavelength(self) -> "Spectrum":
        """A copy ordered by wavelength, for line plots."""
        order = np.argsort(self.wavelength.value)

        def reorder(quantity):
            return None if quantity is None else quantity[order]

        return Spectrum(
            planet=self.planet,
            spectrum_type=self.spectrum_type,
            facility=self.facility,
            instrument=self.instrument,
            wavelength=self.wavelength[order],
            value=self.value[order],
            bandwidth=reorder(self.bandwidth),
            error_plus=reorder(self.error_plus),
            error_minus=reorder(self.error_minus),
            reference=self.reference,
            source_file=self.source_file,
            note=self.note,
            metadata=dict(self.metadata),
        )

    def describe(self) -> list[str]:
        """Provenance lines for the information panel."""
        return [
            "Planet:     {0}".format(self.planet or "unknown"),
            "Type:       {0}".format(self.spectrum_type or "unknown"),
            "Facility:   {0}".format(self.facility or "unknown"),
            "Instrument: {0}".format(self.instrument or "unknown"),
            "Reference:  {0}".format(self.reference or "unknown"),
            "Points:     {0}".format(len(self)),
            "File:       {0}".format(self.source_file or "unknown"),
        ]


@dataclass
class SpectrumCollection:
    """Several spectra for one planet, deliberately kept apart.

    Roadmap section 13: the user may choose to overlay spectra, but the
    application must never silently merge instruments, facilities, epochs,
    publications or reductions into a single curve.
    """

    planet: str
    spectra: list[Spectrum] = field(default_factory=list)

    def add(self, spectrum: Spectrum) -> None:
        if spectrum.is_empty:
            return
        self.spectra.append(spectrum)

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self):
        return iter(self.spectra)

    @property
    def is_empty(self) -> bool:
        return not self.spectra

    def origins(self) -> list[tuple[str, str, str, str, str]]:
        """Distinct measurement origins present in the collection."""
        seen: list[tuple[str, str, str, str, str]] = []
        for spectrum in self.spectra:
            if spectrum.origin_key not in seen:
                seen.append(spectrum.origin_key)
        return seen

    def by_type(self, spectrum_type: str) -> "SpectrumCollection":
        return SpectrumCollection(
            planet=self.planet,
            spectra=[s for s in self.spectra if s.spectrum_type == spectrum_type],
        )

    def concatenated(self) -> None:
        """Deliberately unavailable.

        Merging spectra from different instruments into one array is the
        anti-pattern roadmap section 13 forbids.  Iterate and plot each
        spectrum separately instead.
        """
        raise NotImplementedError(
            "spectra from different instruments, facilities, epochs or "
            "publications must not be merged; iterate the collection instead"
        )
