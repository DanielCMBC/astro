"""IPAC atmospheric table reader (roadmap section 3.1, P0).

The original parser split each row on whitespace and assumed the first three
fields were wavelength, transit depth and error.  In the NASA atmospheric
tables shipped with this repository the real column order is::

    CENTRALWAVELNG  BANDWIDTH  PL_TRANDEP  PL_TRANDEPERR1  PL_TRANDEPERR2 ...

so the program plotted CENTRALWAVELNG against BANDWIDTH - the band width,
not the signal - and treated the transit depth as the error bar.

This module reads the tables with :func:`astropy.table.Table.read` using the
``ascii.ipac`` format, addresses columns by name, and keeps the header
metadata (planet, spectrum type, instrument, facility, reference) attached to
the resulting :class:`~astro_explorer.spectroscopy.models.Spectrum`.

Positional parsing is never used, not even as a fallback: a table whose
columns cannot be identified by name is rejected rather than guessed at.
"""

from __future__ import annotations

import re
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from .models import Spectrum, SpectrumCollection

__all__ = [
    "read_ipac_spectrum",
    "read_planet_spectra",
    "spectrum_files_for_planet",
    "REQUIRED_COLUMNS",
    "SpectrumParseError",
]


class SpectrumParseError(ValueError):
    """Raised when a table cannot be interpreted without guessing."""


#: Columns the reader needs by name.  Absence is an error, not a fallback.
REQUIRED_COLUMNS = ("CENTRALWAVELNG", "PL_TRANDEP")

#: Optional columns, used when present.
OPTIONAL_COLUMNS = ("BANDWIDTH", "PL_TRANDEPERR1", "PL_TRANDEPERR2")

#: Header keywords carrying provenance.
_METADATA_KEYS = ("PL_NAME", "SPEC_TYPE", "INSTRUMENT", "FACILITY", "REFERENCE", "NOTE")


def _read_header_keywords(path: Path) -> dict[str, str]:
    """Read the backslash-prefixed IPAC keyword block.

    Astropy exposes these through ``table.meta['keywords']``, but reading
    them directly keeps the values available even for tables that fail the
    strict parse, so an error message can still name the planet.
    """
    keywords: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.startswith("\\"):
                if line.startswith("|"):
                    break
                continue
            if "=" not in line:
                continue
            key, value = line[1:].split("=", 1)
            keywords[key.strip().upper()] = value.strip()
    return keywords


def _keyword(keywords: dict[str, str], meta: dict, name: str, default: str = "") -> str:
    value = keywords.get(name)
    if value is None:
        raw = (meta or {}).get("keywords", {}).get(name)
        if isinstance(raw, dict):
            value = raw.get("value")
        elif raw is not None:
            value = raw
    if value is None:
        return default
    text = str(value).strip()
    return "" if text.lower() in ("none", "null", "") else text


def _column_unit(table: Table, name: str, fallback: u.UnitBase) -> u.UnitBase:
    """Unit declared in the table header, or ``fallback``.

    IPAC tables write units such as ``microns`` and ``%`` in the third
    header row; honouring them means the reader does not assume microns.
    """
    column = table[name]
    unit = getattr(column, "unit", None)
    if unit is None:
        return fallback
    try:
        return u.Unit(str(unit))
    except (ValueError, TypeError):
        return fallback


def _masked_to_nan(table: Table, name: str) -> np.ndarray:
    """Column as float64 with masked/null entries turned into NaN."""
    column = table[name]
    data = np.asarray(column.filled(np.nan) if hasattr(column, "filled") else column)
    return data.astype(np.float64)


def read_ipac_spectrum(path) -> Spectrum:
    """Read one ``.tbl`` atmospheric table into a :class:`Spectrum`.

    Raises
    ------
    SpectrumParseError
        If the file is not a readable IPAC table, or does not contain the
        columns this reader needs by name.
    """
    path = Path(path)
    keywords = _read_header_keywords(path)

    try:
        table = Table.read(str(path), format="ascii.ipac")
    except Exception as exc:  # astropy raises a variety of parse errors
        raise SpectrumParseError("{0}: not a readable IPAC table ({1})".format(path.name, exc)) from exc

    missing = [name for name in REQUIRED_COLUMNS if name not in table.colnames]
    if missing:
        raise SpectrumParseError(
            "{0}: missing required column(s) {1}; refusing to guess column "
            "positions".format(path.name, ", ".join(missing))
        )

    wavelength_unit = _column_unit(table, "CENTRALWAVELNG", u.micron)
    depth_unit = _column_unit(table, "PL_TRANDEP", u.percent)

    wavelength = _masked_to_nan(table, "CENTRALWAVELNG")
    depth = _masked_to_nan(table, "PL_TRANDEP")

    def optional(name: str, unit: u.UnitBase):
        if name not in table.colnames:
            return None
        return _masked_to_nan(table, name) * unit

    bandwidth = optional("BANDWIDTH", wavelength_unit)
    err_plus = optional("PL_TRANDEPERR1", depth_unit)
    err_minus = optional("PL_TRANDEPERR2", depth_unit)

    # Keep only rows where both axes are real numbers.  Rows are dropped
    # together so the arrays stay aligned.
    keep = np.isfinite(wavelength) & np.isfinite(depth)

    def apply(quantity):
        if quantity is None:
            return None
        return np.abs(quantity[keep])

    return Spectrum(
        planet=_keyword(keywords, table.meta, "PL_NAME"),
        spectrum_type=_keyword(keywords, table.meta, "SPEC_TYPE"),
        facility=_keyword(keywords, table.meta, "FACILITY"),
        instrument=_keyword(keywords, table.meta, "INSTRUMENT"),
        wavelength=wavelength[keep] * wavelength_unit,
        value=depth[keep] * depth_unit,
        bandwidth=None if bandwidth is None else bandwidth[keep],
        error_plus=apply(err_plus),
        error_minus=apply(err_minus),
        reference=_keyword(keywords, table.meta, "REFERENCE"),
        source_file=path.name,
        note=_keyword(keywords, table.meta, "NOTE"),
        metadata={key: keywords[key] for key in _METADATA_KEYS if key in keywords},
    )


def _filename_stub(planet_name: str) -> str:
    """The filename prefix NASA uses for a planet, e.g. ``55 Cnc e`` -> ``55_Cnc_e``."""
    return re.sub(r"[ \-]", "_", planet_name.strip())


def spectrum_files_for_planet(planet_name: str, directories) -> list[Path]:
    """Local ``.tbl`` files belonging to ``planet_name``.

    Matching is anchored on the ``<stub>_<number>`` pattern so that
    ``K2-18 b`` does not also pick up ``K2-180 b``.
    """
    stub = _filename_stub(planet_name)
    pattern = re.compile(r"^{0}_[0-9]".format(re.escape(stub)))

    found: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue
        for candidate in sorted(directory.glob("{0}*.tbl".format(stub))):
            if not pattern.match(candidate.name):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def read_planet_spectra(planet_name: str, directories) -> SpectrumCollection:
    """Read every local spectrum for a planet, keeping them separate.

    Unreadable files are skipped and reported through the collection's
    ``errors`` attribute rather than raising, so one malformed table cannot
    hide every other measurement for that planet.
    """
    collection = SpectrumCollection(planet=planet_name)
    errors: list[str] = []

    for path in spectrum_files_for_planet(planet_name, directories):
        try:
            collection.add(read_ipac_spectrum(path))
        except SpectrumParseError as exc:
            errors.append(str(exc))

    setattr(collection, "errors", errors)
    return collection
