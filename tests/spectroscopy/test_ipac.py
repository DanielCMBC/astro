"""Spectroscopy tests (roadmap section 22, "Spectroscopy").

These are the regression guard for the P0 correction 3.1: the original
parser split rows on whitespace and assumed positions 0, 1, 2 were
wavelength, depth and error, so it plotted BANDWIDTH as the signal.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from astro_explorer.spectroscopy.ipac import (
    REQUIRED_COLUMNS,
    SpectrumParseError,
    read_ipac_spectrum,
    read_planet_spectra,
    spectrum_files_for_planet,
)
from astro_explorer.spectroscopy.models import Spectrum, SpectrumCollection
from astro_explorer.spectroscopy.normalization import Normalization, band_overlays, normalize

TABLES = Path(__file__).resolve().parents[2] / "tables"
SAMPLE = TABLES / "55_Cnc_e_3.10924_3673_1.tbl"

pytestmark = pytest.mark.skipif(not TABLES.is_dir(), reason="local tables directory absent")


# -- column mapping ----------------------------------------------------------


def test_columns_are_addressed_by_name():
    assert REQUIRED_COLUMNS == ("CENTRALWAVELNG", "PL_TRANDEP")


def test_transit_depth_is_not_the_bandwidth():
    """The exact bug from roadmap 3.1."""
    spectrum = read_ipac_spectrum(SAMPLE)
    depth = spectrum.value.value
    bandwidth = spectrum.bandwidth.value
    assert not np.allclose(depth, bandwidth)
    # The first row of the sample file is depth 0.0324, bandwidth 0.029.
    assert np.isclose(depth[0], 0.0324, rtol=1e-6)
    assert np.isclose(bandwidth[0], 0.029, rtol=1e-6)


def test_error_is_not_the_transit_depth():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert np.isclose(spectrum.error_plus.value[0], 0.00602, rtol=1e-6)
    assert not np.allclose(spectrum.error_plus.value, spectrum.value.value)


def test_units_come_from_the_table_header():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert spectrum.wavelength.unit == u.micron
    assert spectrum.value.unit == u.percent


def test_asymmetric_uncertainties_are_retained_and_positive():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert spectrum.error_plus is not None
    assert spectrum.error_minus is not None
    # PL_TRANDEPERR2 is negative in the file; stored magnitudes are positive.
    assert np.all(spectrum.error_minus.value >= 0)
    # ...and genuinely asymmetric, not a copy of the upper error.
    assert not np.allclose(spectrum.error_plus.value, spectrum.error_minus.value)


def test_yerr_array_is_lower_row_first():
    spectrum = read_ipac_spectrum(SAMPLE)
    yerr = spectrum.yerr_array()
    assert yerr.shape == (2, len(spectrum))
    assert np.allclose(yerr[0], spectrum.error_minus.value)
    assert np.allclose(yerr[1], spectrum.error_plus.value)


# -- metadata ----------------------------------------------------------------


def test_metadata_is_extracted():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert spectrum.planet == "55 Cnc e"
    assert spectrum.spectrum_type == "Transmission"
    assert spectrum.instrument == "ALFOSC"
    assert spectrum.facility == "Nordic Optical Telescope"
    assert spectrum.reference == "de Mooij et al. 2014"


def test_label_names_the_instrument_and_publication():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert "de Mooij" in spectrum.label
    assert "ALFOSC" in spectrum.label


def test_note_of_none_is_not_shown_as_the_string_none():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert spectrum.note == ""


# -- keeping spectra separate ------------------------------------------------


def test_multiple_spectra_per_planet_stay_separate():
    """Roadmap 3.1 and 13: never merge two measurements."""
    collection = read_planet_spectra("55 Cnc e", [TABLES])
    assert len(collection) >= 3
    assert len(collection.origins()) == len(collection)

    instruments = {spectrum.instrument for spectrum in collection}
    assert len(instruments) > 1


def test_merging_is_deliberately_unavailable():
    collection = read_planet_spectra("55 Cnc e", [TABLES])
    with pytest.raises(NotImplementedError, match="must not be merged"):
        collection.concatenated()


def test_filename_matching_is_anchored():
    """K2-18 b must not pick up K2-180 b."""
    matches = spectrum_files_for_planet("K2-18 b", [TABLES])
    assert matches
    for path in matches:
        assert path.name.startswith("K2_18_b_")


def test_every_bundled_table_parses():
    """A corpus check: no file may need positional guessing."""
    failures = []
    for path in sorted(TABLES.glob("*.tbl")):
        try:
            read_ipac_spectrum(path)
        except SpectrumParseError as exc:
            failures.append(str(exc))
    assert not failures, failures[:5]


def test_a_well_formed_table_with_wrong_columns_is_rejected(tmp_path):
    """A perfectly valid IPAC table whose columns we cannot identify by name.

    The original parser would have happily read columns 0 and 1 as
    wavelength and depth.  This one refuses.
    """
    from astropy.table import Table

    bad = tmp_path / "Bad_1.tbl"
    Table({"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]}).write(
        str(bad), format="ascii.ipac", overwrite=True
    )
    with pytest.raises(SpectrumParseError, match="missing required column"):
        read_ipac_spectrum(bad)


def test_an_unparseable_file_is_rejected_too(tmp_path):
    bad = tmp_path / "Broken_1.tbl"
    bad.write_text("this is not a table at all\n", encoding="utf-8")
    with pytest.raises(SpectrumParseError):
        read_ipac_spectrum(bad)


def test_unreadable_files_do_not_hide_the_readable_ones(tmp_path):
    bad = tmp_path / "Zz_1.tbl"
    bad.write_text("not a table at all\n", encoding="utf-8")
    collection = read_planet_spectra("Zz", [tmp_path])
    assert collection.is_empty
    assert getattr(collection, "errors")


# -- normalisation -----------------------------------------------------------


def test_normalisation_is_a_labelled_display_operation():
    spectrum = read_ipac_spectrum(SAMPLE)
    shifted = normalize(spectrum, Normalization.MEAN_SUBTRACTED)
    assert np.isclose(np.nanmean(shifted.value.value), 0.0, atol=1e-12)
    assert "mean-subtracted" in shifted.note
    # Shifting must not change the uncertainties.
    assert np.allclose(shifted.error_plus.value, spectrum.error_plus.value)


def test_median_scaling_divides_the_uncertainties_too():
    spectrum = read_ipac_spectrum(SAMPLE)
    scaled = normalize(spectrum, Normalization.MEDIAN_SCALED)
    median = float(np.nanmedian(spectrum.value.value))
    assert np.isclose(np.nanmedian(scaled.value.value), 1.0)
    assert np.allclose(scaled.error_plus.value, spectrum.error_plus.value / median)


def test_no_normalisation_returns_the_same_object():
    spectrum = read_ipac_spectrum(SAMPLE)
    assert normalize(spectrum, Normalization.NONE) is spectrum


def test_band_overlays_only_report_visible_lines():
    from astro_explorer.spectroscopy.normalization import MolecularBand

    bands = [
        MolecularBand("H2O", np.array([1.4, 1.9]) * u.micron),
        MolecularBand("CO2", np.array([15.0]) * u.micron),
    ]
    visible = band_overlays(bands, (0.5, 5.0))
    assert len(visible) == 1
    assert visible[0][0].molecule == "H2O"


# -- the Spectrum type -------------------------------------------------------


def test_empty_spectra_are_not_added_to_a_collection():
    collection = SpectrumCollection(planet="X")
    collection.add(
        Spectrum(
            planet="X",
            spectrum_type="Transmission",
            facility="",
            instrument="",
            wavelength=np.array([]) * u.micron,
            value=np.array([]) * u.percent,
        )
    )
    assert collection.is_empty


def test_sorting_keeps_arrays_aligned():
    spectrum = read_ipac_spectrum(SAMPLE)
    ordered = spectrum.sorted_by_wavelength()
    assert np.all(np.diff(ordered.wavelength.value) >= 0)
    assert len(ordered) == len(spectrum)
    # The pairing of wavelength to depth must survive the sort.
    original = dict(zip(spectrum.wavelength.value, spectrum.value.value))
    for wavelength, value in zip(ordered.wavelength.value, ordered.value.value):
        assert np.isclose(original[wavelength], value)
