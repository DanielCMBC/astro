"""Classification tests (roadmap section 18).

The scheme is explicitly provisional, so these tests check that it *stays*
honest - versioned, multidimensional, and unwilling to look confident about
a planet nobody has characterised - rather than pinning boundary values as
though they were settled.
"""

from __future__ import annotations

import astropy.units as u

from astro_explorer.classification.physical_vector import (
    SCHEME_VERSION,
    AtmosphericEvidence,
    Confidence,
    Irradiation,
    OrbitalRegime,
    Structure,
    Thermal,
    classify_vector,
)
from astro_explorer.classification.traditional import SizeClass, ThermalClass, classify
from astro_explorer.provenance import measured, unknown
from astro_explorer.spectroscopy.molecular_evidence import DetectionStatus, load_evidence


# -- traditional scheme ------------------------------------------------------


def test_earth_is_terrestrial():
    result = classify(1.0, 255.0)
    assert result.size is SizeClass.TERRESTRIAL
    assert result.thermal is ThermalClass.TEMPERATE
    assert result.label == "Terrestrial"


def test_hot_jupiter():
    result = classify(11.2, 1400.0)
    assert result.size is SizeClass.GAS_GIANT
    assert result.thermal is ThermalClass.HOT
    assert "Hot gas giant" == result.label


def test_classification_records_its_basis():
    result = classify(11.2, 1400.0)
    assert "radius" in result.basis
    assert "T_eq" in result.basis


def test_unknown_radius_is_not_quietly_promoted():
    result = classify(None, 300.0)
    assert result.size is SizeClass.UNKNOWN
    assert result.label == "Unknown"


def test_mass_fallback_says_the_radius_was_unknown():
    result = classify(None, 1500.0, mass_jupiter=1.5)
    assert result.size is SizeClass.GAS_GIANT
    assert "radius unknown" in result.basis


def test_classification_accepts_provenance_parameters():
    result = classify(
        measured(1.0, u.R_earth), measured(255.0, u.K), unknown(u.M_jup)
    )
    assert result.size is SizeClass.TERRESTRIAL


# -- the multidimensional vector --------------------------------------------


def test_the_vector_is_not_a_single_scalar():
    vector = classify_vector(
        radius_earth=11.2,
        equilibrium_temperature=1400.0,
        period_days=4.05,
        eccentricity=0.0,
        insolation_earth=800.0,
        atmosphere=AtmosphericEvidence.DETECTED,
    )
    assert vector.code.count("-") == 5
    assert vector.short_code == "G-H-D"  # the roadmap's own example


def test_every_dimension_degrades_independently():
    vector = classify_vector(radius_earth=11.2)
    assert vector.structure is Structure.GIANT
    assert vector.thermal is Thermal.UNKNOWN
    assert vector.orbital is OrbitalRegime.UNKNOWN
    assert vector.irradiation is Irradiation.UNKNOWN


def test_confidence_falls_when_little_is_known():
    known = classify_vector(
        radius_earth=1.0, equilibrium_temperature=255.0, period_days=365.0
    )
    unknown_planet = classify_vector()
    assert known.confidence is Confidence.HIGH
    assert unknown_planet.confidence is Confidence.MINIMAL


def test_an_uncharacterised_planet_cannot_look_well_characterised():
    vector = classify_vector()
    assert vector.code == "?-?-?-?-?-0"


def test_eccentric_orbits_are_flagged_regardless_of_period():
    vector = classify_vector(period_days=2.0, eccentricity=0.6)
    assert vector.orbital is OrbitalRegime.ECCENTRIC


def test_ultra_short_period_regime():
    assert classify_vector(period_days=0.7, eccentricity=0.0).orbital is OrbitalRegime.ULTRA_SHORT


def test_the_scheme_is_versioned_and_labelled_provisional():
    vector = classify_vector(radius_earth=1.0)
    assert vector.scheme_version == SCHEME_VERSION
    assert "draft" in SCHEME_VERSION
    assert "not yet justified" in vector.caveat
    assert any("Provisional" in line for line in vector.describe())


# -- evidence integration ----------------------------------------------------


def test_evidence_table_distinguishes_detection_strength(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_text(
        "planet,molecule,detection_status,publication\n"
        "X b,CO2,DETECTED,Someone 2023\n"
        "X b,DMS,DISPUTED,Someone 2025\n",
        encoding="utf-8",
    )
    table = load_evidence(path)
    assert len(table.for_planet("X b")) == 2
    assert [e.molecule for e in table.detected_for_planet("X b")] == ["CO2"]


def test_a_disputed_molecule_is_never_asserted_as_present():
    assert not DetectionStatus.DISPUTED.asserts_presence
    assert not DetectionStatus.TENTATIVE.asserts_presence
    assert not DetectionStatus.UPPER_LIMIT.asserts_presence
    assert DetectionStatus.DETECTED.asserts_presence


def test_legacy_rows_import_as_unknown_not_as_detections(tmp_path):
    """The old flat CSV had no status column."""
    path = tmp_path / "planet_molecules.csv"
    path.write_text("pl_name,molecule,ref_url\nX b,H2O,http://example\n", encoding="utf-8")
    table = load_evidence(path)
    entry = table.for_planet("X b")[0]
    assert entry.detection_status is DetectionStatus.UNKNOWN
    assert not table.detected_for_planet("X b")


def test_the_shipped_evidence_table_parses():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "molecular_evidence.csv"
    if not path.exists():
        return
    table = load_evidence(path)
    assert len(table) > 10
    # K2-18 b's DMS claim must not read as an established detection.
    dms = [e for e in table.for_planet("K2-18 b") if "DMS" in e.molecule]
    assert dms and not dms[0].detection_status.asserts_presence


def test_description_groups_by_evidence_strength():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "molecular_evidence.csv"
    if not path.exists():
        return
    text = "\n".join(load_evidence(path).describe_planet("K2-18 b"))
    assert "DETECTED" in text
    assert "DISPUTED" in text
