"""Atmospheric spectroscopy: IPAC parsing, spectrum objects, evidence."""

from .ipac import (
    REQUIRED_COLUMNS,
    SpectrumParseError,
    read_ipac_spectrum,
    read_planet_spectra,
    spectrum_files_for_planet,
)
from .models import Spectrum, SpectrumCollection
from .molecular_evidence import (
    EVIDENCE_COLUMNS,
    DetectionStatus,
    EvidenceTable,
    MolecularEvidence,
    load_evidence,
)
from .normalization import (
    MolecularBand,
    Normalization,
    band_overlays,
    load_signatures,
    normalize,
)

__all__ = [
    "EVIDENCE_COLUMNS",
    "REQUIRED_COLUMNS",
    "DetectionStatus",
    "EvidenceTable",
    "MolecularBand",
    "MolecularEvidence",
    "Normalization",
    "Spectrum",
    "SpectrumCollection",
    "SpectrumParseError",
    "band_overlays",
    "load_evidence",
    "load_signatures",
    "normalize",
    "read_ipac_spectrum",
    "read_planet_spectra",
    "spectrum_files_for_planet",
]
