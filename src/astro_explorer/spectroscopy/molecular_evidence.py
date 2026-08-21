"""Structured molecular detection evidence (roadmap section 3.12).

``planet_molecules.csv`` in the original program was a flat
``planet, molecule, url`` list, which the UI presented as "Atmospheric
detections" - stating as fact what the literature often reports as
tentative.  This module replaces that with an evidence table carrying a
detection status, instrument, facility, publication, retrieval method,
confidence and date.

A molecule is never described as present unless the recorded status says so.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "DetectionStatus",
    "MolecularEvidence",
    "EvidenceTable",
    "load_evidence",
    "EVIDENCE_COLUMNS",
]


class DetectionStatus(str, Enum):
    """How strongly the literature supports a molecule's presence."""

    DETECTED = "DETECTED"
    """Robust, generally reproduced detection."""

    TENTATIVE = "TENTATIVE"
    """Reported at low significance or contested."""

    UPPER_LIMIT = "UPPER_LIMIT"
    """Only a non-detection limit on the abundance is published."""

    NOT_DETECTED = "NOT_DETECTED"
    """Searched for and not found."""

    DISPUTED = "DISPUTED"
    """Claimed in one analysis and rejected in another."""

    UNKNOWN = "UNKNOWN"
    """No status recorded."""

    @property
    def label(self) -> str:
        return {
            DetectionStatus.DETECTED: "detected",
            DetectionStatus.TENTATIVE: "tentative",
            DetectionStatus.UPPER_LIMIT: "upper limit only",
            DetectionStatus.NOT_DETECTED: "searched, not detected",
            DetectionStatus.DISPUTED: "disputed",
            DetectionStatus.UNKNOWN: "status not recorded",
        }[self]

    @property
    def asserts_presence(self) -> bool:
        """Only DETECTED may be phrased as "this molecule is present"."""
        return self is DetectionStatus.DETECTED

    @classmethod
    def parse(cls, text: str | None) -> "DetectionStatus":
        if not text:
            return cls.UNKNOWN
        key = str(text).strip().upper().replace(" ", "_").replace("-", "_")
        try:
            return cls(key)
        except ValueError:
            return cls.UNKNOWN


#: Columns of the evidence CSV (roadmap section 3.12).
EVIDENCE_COLUMNS = (
    "planet",
    "molecule",
    "detection_status",
    "instrument",
    "facility",
    "publication",
    "reference_url",
    "retrieval_method",
    "confidence",
    "date",
    "notes",
)


@dataclass(frozen=True)
class MolecularEvidence:
    """One published statement about one molecule in one atmosphere."""

    planet: str
    molecule: str
    detection_status: DetectionStatus = DetectionStatus.UNKNOWN
    instrument: str = ""
    facility: str = ""
    publication: str = ""
    reference_url: str = ""
    retrieval_method: str = ""
    confidence: str = ""
    date: str = ""
    notes: str = ""

    def summary_line(self) -> str:
        """A line that never overstates the evidence."""
        parts = ["{0}: {1}".format(self.molecule, self.detection_status.label)]
        origin = " / ".join(p for p in (self.instrument, self.facility) if p)
        if origin:
            parts.append("({0})".format(origin))
        if self.confidence:
            parts.append("[{0}]".format(self.confidence))
        if self.publication:
            parts.append("- {0}".format(self.publication))
        return " ".join(parts)


@dataclass
class EvidenceTable:
    """All recorded evidence, queryable by planet."""

    entries: list[MolecularEvidence] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def for_planet(self, planet: str) -> list[MolecularEvidence]:
        target = planet.strip().casefold()
        return [entry for entry in self.entries if entry.planet.strip().casefold() == target]

    def detected_for_planet(self, planet: str) -> list[MolecularEvidence]:
        """Only entries whose status actually asserts presence."""
        return [e for e in self.for_planet(planet) if e.detection_status.asserts_presence]

    def describe_planet(self, planet: str) -> list[str]:
        """UI lines grouped by how strong the evidence is."""
        entries = self.for_planet(planet)
        if not entries:
            return ["No molecular evidence recorded for {0}.".format(planet)]

        lines: list[str] = []
        for status in (
            DetectionStatus.DETECTED,
            DetectionStatus.TENTATIVE,
            DetectionStatus.DISPUTED,
            DetectionStatus.UPPER_LIMIT,
            DetectionStatus.NOT_DETECTED,
            DetectionStatus.UNKNOWN,
        ):
            group = [e for e in entries if e.detection_status is status]
            if not group:
                continue
            lines.append("{0}:".format(status.label.upper()))
            lines.extend("  " + entry.summary_line() for entry in group)
        return lines


def _normalise_legacy_row(row: dict) -> dict:
    """Accept the original ``pl_name, molecule, ref_url`` layout.

    A legacy row carries no detection status, so it becomes UNKNOWN rather
    than being promoted to a detection.
    """
    if "planet" in row:
        return row
    return {
        "planet": row.get("pl_name", ""),
        "molecule": row.get("molecule", ""),
        "detection_status": row.get("detection_status", ""),
        "reference_url": row.get("ref_url", ""),
        "notes": "imported from legacy planet_molecules.csv; status not recorded",
    }


def load_evidence(path) -> EvidenceTable:
    """Read an evidence CSV, tolerating the legacy three-column layout."""
    path = Path(path)
    table = EvidenceTable()
    if not path.exists():
        return table

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = [line for line in handle if not line.lstrip().startswith("#")]

    for raw in csv.DictReader(rows):
        row = _normalise_legacy_row({k: (v or "").strip() for k, v in raw.items() if k})
        planet = row.get("planet", "")
        molecule = row.get("molecule", "")
        if not planet or not molecule:
            continue
        table.entries.append(
            MolecularEvidence(
                planet=planet,
                molecule=molecule,
                detection_status=DetectionStatus.parse(row.get("detection_status")),
                instrument=row.get("instrument", ""),
                facility=row.get("facility", ""),
                publication=row.get("publication", ""),
                reference_url=row.get("reference_url", ""),
                retrieval_method=row.get("retrieval_method", ""),
                confidence=row.get("confidence", ""),
                date=row.get("date", ""),
                notes=row.get("notes", ""),
            )
        )
    return table
