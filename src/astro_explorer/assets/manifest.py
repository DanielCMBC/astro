"""Asset manifest with visual provenance (roadmap sections 4.9 and 16).

An exoplanet texture is never a photograph.  Every asset therefore declares
what it actually is, and the UI is required to show that badge, so an artist
concept can never be mistaken for an observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ["AssetType", "AssetRecord", "AssetManifest", "sha256_file"]


class AssetType(str, Enum):
    """Visual provenance category (roadmap section 16)."""

    OBSERVED = "OBSERVED"
    """Derived from real imaging or mapping of this body."""

    NASA_CONCEPT = "NASA_CONCEPT"
    """An artist's impression published by NASA or a mission team."""

    SCIENTIFIC_PROCEDURAL = "SCIENTIFIC_PROCEDURAL"
    """Generated from measured physical parameters."""

    GENERIC_CLASS = "GENERIC_CLASS"
    """A placeholder representing a class of planet, not this planet."""

    @property
    def badge(self) -> str:
        """The text the UI must display over or beside the image."""
        return {
            AssetType.OBSERVED: "Observed imagery",
            AssetType.NASA_CONCEPT: "NASA artist concept - not a photograph",
            AssetType.SCIENTIFIC_PROCEDURAL: (
                "Procedural visualisation - actual appearance unknown"
            ),
            AssetType.GENERIC_CLASS: (
                "Generic class placeholder - not a depiction of this planet"
            ),
        }[self]

    @property
    def implies_observation(self) -> bool:
        return self is AssetType.OBSERVED


def sha256_file(path) -> str:
    """SHA-256 of a file, for manifest integrity checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetRecord:
    """One texture, concept image or shader asset."""

    asset_id: str
    relative_path: str
    asset_type: AssetType
    planet_name: str = ""
    source_url: str = ""
    creator: str = ""
    credit: str = ""
    license: str = ""
    retrieved: str = ""
    sha256: str = ""

    @property
    def badge(self) -> str:
        return self.asset_type.badge

    def describe(self) -> list[str]:
        lines = ["Asset:  {0}".format(self.asset_id), "Status: {0}".format(self.badge)]
        if self.creator:
            lines.append("Credit: {0}".format(self.credit or self.creator))
        if self.source_url:
            lines.append("Source: {0}".format(self.source_url))
        if self.license:
            lines.append("Licence: {0}".format(self.license))
        return lines

    def verify(self, root: Path) -> bool:
        """True when the file exists and matches its recorded hash."""
        path = Path(root) / self.relative_path
        if not path.exists():
            return False
        if not self.sha256:
            return True
        return sha256_file(path) == self.sha256

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "asset_type": self.asset_type.value,
            "planet_name": self.planet_name,
            "source_url": self.source_url,
            "creator": self.creator,
            "credit": self.credit,
            "license": self.license,
            "retrieved": self.retrieved,
            "sha256": self.sha256,
        }


@dataclass
class AssetManifest:
    """The declared set of assets, loadable from and savable to JSON."""

    records: dict[str, AssetRecord] = field(default_factory=dict)

    def add(self, record: AssetRecord) -> None:
        self.records[record.asset_id] = record

    def get(self, asset_id: str) -> AssetRecord | None:
        return self.records.get(asset_id)

    def for_planet(self, planet_name: str) -> list[AssetRecord]:
        target = planet_name.strip().casefold()
        return [r for r in self.records.values() if r.planet_name.strip().casefold() == target]

    def verify_all(self, root: Path) -> dict[str, bool]:
        return {asset_id: record.verify(root) for asset_id, record in self.records.items()}

    @classmethod
    def load(cls, path) -> "AssetManifest":
        path = Path(path)
        manifest = cls()
        if not path.exists():
            return manifest
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for entry in raw.get("assets", []):
            manifest.add(
                AssetRecord(
                    asset_id=entry["asset_id"],
                    relative_path=entry["relative_path"],
                    asset_type=AssetType(entry.get("asset_type", "GENERIC_CLASS")),
                    planet_name=entry.get("planet_name", ""),
                    source_url=entry.get("source_url", ""),
                    creator=entry.get("creator", ""),
                    credit=entry.get("credit", ""),
                    license=entry.get("license", ""),
                    retrieved=entry.get("retrieved", ""),
                    sha256=entry.get("sha256", ""),
                )
            )
        return manifest

    def save(self, path) -> None:
        with open(Path(path), "w", encoding="utf-8") as handle:
            json.dump(
                {"assets": [r.as_dict() for r in self.records.values()]},
                handle,
                indent=2,
            )
