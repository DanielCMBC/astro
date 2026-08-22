"""Data layer: catalogue access, validation, local store, provenance."""

from .nasa_archive import (
    CORE_COLUMNS,
    TAP_SYNC_URL,
    CatalogQuery,
    SolutionPolicy,
    build_query,
    fetch_catalog,
)
from .identity import (
    Catalog,
    EntityId,
    EntityKind,
    normalise_key,
    parse_entity_id,
    planet_id,
    star_id,
)
from .repository import CatalogRepository, SnapshotInfo
from .schema import PlanetRecord, StarRecord, build_planet_record, parse_float
from .synchronizer import (
    SyncResult,
    ValidationReport,
    frame_checksum,
    synchronize,
    validate_catalog,
)

__all__ = [
    "Catalog",
    "EntityId",
    "EntityKind",
    "normalise_key",
    "parse_entity_id",
    "planet_id",
    "star_id",
    "CORE_COLUMNS",
    "TAP_SYNC_URL",
    "CatalogQuery",
    "CatalogRepository",
    "PlanetRecord",
    "SnapshotInfo",
    "SolutionPolicy",
    "StarRecord",
    "SyncResult",
    "ValidationReport",
    "build_planet_record",
    "build_query",
    "fetch_catalog",
    "frame_checksum",
    "parse_float",
    "synchronize",
    "validate_catalog",
]
