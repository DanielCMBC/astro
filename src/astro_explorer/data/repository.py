"""The local scientific store (roadmap section 11).

The application always reads from a validated local snapshot.  The GUI never
depends on a live NASA request: a failed or slow network leaves the last good
snapshot in place, and the program stays fully usable offline.

Storage follows roadmap section 11.1 - SQLite for records, provenance and
synchronisation state; Parquet/Feather for the dense catalogue table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .nasa_archive import SolutionPolicy

__all__ = ["SnapshotInfo", "CatalogRepository"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    solution_policy TEXT    NOT NULL,
    source_table    TEXT    NOT NULL,
    retrieved       TEXT    NOT NULL,
    row_count       INTEGER NOT NULL,
    planet_count    INTEGER NOT NULL,
    sha256          TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    data_file       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started    TEXT NOT NULL,
    finished   TEXT,
    policy     TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    detail     TEXT
);

CREATE TABLE IF NOT EXISTS provenance (
    planet     TEXT NOT NULL,
    field      TEXT NOT NULL,
    value      REAL,
    unit       TEXT,
    status     TEXT NOT NULL,
    source     TEXT,
    reference  TEXT,
    retrieved  TEXT,
    PRIMARY KEY (planet, field)
);
"""


@dataclass(frozen=True)
class SnapshotInfo:
    """What the local store currently holds."""

    solution_policy: SolutionPolicy
    source_table: str
    retrieved: str
    row_count: int
    planet_count: int
    sha256: str
    query: str
    data_file: str

    def describe(self) -> list[str]:
        return [
            "Snapshot policy:   {0}".format(self.solution_policy.label),
            "Source table:      {0}".format(self.source_table),
            "Rows:              {0} ({1} planets)".format(self.row_count, self.planet_count),
            "Last synchronised: {0}".format(self.retrieved),
            "Checksum:          {0}".format(self.sha256[:16]),
        ]


class CatalogRepository:
    """Reads and writes the local snapshot, atomically.

    The active dataset is only replaced once a staged download has passed
    validation (roadmap section 11.2).  A failed synchronisation leaves the
    previous snapshot untouched.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "staging"
        self.staging.mkdir(exist_ok=True)
        self.db_path = self.root / "catalog.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.executescript(SCHEMA)

    # -- reading ---------------------------------------------------------
    def snapshot_info(self) -> SnapshotInfo | None:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM snapshot WHERE id = 1").fetchone()
        if row is None:
            return None
        return SnapshotInfo(
            solution_policy=SolutionPolicy(row["solution_policy"]),
            source_table=row["source_table"],
            retrieved=row["retrieved"],
            row_count=row["row_count"],
            planet_count=row["planet_count"],
            sha256=row["sha256"],
            query=row["query"],
            data_file=row["data_file"],
        )

    def load(self) -> pd.DataFrame | None:
        """Load the active snapshot, or None when the store is empty."""
        info = self.snapshot_info()
        if info is None:
            return None
        path = self.root / info.data_file
        if not path.exists():
            return None
        frame = pd.read_feather(path)
        frame.attrs["solution_policy"] = info.solution_policy.value
        frame.attrs["source_table"] = info.source_table
        frame.attrs["retrieved"] = info.retrieved
        frame.attrs["offline_snapshot"] = True
        return frame

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot_info() is not None

    # -- writing ---------------------------------------------------------
    def stage(self, frame: pd.DataFrame, policy: SolutionPolicy) -> Path:
        """Write a downloaded frame into staging without activating it."""
        path = self.staging / "catalog-{0}.feather".format(policy.table)
        frame.reset_index(drop=True).to_feather(path)
        return path

    def commit(
        self,
        frame: pd.DataFrame,
        policy: SolutionPolicy,
        *,
        query: str = "",
        checksum: str = "",
    ) -> SnapshotInfo:
        """Atomically replace the active dataset (roadmap section 11.2).

        The new file is written under a temporary name and then renamed, so
        a crash mid-write cannot leave a truncated snapshot in place.
        """
        frame = frame.reset_index(drop=True)
        data_file = "catalog-{0}.feather".format(policy.table)
        final_path = self.root / data_file
        temp_path = final_path.with_suffix(".feather.tmp")

        frame.to_feather(temp_path)
        temp_path.replace(final_path)

        info = SnapshotInfo(
            solution_policy=policy,
            source_table=policy.table,
            retrieved=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            row_count=int(len(frame)),
            planet_count=int(frame["pl_name"].nunique()) if "pl_name" in frame else int(len(frame)),
            sha256=checksum,
            query=query,
            data_file=data_file,
        )

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM snapshot WHERE id = 1")
            connection.execute(
                "INSERT INTO snapshot (id, solution_policy, source_table, retrieved,"
                " row_count, planet_count, sha256, query, data_file)"
                " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.solution_policy.value,
                    info.source_table,
                    info.retrieved,
                    info.row_count,
                    info.planet_count,
                    info.sha256,
                    info.query,
                    info.data_file,
                ),
            )
        return info

    def log_sync(self, policy: SolutionPolicy, outcome: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO sync_log (started, finished, policy, outcome, detail)"
                " VALUES (?, ?, ?, ?, ?)",
                (now, now, policy.value, outcome, detail),
            )

    def sync_history(self, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM sync_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- provenance ------------------------------------------------------
    def record_provenance(self, planet: str, fields: dict) -> None:
        """Persist per-field provenance (roadmap section 12)."""
        with sqlite3.connect(self.db_path) as connection:
            for field_name, parameter in fields.items():
                record = parameter.as_dict()
                connection.execute(
                    "INSERT OR REPLACE INTO provenance"
                    " (planet, field, value, unit, status, source, reference, retrieved)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        planet,
                        field_name,
                        record["value"],
                        record["unit"],
                        record["status"],
                        record["provenance"],
                        record["reference"],
                        record["retrieved"],
                    ),
                )

    def provenance_for(self, planet: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM provenance WHERE planet = ? ORDER BY field", (planet,)
            ).fetchall()
        return [dict(row) for row in rows]

    def export_manifest(self) -> str:
        """JSON description of the store, for the DATA view."""
        info = self.snapshot_info()
        return json.dumps(
            {
                "root": str(self.root),
                "snapshot": None if info is None else info.__dict__ | {
                    "solution_policy": info.solution_policy.value
                },
                "recent_syncs": self.sync_history(5),
            },
            indent=2,
            default=str,
        )
