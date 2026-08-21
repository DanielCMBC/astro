"""Validated, atomic catalogue synchronisation (roadmap section 11.2).

The ten steps the roadmap requires are implemented as an explicit pipeline::

    connectivity -> download -> stage -> validate columns -> validate units
    -> validate row counts and identifiers -> reject malformed -> record
    timestamps -> hash -> atomic replace

A synchronisation that fails at any step leaves the previous snapshot
active.  The application never ends up with a half-written catalogue, and
never ends up with no catalogue at all because a download went wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .nasa_archive import CORE_COLUMNS, SolutionPolicy, build_query, fetch_catalog
from .repository import CatalogRepository, SnapshotInfo

__all__ = ["ValidationReport", "SyncResult", "validate_catalog", "synchronize", "frame_checksum"]

#: Columns without which the application cannot function.
MANDATORY_COLUMNS = ("pl_name", "hostname")

#: Plausible ranges used to reject a malformed download.  These are sanity
#: bounds, not science: a value outside them means the columns are wrong or
#: the units changed, not that the planet is unusual.
SANITY_RANGES = {
    "pl_orbper": (1e-4, 1e7),  # days
    "pl_orbsmax": (1e-5, 1e4),  # AU
    "pl_orbeccen": (0.0, 1.0),
    "pl_rade": (0.01, 1e3),  # Earth radii
    "st_teff": (500.0, 1e5),  # K
    "st_rad": (1e-3, 1e4),  # solar radii
    "st_mass": (1e-3, 1e3),  # solar masses
    "sy_dist": (0.1, 1e6),  # pc
}

#: A download with fewer planets than this is assumed to be truncated.
MIN_EXPECTED_PLANETS = 4000


@dataclass
class ValidationReport:
    """Outcome of validating a staged download."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    planet_count: int = 0

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        if self.ok:
            return "valid: {0} rows, {1} planets, {2} warning(s)".format(
                self.row_count, self.planet_count, len(self.warnings)
            )
        return "rejected: " + "; ".join(self.errors)


def frame_checksum(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over the frame's contents (roadmap section 11.2.9)."""
    digest = hashlib.sha256()
    for column in sorted(frame.columns):
        digest.update(column.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame[column], index=False).values.tobytes())
    return digest.hexdigest()


def validate_catalog(
    frame: pd.DataFrame,
    policy: SolutionPolicy,
    *,
    previous: SnapshotInfo | None = None,
    min_planets: int = MIN_EXPECTED_PLANETS,
) -> ValidationReport:
    """Check a staged frame before it is allowed to replace the active one."""
    report = ValidationReport(row_count=int(len(frame)))

    # 4. expected columns
    for column in MANDATORY_COLUMNS:
        if column not in frame.columns:
            report.fail("missing mandatory column {0!r}".format(column))
    if not report.ok:
        return report

    missing_core = [c for c in CORE_COLUMNS if c not in frame.columns]
    if missing_core:
        report.warn("{0} expected column(s) absent, e.g. {1}".format(
            len(missing_core), ", ".join(missing_core[:5])
        ))

    # 6. row counts and unique identifiers
    report.planet_count = int(frame["pl_name"].nunique())
    if report.row_count == 0:
        report.fail("download contains no rows")
        return report
    if report.planet_count < min_planets:
        report.fail(
            "only {0} distinct planets; expected at least {1}. Treating this as "
            "a truncated download rather than a shrinking catalogue".format(
                report.planet_count, min_planets
            )
        )

    if policy is SolutionPolicy.DEFAULT_SOLUTION:
        if "default_flag" in frame.columns:
            non_default = int((frame["default_flag"] != 1).sum())
            if non_default:
                report.fail(
                    "{0} row(s) are not the default solution; the default-solution "
                    "policy must not mix published parameter sets".format(non_default)
                )
        duplicates = int(frame["pl_name"].duplicated().sum())
        if duplicates:
            report.fail(
                "{0} duplicate planet name(s) in a default-solution download".format(duplicates)
            )
    elif policy is SolutionPolicy.COMPOSITE:
        duplicates = int(frame["pl_name"].duplicated().sum())
        if duplicates:
            report.fail("{0} duplicate planet name(s) in pscomppars".format(duplicates))

    # 5. unit sanity: a column that silently changed units shows up here.
    for column, (low, high) in SANITY_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            report.warn("column {0!r} is entirely empty".format(column))
            continue
        outside = int(np.sum((finite < low) | (finite > high)))
        fraction = outside / finite.size
        if fraction > 0.01:
            report.fail(
                "column {0!r}: {1:.1%} of values fall outside the plausible range "
                "[{2:g}, {3:g}]; the column or its unit may have changed".format(
                    column, fraction, low, high
                )
            )
        elif outside:
            report.warn("column {0!r}: {1} value(s) outside [{2:g}, {3:g}]".format(
                column, outside, low, high
            ))

    # A catalogue that suddenly loses a large fraction of its planets is
    # more likely a broken query than a real retraction.
    if previous is not None and previous.planet_count:
        ratio = report.planet_count / previous.planet_count
        if ratio < 0.9:
            report.fail(
                "planet count fell from {0} to {1} ({2:.0%}); refusing to replace "
                "the working snapshot".format(previous.planet_count, report.planet_count, ratio)
            )

    return report


@dataclass
class SyncResult:
    """What a synchronisation attempt did."""

    outcome: str
    report: ValidationReport | None = None
    snapshot: SnapshotInfo | None = None
    detail: str = ""
    started: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome == "committed"

    def describe(self) -> str:
        if self.succeeded:
            return "Synchronisation committed: {0}".format(
                self.report.summary() if self.report else ""
            )
        return "Synchronisation {0}: {1}".format(self.outcome, self.detail)


def synchronize(
    repository: CatalogRepository,
    policy: SolutionPolicy = SolutionPolicy.COMPOSITE,
    *,
    fetcher=fetch_catalog,
    min_planets: int = MIN_EXPECTED_PLANETS,
) -> SyncResult:
    """Run one full synchronisation cycle.

    On any failure the previous snapshot stays active and the application
    remains usable offline.  ``fetcher`` is injectable so the pipeline can be
    tested without network access.
    """
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous = repository.snapshot_info()

    # 1-3. connectivity and download into staging.
    try:
        frame = fetcher(policy)
    except Exception as exc:
        detail = "download failed: {0}".format(exc)
        repository.log_sync(policy, "download_failed", detail)
        return SyncResult("download_failed", detail=detail, started=started, snapshot=previous)

    repository.stage(frame, policy)

    # 4-7. validation; reject malformed updates.
    report = validate_catalog(frame, policy, previous=previous, min_planets=min_planets)
    if not report.ok:
        detail = report.summary()
        repository.log_sync(policy, "rejected", detail)
        return SyncResult("rejected", report=report, detail=detail, started=started, snapshot=previous)

    # 8-10. timestamps, hash, atomic replacement.
    checksum = frame_checksum(frame)
    if previous is not None and previous.sha256 == checksum:
        repository.log_sync(policy, "unchanged", "checksum matches the active snapshot")
        return SyncResult(
            "unchanged", report=report, snapshot=previous, started=started,
            detail="remote data is identical to the local snapshot",
        )

    snapshot = repository.commit(
        frame, policy, query=build_query(policy).adql, checksum=checksum
    )
    repository.log_sync(policy, "committed", report.summary())
    return SyncResult("committed", report=report, snapshot=snapshot, started=started)
