"""Data-layer tests: record policy, validation and offline behaviour.

Covers roadmap section 3.2 (the ``ps`` duplicate problem) and section 11
(offline-first synchronisation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_explorer.data.nasa_archive import SolutionPolicy, build_query
from astro_explorer.data.repository import CatalogRepository
from astro_explorer.data.synchronizer import (
    frame_checksum,
    synchronize,
    validate_catalog,
)


def make_frame(rows: int = 5000, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pl_name": ["P{0} b".format(i) for i in range(rows)],
            "hostname": ["H{0}".format(i) for i in range(rows)],
            "pl_orbper": np.linspace(1.0, 500.0, rows),
            "pl_orbsmax": np.linspace(0.01, 5.0, rows),
            "st_teff": np.linspace(3000.0, 7000.0, rows),
            "st_rad": np.linspace(0.2, 2.0, rows),
            "st_mass": np.linspace(0.2, 2.0, rows),
            "sy_dist": np.linspace(5.0, 500.0, rows),
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


# -- record selection policy (roadmap 3.2) ----------------------------------


def test_default_solution_filters_in_sql_not_in_pandas():
    """The original code did drop_duplicates after sorting by hostname."""
    query = build_query(SolutionPolicy.DEFAULT_SOLUTION)
    assert query.table == "ps"
    assert "default_flag = 1" in query.adql


def test_composite_query_uses_pscomppars():
    query = build_query(SolutionPolicy.COMPOSITE)
    assert query.table == "pscomppars"
    assert "default_flag" not in query.adql


def test_the_two_policies_are_never_described_as_equivalent():
    assert SolutionPolicy.DEFAULT_SOLUTION.label != SolutionPolicy.COMPOSITE.label
    assert SolutionPolicy.DEFAULT_SOLUTION.is_self_consistent
    assert not SolutionPolicy.COMPOSITE.is_self_consistent
    assert "different publications" in SolutionPolicy.COMPOSITE.caveat


def test_default_solution_requests_the_reference_columns():
    query = build_query(SolutionPolicy.DEFAULT_SOLUTION)
    assert "pl_refname" in query.adql
    assert "default_flag" in query.columns


# -- validation --------------------------------------------------------------


def test_a_good_frame_validates():
    report = validate_catalog(make_frame(), SolutionPolicy.COMPOSITE)
    assert report.ok, report.errors


def test_missing_mandatory_columns_are_fatal():
    frame = make_frame().drop(columns=["pl_name"])
    assert not validate_catalog(frame, SolutionPolicy.COMPOSITE).ok


def test_a_truncated_download_is_rejected():
    report = validate_catalog(make_frame(10), SolutionPolicy.COMPOSITE)
    assert not report.ok
    assert "truncated" in report.summary()


def test_duplicate_planet_names_are_rejected():
    frame = make_frame()
    frame.loc[1, "pl_name"] = frame.loc[0, "pl_name"]
    assert not validate_catalog(frame, SolutionPolicy.COMPOSITE).ok


def test_non_default_rows_are_rejected_under_the_default_policy():
    frame = make_frame()
    frame["default_flag"] = 1
    frame.loc[0, "default_flag"] = 0
    report = validate_catalog(frame, SolutionPolicy.DEFAULT_SOLUTION)
    assert not report.ok
    assert "default solution" in report.summary()


def test_a_unit_change_is_caught():
    """Period suddenly in seconds rather than days."""
    frame = make_frame()
    frame["pl_orbper"] = frame["pl_orbper"] * 86400.0 * 1000.0
    report = validate_catalog(frame, SolutionPolicy.COMPOSITE)
    assert not report.ok
    assert "unit may have changed" in report.summary()


def test_eccentricity_above_one_is_caught():
    frame = make_frame()
    frame["pl_orbeccen"] = 1.5
    assert not validate_catalog(frame, SolutionPolicy.COMPOSITE).ok


def test_checksum_is_stable_and_order_independent_across_columns():
    frame = make_frame(100)
    assert frame_checksum(frame) == frame_checksum(frame.copy())
    reordered = frame[list(reversed(frame.columns))]
    assert frame_checksum(frame) == frame_checksum(reordered)


def test_checksum_changes_when_data_changes():
    frame = make_frame(100)
    changed = frame.copy()
    changed.loc[0, "pl_orbper"] = 999.0
    assert frame_checksum(frame) != frame_checksum(changed)


# -- offline-first synchronisation (roadmap 11) -----------------------------


@pytest.fixture()
def repository(tmp_path) -> CatalogRepository:
    return CatalogRepository(tmp_path / "store")


def test_first_sync_commits(repository):
    frame = make_frame()
    result = synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: frame)
    assert result.succeeded
    assert repository.has_snapshot
    assert len(repository.load()) == len(frame)


def test_a_network_failure_leaves_the_snapshot_intact(repository):
    good = make_frame()
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: good)

    def boom(_policy):
        raise ConnectionError("no network")

    result = synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=boom)
    assert not result.succeeded
    assert result.outcome == "download_failed"
    assert len(repository.load()) == len(good)


def test_a_rejected_download_leaves_the_snapshot_intact(repository):
    good = make_frame()
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: good)

    result = synchronize(
        repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame(10)
    )
    assert result.outcome == "rejected"
    assert repository.snapshot_info().planet_count == len(good)


def test_a_large_drop_in_planet_count_is_refused(repository):
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame(5000))
    result = synchronize(
        repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame(4200)
    )
    assert result.outcome == "rejected"
    assert "refusing to replace" in result.detail


def test_identical_data_is_recognised_as_unchanged(repository):
    frame = make_frame()
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: frame)
    result = synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: frame.copy())
    assert result.outcome == "unchanged"


def test_the_application_works_with_no_network_at_all(repository):
    """Startup must succeed offline once a snapshot exists."""
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame())
    reopened = CatalogRepository(repository.root)
    frame = reopened.load()
    assert frame is not None
    assert frame.attrs["offline_snapshot"] is True


def test_snapshot_records_its_provenance(repository):
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame())
    info = repository.snapshot_info()
    assert info.source_table == "pscomppars"
    assert info.sha256
    assert info.retrieved
    assert "pscomppars" in "\n".join(info.describe())


def test_sync_history_is_logged(repository):
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame())
    synchronize(repository, SolutionPolicy.COMPOSITE, fetcher=lambda _p: make_frame(10))
    outcomes = [entry["outcome"] for entry in repository.sync_history()]
    assert "committed" in outcomes
    assert "rejected" in outcomes


def test_provenance_can_be_persisted_and_read_back(repository):
    import astropy.units as u

    from astro_explorer.provenance import derived, measured

    repository.record_provenance(
        "Test b",
        {
            "semimajor_axis": derived(0.05, u.au, provenance="kepler3"),
            "period": measured(3.2, u.day, provenance="nasa"),
        },
    )
    rows = repository.provenance_for("Test b")
    statuses = {row["field"]: row["status"] for row in rows}
    assert statuses["semimajor_axis"] == "DERIVED"
    assert statuses["period"] == "MEASURED"
