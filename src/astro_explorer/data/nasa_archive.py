"""NASA Exoplanet Archive access with an explicit record-selection policy.

Roadmap section 3.2 (P0).  The original program ran ``select ... from ps``
and then ``drop_duplicates(subset=["pl_name"])``, which keeps whichever
published parameter set happened to sort first.  The ``ps`` table holds one
row per *published solution*, so that silently mixes references: the mass
from one paper, the radius from another, the period from a third.

Two concepts are now distinct and never interchangeable:

``DEFAULT_SOLUTION``
    ``ps`` filtered to ``default_flag = 1``.  One internally consistent
    published solution per planet, chosen by the archive.

``COMPOSITE``
    ``pscomppars``.  Maximally populated, but individual columns come from
    different references by construction, so the row as a whole is not a
    single self-consistent solution.

Nothing here treats them as equivalent, and the selected policy travels with
the data so the UI can state which one is on screen.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import pandas as pd
import requests

__all__ = [
    "SolutionPolicy",
    "CatalogQuery",
    "TAP_SYNC_URL",
    "fetch_catalog",
    "build_query",
    "CORE_COLUMNS",
    "COMPOSITE_ONLY_COLUMNS",
    "DEFAULT_ONLY_COLUMNS",
]

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


class SolutionPolicy(str, Enum):
    """Which NASA record concept to use (roadmap section 3.2)."""

    DEFAULT_SOLUTION = "DEFAULT_SOLUTION"
    """``ps`` where ``default_flag = 1``: one self-consistent reference."""

    COMPOSITE = "COMPOSITE"
    """``pscomppars``: most complete, but columns mix references."""

    @property
    def table(self) -> str:
        return "ps" if self is SolutionPolicy.DEFAULT_SOLUTION else "pscomppars"

    @property
    def label(self) -> str:
        return {
            SolutionPolicy.DEFAULT_SOLUTION: "NASA default solution (ps, default_flag=1)",
            SolutionPolicy.COMPOSITE: "NASA composite parameters (pscomppars)",
        }[self]

    @property
    def caveat(self) -> str:
        if self is SolutionPolicy.COMPOSITE:
            return (
                "Composite table: individual columns may come from different "
                "publications, so the row is not one self-consistent solution."
            )
        return (
            "Default solution: all columns come from the reference the archive "
            "marks as default for this planet."
        )

    @property
    def is_self_consistent(self) -> bool:
        return self is SolutionPolicy.DEFAULT_SOLUTION


#: Columns present in both tables.
CORE_COLUMNS = (
    "pl_name",
    "hostname",
    "pl_letter",
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbsmax",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",
    "pl_orbeccen",
    "pl_orbeccenerr1",
    "pl_orbeccenerr2",
    "pl_orbincl",
    "pl_orbinclerr1",
    "pl_orbinclerr2",
    "pl_orblper",
    "pl_orbtper",
    "pl_tranmid",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_radj",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_bmassj",
    "pl_bmassprov",
    "pl_dens",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "st_mass",
    "st_masserr1",
    "st_masserr2",
    "st_lum",
    "st_spectype",
    "st_met",
    "st_age",
    "ra",
    "dec",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_plx",
    "sy_plxerr1",
    "sy_plxerr2",
    "sy_vmag",
    "sy_gaiamag",
    "discoverymethod",
    "disc_year",
    "disc_facility",
)

#: Present only in ``ps``; they are what makes the default solution traceable.
DEFAULT_ONLY_COLUMNS = ("default_flag", "pl_refname", "st_refname", "sy_refname", "rowupdate")

#: Present only in ``pscomppars``.
COMPOSITE_ONLY_COLUMNS = ()


@dataclass(frozen=True)
class CatalogQuery:
    """A TAP query plus the policy that produced it."""

    policy: SolutionPolicy
    adql: str
    columns: tuple[str, ...]

    @property
    def table(self) -> str:
        return self.policy.table


def build_query(policy: SolutionPolicy = SolutionPolicy.COMPOSITE) -> CatalogQuery:
    """Build the ADQL for a policy.

    The default-solution query filters on ``default_flag = 1`` in SQL rather
    than de-duplicating in pandas afterwards, so the choice of row is the
    archive's documented one and not an artefact of sort order.
    """
    columns = list(CORE_COLUMNS)
    if policy is SolutionPolicy.DEFAULT_SOLUTION:
        columns += list(DEFAULT_ONLY_COLUMNS)
    else:
        columns += list(COMPOSITE_ONLY_COLUMNS)

    where = " where default_flag = 1" if policy is SolutionPolicy.DEFAULT_SOLUTION else ""
    adql = "select {0} from {1}{2} order by hostname, pl_name".format(
        ", ".join(columns), policy.table, where
    )
    return CatalogQuery(policy=policy, adql=adql, columns=tuple(columns))


def fetch_catalog(
    policy: SolutionPolicy = SolutionPolicy.COMPOSITE,
    *,
    timeout: float = 90.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download a catalogue table.

    The returned frame carries ``attrs`` describing where it came from, so
    provenance survives caching and the UI can name the policy on screen.
    """
    query = build_query(policy)
    getter = session.get if session is not None else requests.get
    response = getter(
        TAP_SYNC_URL,
        params={"query": " ".join(query.adql.split()), "format": "csv"},
        timeout=timeout,
    )
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text))
    frame.attrs["solution_policy"] = policy.value
    frame.attrs["source_table"] = policy.table
    frame.attrs["retrieved"] = datetime.now(timezone.utc).isoformat()
    frame.attrs["query"] = query.adql
    return frame
