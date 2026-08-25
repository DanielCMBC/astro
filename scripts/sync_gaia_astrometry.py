"""Refresh the validated Gaia DR3 astrometry cache. Explorer C3.6.

    python scripts/sync_gaia_astrometry.py [--catalog PATH] [--out PATH]

This is the **online** half of the C3.6 architecture and the only place the
project reaches the Gaia archive::

    NASA host identity -> gaia_dr3_id -> Gaia TAP -> validate -> atomic cache

Nothing in the application calls it. The explorer reads the committed cache
this writes, so startup and rendering perform no network access at all, and
a refresh that fails - no connectivity, a TAP outage, a malformed response,
a source that fails validation - leaves the previous cache in place and the
explorer fully usable.

Run it deliberately, check the diff, and commit the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astro_explorer.data.gaia import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    GaiaAstrometryCache,
    gaia_dr3_ids_from_catalog,
)


def _host_identities(catalog_path: Path) -> dict[str, str]:
    """``hostname -> Gaia DR3 source_id`` from a NASA ``ps`` snapshot.

    The snapshot committed for the vertical slice predates C3.6 and has no
    ``gaia_dr3_id`` column, so the identifiers are fetched for exactly the
    hosts it contains rather than for the whole archive. That keeps the
    cross-match the archive's, and the query small.
    """
    import pandas as pd
    import requests

    frame = pd.read_csv(catalog_path)
    hosts = sorted(str(h) for h in frame["hostname"].dropna().unique())

    if "gaia_dr3_id" in frame.columns:
        return gaia_dr3_ids_from_catalog(frame.to_dict("records"))

    quoted = ", ".join("'{0}'".format(h.replace("'", "''")) for h in hosts)
    query = (
        "select hostname, gaia_dr3_id from ps "
        "where default_flag = 1 and hostname in ({0})".format(quoted)
    )
    response = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        params={"query": query, "format": "csv"},
        timeout=90.0,
    )
    response.raise_for_status()

    import io

    rows = pd.read_csv(io.StringIO(response.text)).to_dict("records")
    return gaia_dr3_ids_from_catalog(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "reference_systems" / "vertical_slice_ps_default.csv",
        help="NASA ps snapshot whose hosts should be cached",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_CACHE_PATH,
        help="where the validated cache is written",
    )
    args = parser.parse_args(argv)

    identities = _host_identities(args.catalog)
    if not identities:
        print("no Gaia DR3 identifiers found; nothing to refresh")
        return 1

    for host, source_id in sorted(identities.items()):
        print("  {0:<14} -> Gaia DR3 {1}".format(host, source_id))

    cache = GaiaAstrometryCache(args.out)
    result = cache.refresh(sorted(set(identities.values())))
    print(result.describe())

    for source_id, errors in sorted(result.rejected.items()):
        for error in errors:
            print("  rejected {0}: {1}".format(source_id, error))

    if not result.succeeded:
        print("the previous cache is unchanged and still usable")
        return 1

    # The host -> source_id map is written beside the cache so the offline
    # runtime can find a star's astrometry from the name the catalogue uses,
    # without either re-querying NASA or cone-matching by position.
    import json

    map_path = args.out.with_name("gaia_dr3_hosts.json")
    map_path.write_text(
        json.dumps(dict(sorted(identities.items())), indent=2) + "\n", encoding="utf-8"
    )
    print("host map written to {0}".format(map_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - a command-line entry point
    raise SystemExit(main())
