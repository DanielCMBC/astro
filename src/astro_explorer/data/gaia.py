"""Gaia DR3 as the authoritative astrometric source, cached for offline use.

Explorer C3.6. The NASA Exoplanet Archive gives every planet a host RA and
Dec and, in the ``ps`` table, a ``gaia_dr3_id``. What it does not give is a
coordinate epoch: ``ra`` and ``dec`` arrive with no ``ref_epoch`` column, and
``sy_pmra``/``sy_pmdec``/``st_radv`` are compiled values whose epoch is not
stated either. An unlabelled position cannot open the epoch gate, however
many motion columns sit beside it - so this module goes to the catalogue
that does state one.

Why the identifier and not a cone search
----------------------------------------

``gaia_dr3_id`` is a *deterministic cross-match the archive already made*.
Re-deriving it by cone-matching on name or position would substitute our
guess for their curated answer, and would do so most confidently in exactly
the crowded fields where it is most likely to be wrong: a 2-arcsecond cone
around a high-proper-motion star at an epoch we cannot state is not a
reliable way to pick one source out of several. So the identifier is
parsed, not searched for, and a row whose identifier is absent gets no
astrometry rather than a nearby star's.

What Gaia DR3 actually says
---------------------------

* the reference epoch is **J2016.0**, and it is stored from ``ref_epoch``
  rather than assumed. Sixteen years separate it from J2000, which is 33
  arcsec of motion for HD 219134;
* positions and proper motions are **ICRS**;
* times are **TCB**;
* ``pmra`` is already :math:`\\mu_\\alpha^* = \\dot\\alpha\\cos\\delta`, the
  great-circle rate. It maps *directly* to Astropy's ``pm_ra_cosdec``.
  Multiplying by ``cos(dec)`` again is the classic error here, and it is
  silent: it shrinks a star's motion by a factor that is 1 at the equator
  and only becomes visible at high declination;
* ``astrometric_params_solved`` says how many parameters the solution
  actually fitted - 3 for a position-only source, 31 for the five-parameter
  solution, 95 for the six-parameter one. A 2-parameter source has no
  parallax and no proper motion at all, and this is the column that says so
  rather than leaving the reader to infer it from three empty cells.

Online refresh, offline runtime
-------------------------------

::

    online sync:
    NASA host identity -> gaia_dr3_id -> Gaia TAP -> validate -> atomic cache

    offline runtime:
    validated local cache -> AstrometricState -> Astropy propagation

Startup and rendering never touch the network. :func:`refresh_gaia_cache`
is the only function here that does, it is never called from the
application path, and a refresh that fails at any step leaves the previous
validated cache exactly where it was - the same contract
:mod:`astro_explorer.data.synchronizer` holds for the planet catalogue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import astropy.units as u
from astropy.time import Time

from ..coordinates.astrometry import AstrometricState
from ..coordinates.frames import sky_position
from ..provenance import Parameter, measured, unknown

__all__ = [
    "GAIA_TAP_SYNC_URL",
    "GAIA_DR3_TABLE",
    "GAIA_DR3_RELEASE",
    "GAIA_DR3_REFERENCE_EPOCH_JYEAR",
    "GAIA_DR3_TIME_SCALE",
    "GAIA_COLUMNS",
    "GAIA_CORRELATION_COLUMNS",
    "CACHE_SCHEMA_VERSION",
    "GaiaAstrometryRecord",
    "GaiaAstrometryCache",
    "GaiaIdentityError",
    "GaiaRefreshResult",
    "parse_gaia_dr3_id",
    "gaia_dr3_ids_from_catalog",
    "build_gaia_query",
    "fetch_gaia_astrometry",
    "validate_gaia_record",
    "astrometric_state_from_gaia",
    "DEFAULT_CACHE_PATH",
]

GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_DR3_TABLE = "gaiadr3.gaia_source"
GAIA_DR3_RELEASE = "Gaia DR3"

#: Gaia DR3's reference epoch, as documented. It is written here for
#: *validation* only - :class:`GaiaAstrometryRecord` stores the ``ref_epoch``
#: the archive actually returned, and a row disagreeing with this constant
#: is rejected rather than quietly corrected. A future data release changes
#: the epoch, and a hard-coded 2016.0 that had been used instead of the
#: column would keep propagating from the wrong year without a symptom.
GAIA_DR3_REFERENCE_EPOCH_JYEAR = 2016.0

#: Gaia times are TCB. The difference from TDB is about 20 s at J2016 plus a
#: drift of ~0.5 s/yr; on the fastest proper motion known that is under a
#: microarcsecond, but the scale is carried explicitly because a time
#: without one is what this whole slice exists to stop.
GAIA_DR3_TIME_SCALE = "tcb"

#: The astrometry itself, plus the uncertainty of every value. The errors
#: are fetched even though nothing propagates them yet: a cache that stored
#: only the values would have to be rebuilt from scratch the day something
#: does, and re-querying is the step that needs the network.
GAIA_COLUMNS = (
    "source_id",
    "ref_epoch",
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "radial_velocity",
    "radial_velocity_error",
    "astrometric_params_solved",
)

#: The astrometric correlation coefficients. RA, Dec, parallax and the two
#: proper-motion components are jointly fitted and strongly correlated, so
#: the covariance is not reconstructible from the five diagonal errors -
#: which is exactly what a later uncertainty propagation would need.
GAIA_CORRELATION_COLUMNS = (
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
)

#: Where the validated cache lives, relative to the repository root.
DEFAULT_CACHE_PATH = Path("data/reference_systems/gaia_dr3_astrometry.json")

#: ``hostname -> source_id``, written beside the cache by the sync script.
#: Separate from the cache because it is the *cross-match*, which belongs to
#: NASA, while the cache is the *astrometry*, which belongs to Gaia. Merging
#: them would make a re-run of either look like a change to both.
DEFAULT_HOST_MAP_PATH = Path("data/reference_systems/gaia_dr3_hosts.json")

#: ``Gaia DR3 2009481748875806976``. The archive is consistent about the
#: prefix, but the pattern accepts a bare number too, because the useful
#: content is the digits and a stricter match would reject a hand-entered
#: identifier for a cosmetic reason.
_GAIA_ID = re.compile(r"(?:gaia\s*dr3\s*)?(\d{5,25})\s*$", re.IGNORECASE)

#: Sanity bounds. These reject a column that changed meaning or unit; they
#: are not astrophysics. The proper-motion bound is an order of magnitude
#: above Barnard's Star, the fastest known at ~10.4 arcsec/yr.
_SANITY = {
    "ra": (0.0, 360.0),
    "dec": (-90.0, 90.0),
    "parallax": (-100.0, 1.0e4),
    "pmra": (-1.0e5, 1.0e5),
    "pmdec": (-1.0e5, 1.0e5),
    "radial_velocity": (-1.0e4, 1.0e4),
}

#: ``astrometric_params_solved`` values that include parallax and proper
#: motion. 3 is a two-parameter (position-only) solution.
_FIVE_PARAMETER = 31
_SIX_PARAMETER = 95

#: The cache document format. Bumped when the stored fields change meaning,
#: so an old document is refused rather than half-understood: a reader that
#: silently accepts a schema it does not know is how a renamed column
#: becomes an absent value becomes a missing measurement.
CACHE_SCHEMA_VERSION = 1


class GaiaIdentityError(ValueError):
    """The archive did not return exactly the sources that were asked for.

    An identity failure is better than a plausible nearby wrong star, so
    this is raised rather than worked around. In particular there is **no**
    cone-search fallback: if an exact NASA cross-match identifier fails to
    resolve, the honest outcome is no astrometry for that host, not the
    astrometry of whatever sits closest to a position whose epoch we could
    not state.
    """


def _finite(value: Any) -> float | None:
    """A finite float, or None for anything the archive left empty."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("null", "nan", "none", "--"):
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_gaia_dr3_id(value: Any) -> str | None:
    """``"Gaia DR3 2009481748875806976"`` -> ``"2009481748875806976"``.

    Deterministic, and returns the identifier as a **string**. A DR3
    ``source_id`` is a 64-bit integer that encodes a HEALPix index, and
    several of them exceed 2^53: round-tripping one through a float - which
    is what happens the moment it meets a pandas numeric column or a JSON
    reader that prefers numbers - silently changes which star it names.

    Returns None for anything that is not an identifier, so a row without a
    cross-match yields no astrometry rather than a nearby star's.
    """
    if value is None:
        return None
    match = _GAIA_ID.match(str(value).strip())
    if match is None:
        return None
    return match.group(1).lstrip("0") or "0"


def gaia_dr3_ids_from_catalog(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """``hostname -> Gaia DR3 source_id`` for every row that has one.

    Keyed by host rather than by planet: astrometry is a property of the
    star, and six TRAPPIST-1 rows carrying the same identifier should
    produce one cache entry and one query, not seven.
    """
    found: dict[str, str] = {}
    for row in rows:
        host = str(row.get("hostname") or "").strip()
        source_id = parse_gaia_dr3_id(row.get("gaia_dr3_id"))
        if not host or source_id is None:
            continue
        existing = found.get(host)
        if existing is not None and existing != source_id:
            # Two different identifiers for one host name is a cross-match
            # disagreement, not something to resolve by taking the last one.
            raise ValueError(
                "host {0!r} carries two Gaia DR3 identifiers ({1} and {2}); "
                "refusing to guess which star the astrometry belongs "
                "to".format(host, existing, source_id)
            )
        found[host] = source_id
    return found


@dataclass(frozen=True)
class GaiaAstrometryRecord:
    """One Gaia DR3 source, stored as the archive gave it.

    Nothing here is converted or defaulted. ``ref_epoch`` is the column, not
    :data:`GAIA_DR3_REFERENCE_EPOCH_JYEAR`; a missing ``radial_velocity`` is
    ``None`` and not ``0.0``. The interpretation happens once, in
    :func:`astrometric_state_from_gaia`, where it can be read against the
    rules it follows.
    """

    source_id: str
    ref_epoch: float | None = None
    ra: float | None = None
    ra_error: float | None = None
    dec: float | None = None
    dec_error: float | None = None
    parallax: float | None = None
    parallax_error: float | None = None
    pmra: float | None = None
    pmra_error: float | None = None
    pmdec: float | None = None
    pmdec_error: float | None = None
    radial_velocity: float | None = None
    radial_velocity_error: float | None = None
    astrometric_params_solved: int | None = None
    correlations: dict[str, float] = field(default_factory=dict)
    release: str = GAIA_DR3_RELEASE

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "GaiaAstrometryRecord":
        source_id = parse_gaia_dr3_id(row.get("source_id"))
        if source_id is None:
            raise ValueError("a Gaia row without a usable source_id is not a source")
        solved = _finite(row.get("astrometric_params_solved"))
        return cls(
            source_id=source_id,
            ref_epoch=_finite(row.get("ref_epoch")),
            ra=_finite(row.get("ra")),
            ra_error=_finite(row.get("ra_error")),
            dec=_finite(row.get("dec")),
            dec_error=_finite(row.get("dec_error")),
            parallax=_finite(row.get("parallax")),
            parallax_error=_finite(row.get("parallax_error")),
            pmra=_finite(row.get("pmra")),
            pmra_error=_finite(row.get("pmra_error")),
            pmdec=_finite(row.get("pmdec")),
            pmdec_error=_finite(row.get("pmdec_error")),
            radial_velocity=_finite(row.get("radial_velocity")),
            radial_velocity_error=_finite(row.get("radial_velocity_error")),
            astrometric_params_solved=None if solved is None else int(solved),
            correlations={
                name: _finite(row.get(name))
                for name in GAIA_CORRELATION_COLUMNS
                if _finite(row.get(name)) is not None
            },
        )

    @property
    def has_proper_motion(self) -> bool:
        return self.pmra is not None and self.pmdec is not None

    @property
    def reference_epoch(self) -> Time | None:
        """``ref_epoch`` as an Astropy time in Gaia's own TCB scale."""
        if self.ref_epoch is None:
            return None
        return Time(float(self.ref_epoch), format="jyear", scale=GAIA_DR3_TIME_SCALE)

    def as_dict(self) -> dict[str, Any]:
        record = {
            "source_id": self.source_id,
            "release": self.release,
            "ref_epoch": self.ref_epoch,
            "ra": self.ra,
            "ra_error": self.ra_error,
            "dec": self.dec,
            "dec_error": self.dec_error,
            "parallax": self.parallax,
            "parallax_error": self.parallax_error,
            "pmra": self.pmra,
            "pmra_error": self.pmra_error,
            "pmdec": self.pmdec,
            "pmdec_error": self.pmdec_error,
            "radial_velocity": self.radial_velocity,
            "radial_velocity_error": self.radial_velocity_error,
            "astrometric_params_solved": self.astrometric_params_solved,
        }
        if self.correlations:
            record["correlations"] = dict(sorted(self.correlations.items()))
        return record


def validate_gaia_record(record: GaiaAstrometryRecord) -> list[str]:
    """Everything wrong with one record. Empty means it may be cached.

    A rejected record is left out of the cache entirely rather than stored
    with a warning: the consumer of this cache builds absolute positions
    from it, and "cached but suspect" is a state nothing downstream has a
    way to act on.
    """
    errors: list[str] = []

    if not record.source_id.isdigit():
        errors.append("source_id {0!r} is not a Gaia identifier".format(record.source_id))

    if record.ref_epoch is None:
        errors.append("no ref_epoch; the reference epoch must be stored, not inferred")
    elif abs(record.ref_epoch - GAIA_DR3_REFERENCE_EPOCH_JYEAR) > 1e-6:
        errors.append(
            "ref_epoch is J{0:.4f}, not the documented J{1:.1f} for {2}; the "
            "release or the column has changed".format(
                record.ref_epoch, GAIA_DR3_REFERENCE_EPOCH_JYEAR, record.release
            )
        )

    if record.ra is None or record.dec is None:
        errors.append("a source without ra and dec is not a position")

    for name, (low, high) in _SANITY.items():
        value = getattr(record, name)
        if value is not None and not (low <= value <= high):
            errors.append(
                "{0} = {1:g} is outside the plausible range [{2:g}, {3:g}]; the "
                "column or its unit may have changed".format(name, value, low, high)
            )

    # One proper-motion component without the other is a motion at the wrong
    # position angle, and it looks exactly like a motion.
    if (record.pmra is None) != (record.pmdec is None):
        errors.append("exactly one proper-motion component is present; a pair or neither")

    solved = record.astrometric_params_solved
    if solved is None:
        errors.append("astrometric_params_solved is absent, so the solution is undescribed")
    elif record.has_proper_motion and solved not in (_FIVE_PARAMETER, _SIX_PARAMETER):
        errors.append(
            "proper motion is present but astrometric_params_solved = {0} "
            "describes a solution that did not fit one".format(solved)
        )

    return errors


def _check_identities(
    requested: Sequence[str], fetched: Sequence["GaiaAstrometryRecord"]
) -> None:
    """Every returned row must be one of the requested sources, exactly once.

    Three ways this fails and all three are refusals:

    * a returned ``source_id`` that was not asked for - the query matched
      something else, and caching it would file another star's astrometry
      under a host's name;
    * the same ``source_id`` twice - ``gaia_source`` is keyed on it, so a
      duplicate means the response is not what it claims to be, and picking
      one of the two would be a coin toss between two answers;
    * a requested id that came back empty is *not* an error here. A host
      genuinely absent from DR3 has no astrometry, and the rest of the
      batch should still be cached.
    """
    wanted = {str(s) for s in requested}
    seen: set[str] = set()
    for record in fetched:
        if record.source_id not in wanted:
            raise GaiaIdentityError(
                "the archive returned source_id {0}, which was not "
                "requested; refusing to cache another star's "
                "astrometry".format(record.source_id)
            )
        if record.source_id in seen:
            raise GaiaIdentityError(
                "source_id {0} was returned more than once; gaia_source is "
                "keyed on it, so the response is not what it claims to "
                "be".format(record.source_id)
            )
        seen.add(record.source_id)


def build_gaia_query(
    source_ids: Sequence[str], *, table: str = GAIA_DR3_TABLE
) -> str:
    """ADQL selecting the preserved astrometry for specific source ids.

    An ``in`` list on the primary key, not a cone search: the identifiers
    came from the archive's own cross-match, and this query's only job is to
    fetch what that cross-match already decided.
    """
    if not source_ids:
        raise ValueError("no Gaia source ids to query")
    for source_id in source_ids:
        if not str(source_id).isdigit():
            # An identifier reaching the query unparsed is how a string ends
            # up interpolated into ADQL.
            raise ValueError("{0!r} is not a Gaia source id".format(source_id))
    columns = ", ".join(GAIA_COLUMNS + GAIA_CORRELATION_COLUMNS)
    return "select {0} from {1} where source_id in ({2})".format(
        columns, table, ", ".join(str(s) for s in source_ids)
    )


def fetch_gaia_astrometry(
    source_ids: Sequence[str],
    *,
    session=None,
    timeout: float = 90.0,
    url: str = GAIA_TAP_SYNC_URL,
) -> list[GaiaAstrometryRecord]:
    """Download astrometry for ``source_ids`` from the Gaia TAP service.

    **Network access.** Nothing on the application path calls this; see
    :func:`refresh_gaia_cache`, which is the supported entry point and the
    one that keeps the previous cache when this fails.
    """
    import csv
    import io

    import requests

    query = build_gaia_query(source_ids)
    poster = session.post if session is not None else requests.post
    response = poster(
        url,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=timeout,
    )
    response.raise_for_status()

    rows = list(csv.DictReader(io.StringIO(response.text)))
    return [GaiaAstrometryRecord.from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# The validated local cache
# ---------------------------------------------------------------------------


def _checksum(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON form of the sources."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class GaiaRefreshResult:
    """What one refresh attempt did, and what the cache holds now."""

    outcome: str
    detail: str = ""
    written: int = 0
    rejected: dict[str, list[str]] = field(default_factory=dict)
    checksum: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome in ("committed", "unchanged")

    def describe(self) -> str:
        if self.outcome == "committed":
            return "Gaia astrometry cache updated: {0} source(s)".format(self.written)
        if self.outcome == "unchanged":
            return "Gaia astrometry cache already current"
        return "Gaia refresh {0}: {1}".format(self.outcome, self.detail)


class GaiaAstrometryCache:
    """The validated Gaia astrometry the application actually reads.

    Runtime is read-only and offline: :meth:`load` opens one JSON file and
    :meth:`state_for` turns an entry into an
    :class:`~astro_explorer.coordinates.astrometry.AstrometricState`. No
    method on this class other than :meth:`refresh` can reach the network,
    and :meth:`refresh` is not called during startup or rendering.
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)

    # -- offline runtime -------------------------------------------------
    @property
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any]:
        """The cache document, or an empty one when there is no cache yet.

        A missing cache is a normal state - the explorer runs without any
        astrometry, and every absolute position is blocked with a reason.
        It is not an error to be raised inside a render loop.
        """
        if not self.path.exists():
            return {
                "release": GAIA_DR3_RELEASE,
                "schema_version": CACHE_SCHEMA_VERSION,
                "sources": {},
            }
        with self.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict) or not isinstance(document.get("sources"), dict):
            raise ValueError(
                "{0} is not a Gaia astrometry cache document".format(self.path)
            )

        # Fail closed on a document this code does not understand. An
        # unrecognised schema or a different data release is refused rather
        # than read optimistically: the fields have the same names across
        # releases and different reference epochs behind them, so a
        # best-effort read would propagate from the wrong year in silence.
        version = document.get("schema_version")
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                "{0} declares cache schema {1!r}; this build reads {2}. "
                "Re-run scripts/sync_gaia_astrometry.py rather than reading "
                "it optimistically".format(self.path, version, CACHE_SCHEMA_VERSION)
            )
        release = document.get("release")
        if release != GAIA_DR3_RELEASE:
            raise ValueError(
                "{0} holds {1!r}, not {2}; the reference epoch and the "
                "column meanings differ between releases".format(
                    self.path, release, GAIA_DR3_RELEASE
                )
            )
        return document

    def records(self) -> dict[str, GaiaAstrometryRecord]:
        """``source_id -> record`` for everything cached."""
        document = self.load()
        found: dict[str, GaiaAstrometryRecord] = {}
        for source_id, entry in document.get("sources", {}).items():
            row = dict(entry)
            row.setdefault("source_id", source_id)
            # The correlations are nested in the document and flat in a TAP
            # row; flattening here means one constructor reads both.
            row.update(row.pop("correlations", None) or {})
            record = GaiaAstrometryRecord.from_row(row)
            found[record.source_id] = record
        return found

    def record_for(self, source_id: str | None) -> GaiaAstrometryRecord | None:
        if source_id is None:
            return None
        return self.records().get(str(source_id))

    def state_for(self, source_id: str | None, *, name: str = "") -> AstrometricState | None:
        """The cached source as an astrometric state, or None if not cached."""
        record = self.record_for(source_id)
        if record is None:
            return None
        return astrometric_state_from_gaia(record, name=name)

    # -- online refresh --------------------------------------------------
    def write(self, records: Iterable[GaiaAstrometryRecord]) -> str:
        """Replace the cache atomically. Returns the new checksum.

        The document is written to a sibling temporary file and renamed.
        ``os.replace`` is atomic within a filesystem on every platform this
        runs on, so a crash or a full disk mid-write leaves the previous
        cache intact rather than a truncated one that parses.
        """
        sources = {
            record.source_id: record.as_dict()
            for record in sorted(records, key=lambda r: r.source_id)
        }
        checksum = _checksum(sources)
        document = {
            "release": GAIA_DR3_RELEASE,
            "schema_version": CACHE_SCHEMA_VERSION,
            "table": GAIA_DR3_TABLE,
            "reference_epoch_jyear": GAIA_DR3_REFERENCE_EPOCH_JYEAR,
            "time_scale": GAIA_DR3_TIME_SCALE,
            "pm_ra_convention": "pmra is mu_alpha* = d(alpha)/dt cos(delta)",
            "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": checksum,
            "sources": sources,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        return checksum

    def refresh(
        self,
        source_ids: Sequence[str],
        *,
        fetcher=fetch_gaia_astrometry,
    ) -> GaiaRefreshResult:
        """One full refresh cycle: fetch, validate, replace atomically.

        **The only function in this module the application never calls.**
        Every failure mode ends with the previous cache still in place and
        still readable, because the alternative - an explorer that cannot
        show a position because a TAP service was down this morning - is
        worse than one showing yesterday's identical astrometry.
        """
        try:
            fetched = list(fetcher(source_ids))
        except Exception as exc:  # noqa: BLE001 - any transport failure is one outcome
            return GaiaRefreshResult(
                "download_failed",
                detail="{0}: {1}".format(type(exc).__name__, exc),
            )

        if not fetched:
            return GaiaRefreshResult(
                "rejected", detail="the query returned no sources"
            )

        # Identity first: a row that is not one of the requested sources is
        # a different star, and validating it would only establish that the
        # wrong star has good astrometry.
        try:
            _check_identities(source_ids, fetched)
        except GaiaIdentityError as exc:
            return GaiaRefreshResult("rejected", detail=str(exc))

        accepted: list[GaiaAstrometryRecord] = []
        rejected: dict[str, list[str]] = {}
        for record in fetched:
            errors = validate_gaia_record(record)
            if errors:
                rejected[record.source_id] = errors
            else:
                accepted.append(record)

        if not accepted:
            return GaiaRefreshResult(
                "rejected",
                detail="every returned source failed validation",
                rejected=rejected,
            )

        previous = ""
        try:
            previous = str(self.load().get("sha256") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            # An unreadable or unrecognised existing cache is exactly the
            # case a refresh is meant to repair, so the write proceeds. The
            # empty checksum makes sure it is never mistaken for "unchanged".
            previous = ""

        checksum = _checksum(
            {r.source_id: r.as_dict() for r in sorted(accepted, key=lambda r: r.source_id)}
        )
        if previous and previous == checksum:
            return GaiaRefreshResult(
                "unchanged", written=len(accepted), checksum=checksum, rejected=rejected
            )

        self.write(accepted)
        return GaiaRefreshResult(
            "committed", written=len(accepted), checksum=checksum, rejected=rejected
        )


def refresh_gaia_cache(
    cache: GaiaAstrometryCache,
    source_ids: Sequence[str],
    *,
    fetcher=fetch_gaia_astrometry,
) -> GaiaRefreshResult:
    """Module-level alias for :meth:`GaiaAstrometryCache.refresh`."""
    return cache.refresh(source_ids, fetcher=fetcher)


__all__ += ["refresh_gaia_cache"]


# ---------------------------------------------------------------------------
# Gaia record -> AstrometricState
# ---------------------------------------------------------------------------


def _gaia_parameter(
    value: float | None,
    error: float | None,
    unit,
    column: str,
    note: str = "",
) -> Parameter:
    """One Gaia column as a Parameter. Absent stays UNKNOWN, never zero."""
    if value is None:
        return unknown(
            unit,
            provenance="{0}.{1}".format(GAIA_DR3_TABLE, column),
            note=note or "not published in {0}".format(GAIA_DR3_RELEASE),
        )
    # Built directly rather than through ``measured`` so the convention note
    # travels on the parameter. ``pmra`` in particular has to say what it is
    # everywhere it is read, not only in this module's documentation.
    return replace(
        measured(
            float(value),
            unit,
            error_plus=error,
            error_minus=error,
            provenance="{0}.{1}".format(GAIA_DR3_TABLE, column),
            reference=GAIA_DR3_RELEASE,
        ),
        note=note,
    )


def _gaia_angle(value: float | None, error_mas: float | None, column: str) -> Parameter:
    """A Gaia coordinate: value in degrees, uncertainty in milliarcseconds.

    The two columns are in different units, which is easy to miss and
    catastrophic to get wrong in the direction that keeps the number small:
    ``ra_error`` read as degrees would make a 0.02 mas Gaia position look
    uncertain by more than a degree.
    """
    if value is None:
        return unknown(
            u.deg,
            provenance="{0}.{1}".format(GAIA_DR3_TABLE, column),
            note="not published in {0}".format(GAIA_DR3_RELEASE),
        )
    error_deg = (
        None if error_mas is None else float((float(error_mas) * u.mas).to_value(u.deg))
    )
    return measured(
        float(value),
        u.deg,
        error_plus=error_deg,
        error_minus=error_deg,
        provenance="{0}.{1}".format(GAIA_DR3_TABLE, column),
        reference=GAIA_DR3_RELEASE,
    )


def astrometric_state_from_gaia(
    record: GaiaAstrometryRecord, *, name: str = ""
) -> AstrometricState:
    """Interpret one cached Gaia source as an astrometric state.

    The three interpretations that matter, all in one place:

    * the reference epoch comes from ``ref_epoch``, in Gaia's TCB scale;
    * ``pmra`` becomes ``pm_ra_cosdec`` **unchanged**. Gaia's ``pmra`` is
      already :math:`\\dot\\alpha\\cos\\delta`, so this is an assignment and
      not a conversion, and writing it as one is the point: the tempting
      ``* cos(dec)`` here would be undetectable at low declination and
      wrong by a factor of two at 60 degrees;
    * an absent ``radial_velocity`` becomes an UNKNOWN parameter. Gaia
      publishes an RV for a minority of sources, and the majority do not
      thereby have an RV of zero. The distinction is what stops a
      direction-only source being propagated as a 3D position.

    The distance is left to
    :func:`~astro_explorer.coordinates.frames.distance_from_parallax`, which
    already refuses to turn a negative parallax - a real and common Gaia
    outcome for faint sources - into a position at the edge of the universe.
    """
    parallax = _gaia_parameter(record.parallax, record.parallax_error, u.mas, "parallax")
    position = sky_position(
        name or "Gaia DR3 {0}".format(record.source_id),
        _gaia_angle(record.ra, record.ra_error, "ra"),
        _gaia_angle(record.dec, record.dec_error, "dec"),
        parallax_mas=parallax,
    )

    return AstrometricState(
        position=position,
        source_catalog=GAIA_DR3_TABLE,
        source_id=record.source_id,
        release=record.release,
        reference_epoch=record.reference_epoch,
        pm_ra_cosdec=_gaia_parameter(
            record.pmra,
            record.pmra_error,
            u.mas / u.yr,
            "pmra",
            note="Gaia pmra is mu_alpha* = d(alpha)/dt cos(delta)",
        ),
        pm_dec=_gaia_parameter(record.pmdec, record.pmdec_error, u.mas / u.yr, "pmdec"),
        radial_velocity=_gaia_parameter(
            record.radial_velocity,
            record.radial_velocity_error,
            u.km / u.s,
            "radial_velocity",
        ),
        reference=GAIA_DR3_RELEASE,
    )


# ---------------------------------------------------------------------------
# Offline lookup by host name
# ---------------------------------------------------------------------------


def _candidate_paths(relative: Path, resources=None) -> list[Path]:
    """Where to look for a committed data file, in order.

    Mirrors :func:`~astro_explorer.app.vertical_slice.load_reference_catalog`
    so a frozen build finds the astrometry the same way it finds the
    catalogue snapshot and the spectra.
    """
    candidates: list[Path] = []
    if resources is not None:
        candidates.extend(Path(root) / relative for root in resources.roots())
    candidates.append(Path(__file__).resolve().parents[3] / relative)
    candidates.append(Path.cwd() / relative)
    return candidates


class GaiaHostIndex:
    """Host name -> Gaia DR3 astrometry, entirely from committed files.

    The offline runtime half of this module. It performs no network access
    of any kind: both files it reads are produced by
    ``scripts/sync_gaia_astrometry.py`` and committed, and a missing file is
    an empty index rather than an error - an explorer with no cached
    astrometry runs perfectly well and blocks every absolute position with a
    stated reason.
    """

    def __init__(
        self,
        cache: "GaiaAstrometryCache | None" = None,
        *,
        host_map_path: Path | str | None = None,
        resources=None,
    ):
        if cache is None:
            found = next(
                (c for c in _candidate_paths(DEFAULT_CACHE_PATH, resources) if c.exists()),
                None,
            )
            cache = GaiaAstrometryCache(found or DEFAULT_CACHE_PATH)
        self.cache = cache

        if host_map_path is None:
            host_map_path = next(
                (c for c in _candidate_paths(DEFAULT_HOST_MAP_PATH, resources) if c.exists()),
                DEFAULT_HOST_MAP_PATH,
            )
        self.host_map_path = Path(host_map_path)

    @property
    def hosts(self) -> dict[str, str]:
        """``hostname -> source_id``, or empty when no map is committed."""
        if not self.host_map_path.exists():
            return {}
        with self.host_map_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("{0} is not a host map".format(self.host_map_path))
        found: dict[str, str] = {}
        for host, value in document.items():
            source_id = parse_gaia_dr3_id(value)
            if source_id is not None:
                found[str(host)] = source_id
        return found

    def source_id_for(self, host_name: str) -> str | None:
        return self.hosts.get(str(host_name))

    def state_for_host(self, host_name: str) -> AstrometricState | None:
        """The host's astrometric state, or None when it is not cached.

        None is a normal answer. It means "this star's astrometry is not in
        the offline store", which is a different statement from "this star
        has no proper motion" - and the caller gets to keep them apart
        because it receives no state at all rather than an empty one.
        """
        source_id = self.source_id_for(host_name)
        if source_id is None:
            return None
        return self.cache.state_for(source_id, name=str(host_name))


__all__ += ["GaiaHostIndex", "DEFAULT_HOST_MAP_PATH"]
