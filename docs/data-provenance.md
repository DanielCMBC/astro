# Data and provenance

## The `Parameter` type

Every scientific quantity is a `Parameter` carrying value, unit, asymmetric
uncertainty, source, reference, retrieval date and an explicit `Status`
(roadmap section 12):

| Status | Meaning |
|---|---|
| `MEASURED` | published in a catalogue or paper |
| `DERIVED` | computed here from measured values by a documented relation |
| `ASSUMED_FOR_VISUALIZATION` | not known; a placeholder so the scene can be drawn |
| `UNKNOWN` | not known and not substituted |

Rules the type enforces on its own:

* a NaN value degrades to `UNKNOWN`, so a missing number can never present
  itself as a measurement;
* a negative lower error is normalised to a positive magnitude, because IPAC
  and NASA write it as a negative number;
* `require(unit)` raises rather than substituting, for call sites where
  inventing a number would corrupt the science;
* `format()` appends the status label to anything that is not `MEASURED`, so
  a UI cannot accidentally hide an assumption.

## Record-selection policy

The `ps` table holds one row per *published solution*. The original program
queried it and then ran `drop_duplicates(subset=["pl_name"])`, which keeps
whichever reference happened to sort first and mixes parameters across
papers.

Two concepts are now distinct and never interchangeable (roadmap 3.2):

| Policy | Table | Meaning |
|---|---|---|
| `DEFAULT_SOLUTION` | `ps` where `default_flag = 1` | one internally consistent published solution per planet |
| `COMPOSITE` | `pscomppars` | maximally populated, but columns come from different references |

The default-solution filter happens in ADQL, not in pandas, so the row is
the archive's documented choice rather than an artefact of sort order. The
active policy and its caveat are shown in the Data & Provenance tab.

## Missing-value policy

| Missing | What happens |
|---|---|
| semimajor axis | derived from `P` and `M*` (`DERIVED`), else `UNKNOWN`. Never 1 AU. |
| eccentricity | stays `UNKNOWN`. `for_display()` may assume 0, tagged and explained. |
| inclination, argument of periastron | stay `UNKNOWN`; normalised to 0 only for display. |
| ascending node | never published for exoplanets; always normalised for display. |
| orbital phase | no epoch means no current position; the orbit is drawn without a body on it. |
| parallax non-positive | `UNKNOWN`, or a catalogue distance if one exists. Never a placeholder distance. |
| stellar mass | luminosity, `T_eq` and derived axis all stay `UNKNOWN`. |

`tests/test_missing_data.py` ends with a guard that walks a record built
from an almost-empty row and asserts that no field silently equals an
Earth-like default.

## Offline-first synchronisation

```
connectivity -> download -> stage -> validate columns -> validate units
-> validate row counts and identifiers -> reject malformed
-> record timestamps -> hash -> atomic replace
```

Validation rejects a download when:

* a mandatory column is missing;
* fewer planets are present than a truncated download would suggest;
* planet names are duplicated;
* a default-solution download contains non-default rows;
* more than 1% of a column's values fall outside a plausible range, which is
  how a silent unit change announces itself;
* the planet count fell by more than 10% against the working snapshot.

The commit writes to a temporary file and renames, so a crash mid-write
cannot leave a truncated snapshot. Every attempt is recorded in `sync_log`.

## Local store

* **SQLite** (`catalog.sqlite`) - snapshot metadata, synchronisation log,
  per-field provenance.
* **Feather/Arrow** - the dense catalogue table.
* **`tables/`** - IPAC atmospheric spectra.
* **`molecular_evidence.csv`** - structured detection evidence.

## Molecular evidence

`molecular_evidence.csv` replaces the flat `planet_molecules.csv`. Each row
records planet, molecule, `detection_status`, instrument, facility,
publication, reference URL, retrieval method, confidence, date and notes.

`DetectionStatus.asserts_presence` is true only for `DETECTED`. A
`TENTATIVE`, `DISPUTED` or `UPPER_LIMIT` claim is displayed under its own
heading and is never phrased as "this molecule is present" - the K2-18 b
dimethyl sulfide claim is the worked example, recorded as `DISPUTED`.

Legacy three-column rows still load, but import as `UNKNOWN` rather than
being promoted to detections.
