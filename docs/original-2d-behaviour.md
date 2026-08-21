# The original 2D application, documented

Roadmap Phase 0 asks for the original program's behaviour to be recorded
before it is changed. The file itself is preserved verbatim at
`legacy/exoplanet_analyzer_original.py`.

## What it was

A single 700-line Tkinter module, `ExoplanetScientificSuite`, that on startup
downloaded the NASA `pscomppars` table over TAP, cached it to
`exoplanet_cache.feather`, and presented five tabs: Overview, Orbit, HR
Diagram, Black Body and Atmospheric Spectra.

## What it did correctly

These concepts survive into the new architecture unchanged in spirit:

* NASA Exoplanet Archive retrieval over TAP with a feather cache fallback;
* host-star then planet selection;
* eccentric Keplerian orbit geometry, `r = a(1-e^2) / (1 + e cos nu)`;
* a genuine Newton solver for Kepler's equation (12 fixed iterations);
* Planck's law with `np.expm1`, and Wien's displacement law;
* a stellar population scatter plot with the temperature axis inverted;
* local IPAC atmospheric tables plotted with asymmetric error bars;
* a molecule lookup keyed on planet name;
* PyInstaller packaging.

Note that this file had already been partly corrected relative to the state
the roadmap describes: it queried `pscomppars` rather than `ps`, it addressed
spectrum columns by name from the `|`-delimited header rather than by fixed
position, and the HR tab already plotted luminosity. The roadmap's P0 items
3.1, 3.2 and 3.6 were therefore partially addressed here; what remained is
listed below.

## What was wrong, and where it is fixed

| # | Original behaviour | Where it lived | Now |
|---|---|---|---|
| 3.1 | Column *index* looked up from the header, but the data row then split on whitespace - see below; `FACILITY` and `SPEC_TYPE` discarded; units assumed to be microns and percent rather than read | `parse_spectrum_table` | `spectroscopy/ipac.py` reads via `astropy.table` `ascii.ipac`, honours the declared units, and keeps all metadata |
| 3.2 | `pscomppars` used with no statement that its columns mix references | `CATALOG_QUERY` | `data/nasa_archive.py` `SolutionPolicy`, with the caveat displayed |
| 3.3 | `draw_orbit` used `parse_float(..., default=1.0)`, so a planet with no published axis was drawn on a 1 AU orbit | `draw_orbit` | derived from Kepler's third law and marked `DERIVED`, or left `UNKNOWN` and not drawn |
| 3.4 | `df["pl_orbeccen"].fillna(0.0)`; the Overview tab then printed that 0 as though measured | `prepare_catalog` | stays `UNKNOWN`; the circle is a labelled display assumption |
| 3.5 | `mean_anomaly = 2 pi frame / 160`; every planet completed an orbit in 6.4 s regardless of period | `draw_orbit.update` | `TimeController` with `REAL`, `SCALED` and `NORMALIZED` modes |
| 3.6 | Tab titled "HR Diagram"; the radius plot was absent | `setup_hr_tab` | both diagrams, each correctly named, in their own tabs |
| 3.7 | Titled "Black Body Spectrum" with no caveat | `draw_black_body` | labelled "Ideal blackbody approximation" with the reason |
| 3.8 | `H_PLANCK`, `C_LIGHT`, `K_BOLTZMANN`, `B_WIEN` typed as literals | module header | `astropy.constants`; a test forbids those literals reappearing |
| 3.9 | Units implicit in variable names | throughout | `astropy.units` on every `Parameter` |
| 3.10 | `dependencies.bat` installed `tenacity` (unused) and omitted SciPy and Astropy | `dependencies.bat` | `pyproject.toml` is authoritative; the `.bat` wraps it |
| 3.11 | `datas` bundled `tables` and `atmospheric_signatures.json` but **not** `planet_molecules.csv`, which `load_local_assets` needs; a frozen build therefore always fell back to the five-row table hard-coded in the source | `.spec`, `load_local_assets` | `DECLARED_RESOURCES` is the single list, and the spec generates `datas` from it |
| 3.12 | A `messagebox` popup headed "Atmospheric detections" listing molecules as fact | `run_analysis` | `molecular_evidence.csv` with detection status, instrument and reference |

### The spectrum parser worked by luck

`parse_spectrum_table` found each column's *index* from the `|`-delimited
header, which looks safe, but then did `parts = line.split()` on the data
row. IPAC tables are fixed-width, and several columns hold free text with
spaces in it - `PL_TRANDEP_AUTHORS` is `de Mooij et al. 2014`, and
`PL_TRANDEP_URL` is a path. In the sample table that produces **36
whitespace tokens for 29 declared columns**, so every index past the first
text field points at the wrong value:

```
CENTRALWAVELNG   idx=0    token=0.47350     correct
PL_TRANDEP       idx=2    token=0.03240     correct
PL_RATROR        idx=8    token=0.01800     correct
PL_RADJ          idx=14   token=et          garbage
ST_RAD           idx=24   token=null        garbage
```

The five columns the program actually read all sit at indices 0-4, ahead of
any text field, so the plots came out right. Reading one more column, or
meeting a table whose first columns differ, would have produced silent
nonsense. `astropy.table.Table.read(..., format="ascii.ipac")` respects the
fixed-width layout, so the class of bug is gone rather than avoided.

### Two further issues found while reading the file

* **eccentricity clipped to 0.98.** `solve_kepler` did
  `np.clip(eccentricity, 0.0, 0.98)`, silently changing a genuinely more
  eccentric orbit into a different one. The new solver raises on `e >= 1`
  and solves everything below it.
* **`pl_flux_earth` mixed measured and derived values in one column.**
  `df["pl_insol"].where(notna, derived_flux)` produced a column where some
  cells were published insolation and others were computed from a derived
  luminosity and a possibly-fabricated axis, with no way to tell which. The
  new `PlanetRecord.insolation` returns a `Parameter` that says.

## Behaviour deliberately not carried over

* The modal popup on every analysis. Evidence now lives in its own tab.
* `blit=False` `FuncAnimation` over a fixed 160-frame cycle.
* Reading `planet_molecules.csv` with a hard-coded five-row fallback table
  embedded in the source.
