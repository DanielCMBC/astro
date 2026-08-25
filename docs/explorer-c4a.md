# Explorer C4a: the HR diagram integration

C1 asked whether the 3D scene could display a derived *region* without
duplicating the physics that produced it. C2 asked whether it could display
an *orientation* without claiming one. C4a asks the same question of a
**plot**, and the answer had been quietly "no" for as long as the plot has
existed.

The plotting code was already there, and already correctly named — the
original program's temperature-radius scatter is not an HR diagram, and each
has had its own function and title since. What did not exist was a model
between the stellar physics and the figure.

## Three silent symptoms

None of these produced a wrong-looking picture. That is what made them worth
fixing.

**The marker lost its provenance.** The old code read
`record.host.luminosity.value_in(u.L_sun)` and plotted the number. Almost no
exoplanet host has a *published* luminosity — the archive gives a radius and
a temperature, and `luminosity_from_radius_and_teff` derives the rest — so a
dot on the diagram was usually a derived quantity presented exactly like a
measured one.

**"Is this derived?" was a spelling test.** It compared
`status.value == "DERIVED"`, a string comparison standing in for a type
test. Rename the enum member and the label silently stops appearing.

**An unmeasured star vanished.** A host with no temperature, no luminosity,
or a non-positive one was omitted with no annotation. Nothing appeared, and
nothing said why — which a reader parses as "not interesting" rather than
"not measured".

**And the plot recomputed stellar physics.** `population_arrays` carried its
own `R² (T/T_sun)⁴` alongside the Stefan-Boltzmann function in
`physics/stellar.py`. Two copies of one identity agree until one is edited.

**The background population flattened its provenance.** Worse than the
duplicated formula, and found by the local audit: the scatter returned three
bare numeric arrays, so the published `st_lum` luminosities and the derived
ones were drawn as one undifferentiated cloud. That is the same claim the
selected marker was making — every dot is the same kind of thing — repeated
for every other host in the catalogue.

**And the main-sequence guide posed as data.** A hand-entered table labelled
`Main sequence (reference)`, sitting next to thousands of catalogue points
with no citation behind it. "Reference to what?" is a fair question and the
software should already know the answer.

## The model

`physics/hr_diagram.py` is the single crossing point. `hr_placement()`
returns an `HRPlacement` holding the star's own `Parameter` objects — not
copies of their numbers — so a caller can ask it the uncertainty, the
provenance tag or the reference and get the same answer the info panel
gives, because it *is* the same object.

```
StarRecord.effective_temperature ─┐
                                  ├─→ hr_placement() ─→ HRPlacement ─→ figure
StarRecord.luminosity ────────────┘
```

It takes parameters rather than a `StarRecord` because the record lives in
the data layer and the golden rule keeps physics out of it. The plot
computes nothing.

| the placement says | when |
|---|---|
| `is_plottable` | both coordinates exist |
| `teff_k` / `luminosity_solar` | the coordinates, or `None` — never a substitute |
| `log_luminosity` | `log10(L/L_sun)`, the quantity the axis is linear in |
| `status` | pessimistic: an assumed input makes the whole placement assumed |
| `luminosity_is_derived` | a typed question, not a string comparison |
| `blockers` | why it has no place, when it has none |

A placement that cannot be drawn is still a placement. The figure annotates
the reasons instead of showing one fewer point.

## The background keeps its provenance too

`HRPopulation` is the array-oriented counterpart. Thousands of rows cannot
each afford a `Parameter`, so it is arrays — but not *bare* arrays:
`luminosity_status` carries per point where that luminosity came from.

| the archive gave | the luminosity is |
|---|---|
| `st_lum` | `MEASURED` — published, converted from log10 once, used unchanged |
| a radius and a temperature | `DERIVED` — Stefan-Boltzmann, computed here |
| neither | `UNKNOWN` — no point is drawn |

The two are drawn with **different markers**, not different colours. Colour
is the first thing lost to a greyscale print, a projector or a colour-blind
reader, and the distinction being carried is scientific rather than
decorative — the same reason the C2 orientation overlay uses a dash pattern
instead of a hue.

`population_arrays()` still exists for callers that only want numbers, and
its docstring says plainly that it has flattened the provenance out and that
anything which *draws* the population should use `population_for()`.

The tiering itself lives in `physics/hr_diagram.py`, not in the plot: which
kind of luminosity a row has is a scientific judgement. This module's share
is dataframe plumbing — pull the columns, de-duplicate so a system with
eight planets is one point, hand plain arrays to `hr_population()`.

## The main-sequence guide is illustrative, and says so

`MAIN_SEQUENCE_GUIDE` moved into the physics layer, where both diagrams read
one table, and it is now labelled:

```
Illustrative main-sequence guide (not catalogue data)
```

drawn dashed rather than solid. `MAIN_SEQUENCE_GUIDE_DISCLOSURE` records
what it is: approximate hand-entered points, no catalogue provenance, no
citation — because none can honestly be attached. They were not taken from a
specific published table, fitted, or derived from a stellar model here.

They are also deliberately **not** fitted from the exoplanet-host sample.
That sample is selection-biased — it is the stars people chose to survey for
planets — so a "main sequence" drawn through it would be a property of the
survey rather than of stellar structure.

Replacing it with a sourced isochrone or stellar-evolution grid, carrying
its own reference and model metadata, is future work. Until then the label
answers the question rather than raising it.

## The axes are stated, not assumed

An HR diagram is not a scatter of two stellar columns. It has a convention,
it is old, and it is backwards:

* **`TEFF_AXIS`** — effective temperature, increasing to the **left**. The
  ordering is historical: O B A F G K M. A plot with hot stars on the right
  is a temperature-luminosity scatter that looks like an HR diagram;
* **`LUMINOSITY_AXIS`** — luminosity in `L_sun`, base-10 logarithmic. The
  main sequence spans about eight decades, so a linear axis shows one star
  and a smear along the bottom.

A colour-magnitude diagram — apparent colour against magnitude, with
magnitudes increasing *downward* — is a different plot that looks similar,
and `test_a_colour_magnitude_diagram_is_not_what_this_draws` pins that this
is not quietly one.

## Refusals

```
no effective temperature  → no x coordinate, and the reason travels
no luminosity             → no y coordinate
L ≤ 0                     → refused, not clamped
```

The last one matters: a zero or negative luminosity is a bad catalogue row,
not a very faint star. It has no logarithm, and nudging it to a small
positive number to make it plottable would invent a brightness.

A star missing two things reports two, the same rule the absolute-position
gates follow.

## What travelled with this slice

Three cleanups the C3.6 remote audit asked to fold into the next feature
commit rather than push on their own:

* the HD 219134 proper-motion figure in `coordinates/astrometry.py` said
  `0.14 mas` for a decade of 2.1 arcsec/yr motion. It is **21 arcsec**, and
  since one arcsec at one parsec is one AU, the 136 AU conclusion beside it
  was right all along;
* `roadmap-status.md` still said `data/gaia.py` was absent and that only the
  NASA TAP fetcher existed. C3.6 shipped both;
* `EXPLORER_C36_PUSH_APPROVAL.md` moved into `docs/reviews/archive/`.

And one that belongs to C4a itself: `combined_status()` now lives in
`provenance.py` beside `Status`. The coordinate inspector had a private copy
of the pessimistic rule and the HR placement needed the same one — two
copies would eventually disagree about the same pair of inputs.

## Where it lives

| module | role |
|---|---|
| `physics/hr_diagram.py` | `HRPlacement`, `HRPopulation`, the axis conventions, the main-sequence guide |
| `physics/stellar.py` | `luminosity_ratio_from_radius_and_teff` — one Stefan-Boltzmann identity, scalar or array |
| `ui/plots/hr.py` | draws the placement; owns no science |
| `provenance.py` | `combined_status` — the shared pessimistic rule |
| `tests/regression/test_explorer_c4a.py` | the slice's specification |

## What comes after

Two things this slice names and does not fix, both flagged by the local
audit as non-blocking:

* **error bars.** The selected star's `Parameter` objects already carry
  asymmetric uncertainties on both axes, and the marker does not yet show
  them. Worth adding before an MVP freeze, especially where the luminosity
  is derived;
* **one canonical star per host.** The population de-duplicates the planet
  table with `drop_duplicates(subset=["hostname"])`, which picks whichever
  planet row appears first as the representative stellar row. Stellar
  parameters can differ between published solutions, so a dedicated stellar
  snapshot would be a stronger long-term population model.

C4b is the blackbody spectrum, against the existing Planck implementation as
the single source of truth. C4c reconnects the corrected IPAC spectroscopy
to the selected planet, keeping measurement, interpretation and annotation
in three separate layers — a molecule label must never become a detected
molecule because a wavelength falls near a known band.
