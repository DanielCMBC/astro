# Explorer B: identity, panels and time

Review sections 5, 7 and 13. Explorer A could fly and select; this adds
knowing *what* was selected in a way that survives the catalogue changing
under it, and showing what is known about it without ever overstating it.

## Unknown position is not the Sun

The blocking fix. `Explorer.focus()` used to fall back to `np.zeros(3)`
when a host had no usable distance, which quietly converted

```
absolute host position = UNKNOWN
```

into

```
absolute host position = the solar origin
```

- exactly the class of substitution the whole project exists to prevent, and
the more insidious for looking like a sensible default.

Navigation now requires a position:

```python
explorer.focus("TRAPPIST-1", planets, star)
# UnknownSystemPositionError: TRAPPIST-1 has no usable absolute position,
# so it cannot be navigated to. Use open_detached() to inspect the system
# locally without claiming a galactic position.
```

### Detached systems

A system with no known galactic position is still perfectly usable
*locally*, so `open_detached()` opens it with no absolute claim at all. The
two operations are distinct:

| | `focus()` — fly to | `open_detached()` — inspect locally |
|---|---|---|
| needs an absolute position | yes | no |
| appears in the neighbourhood view | yes | no |
| has a distance from Earth | yes | **no** |
| can be flown to | yes | raises |
| star at `(0,0,0)` in its own frame | yes | yes |

The guarantee is structural rather than a convention in the explorer. A
`SystemFrame` built without a position is **unlocated**, and the coordinate
layer itself refuses to place it:

```python
frame.to_absolute_pc(frame.star_position())
# UnlocatedFrameError: TRAPPIST-1 frame [AU] (unlocated) has no known
# galactic position, so its coordinates cannot be expressed in absolute
# parsecs
```

`frame.contains(anything)` is `False` for an unlocated frame - it is not
*near* the Sun, it is nowhere - and `distance_to_system_pc()` returns
`None` rather than a number measured from a fiction.

## Stable entity keys

A catalogue name survives a scene rebuild but not a catalogue release.
Identity and naming are now separate:

```
entity_id     planet:nasa:HD_80606_b     what selection and picking use
display_name  HD 80606 b                 what a label draws
```

`RenderPlanet.identifier` carries the key; `RenderPlanet.label` carries the
name. Label decluttering matches its priority set against the **key**, so
renaming a body cannot cost it its pinned label.

The id is a pure function of the catalogue key, so it is recomputed rather
than stored and synchronised. Punctuation that carries meaning survives
normalisation (`2MASS J0437+2331`), and case is preserved because it is
significant - `K2-18 b` is a planet, `K2-18 B` would be a stellar companion.

## The generation token

The legacy prototype's async race: select A, select B, B's request
finishes, A's finishes later and overwrites B's panel with stale data.

Every selection change bumps a counter and the selection carries it:

```python
selection_a = explorer.select("star:nasa:HD_80606")
selection_b = explorer.select("planet:nasa:HD_80606_b", "planet")

explorer.is_current(selection_b)   # True  - still wanted
explorer.is_current(selection_a)   # False - discard the late result
```

## The panel model

A panel is a **model**, not a widget: rows of text and status that any front
end renders. Two invariants:

**UNKNOWN never becomes a numeric placeholder.** A row with no value has
`value is None` and reads "unknown". Never `0`, never a dash that could pass
for a measurement.

**Every scientific row exposes its provenance** — source column, status and
publication, carried on the row rather than in a footnote that can drift.

`Emphasis` says what a row *means* without naming a colour:

| Emphasis | Meaning |
|---|---|
| `MEASURED` | published |
| `DERIVED` | computed here by a documented relation |
| `ASSUMED` | a display placeholder — must be visually distinct |
| `UNKNOWN` | not known, not substituted |
| `PLAIN` | non-scientific text |

Assumptions are additionally collected into their own block, so a reader
scanning for measurements never has to separate them by eye:

```
DISPLAY ASSUMPTIONS (not measurements)
  Ascending node: not observable; normalised to 0 deg for display
```

Building a panel never mutates a record — `test_building_a_panel_does_not_mutate_the_record`
and `test_selecting_a_planet_does_not_change_the_scene_geometry` both check
it, so selecting a planet cannot perturb the orbit it is describing.

### Phase status is carried through

| System | Phase status | Emphasis |
|---|---|---|
| HD 80606 b | `CONSTRAINED` | `MEASURED` |
| Kepler-11 ×6 | `PARTIALLY_CONSTRAINED` | `DERIVED` |
| TRAPPIST-1 ×7 | `ASSUMED` | `ASSUMED` |

## Time controls

`TimeControls` owns a Julian date and hands it to
`OrbitalElements.phase_at()`. It integrates nothing itself and has no
notion of "one orbit per animation cycle".

```python
controls = TimeControls.for_system(records)   # starts at a published epoch
controls.play(); controls.advance(3.0)        # 3 s at the chosen rate
controls.step_fraction(period_days, 0.5)      # half of *this planet's* year
```

Starting from a published periastron or transit means the very first frame
is a position the ephemeris vouches for rather than an arbitrary date.
Where no epoch exists anywhere in the system, any date will do and the
phase reports itself `ASSUMED` regardless.

The default mode is physical. The normalised educational clock still
exists and still declares itself non-physical wherever it appears.

### The clock runs on one canonical axis

The date is `epoch_jd`, not `epoch_bjd`. It is a *full Julian date*, and
nothing in this class certifies it as `BJD_TDB` — so it is not named as
though it were. Only an epoch published in `BJD_TDB` is displayed under
that name:

```
Epoch:             JD 2458882.34400            # scale not stated
Epoch:             BJD_TDB 2458882.34400       # scale stated
```

A catalogue time reaches the clock only through `Epoch.canonical_jd`, which
applies the mission offset:

```
catalogue epoch
    ↓
Epoch + TimeScale
    ↓  canonical_jd:  BKJD +2454833,  BTJD +2457000
canonical full JD  +  residual scale uncertainty
    ↓
TimeControls.epoch_jd  /  .source_scale  /  .scale_uncertainty_days
    ↓
OrbitalElements.phase_at()
```

`phase_at()` goes through `elements.epoch_of(kind)` for exactly the same
reason, so both ends of the path are on the same axis. That equivalence is
the point, and it is tested directly: an epoch of `1000` BKJD and one of
`2455833` full JD give the same phase at the same instant
(`test_phase_is_identical_before_and_after_offset_normalisation`).

What the offset cannot absorb is carried rather than discarded. The
reference-frame light time and the leap seconds need a sky position and a
light-time to resolve, so `scale_uncertainty_days` reports them — about
568 s for an unstated scale (the geocentric term, ~499 s, plus leap
seconds), ~77 s for an `HJD_UTC` one (only the Sun's ~8 s barycentric
wobble plus leap seconds), zero for a stated barycentric one — and the
clock says so:

```
Time-system error: +/- 568 s; the published scale was not stated, so the
                   offset to BJD_TDB is unresolved
```

The leap-second half of that residual is evaluated at the epoch's own
date rather than read from a literal — see *TDB - UTC is a function of the
date* in `docs/orbital-semantics.md`.

### Which epoch starts the clock

Not "the first one in record order". The ranking is explicit:

1. the selected planet's own constrained epoch — `for_system(records,
   selected_name=...)`;
2. any planet's **periastron** epoch;
3. any planet's **transit** epoch;
4. the caller's `fallback_jd`.

A periastron epoch outranks a transit epoch because `M = 0` at periastron
directly, whereas a transit time reaches the mean anomaly through
`ν = π/2 − ω` and so inherits the argument-of-periastron convention
ambiguity. Equal-ranked candidates fall back to record order, so the choice
is deterministic. A `MEAN_ANOMALY_AT_EPOCH` is never a candidate: it is an
angle in radians, not a date, and `Epoch.is_dated` says so.

## Where the camera is

The camera is held in exactly one of two states, never a hybrid:

| State | Field | Set when |
|---|---|---|
| located | `_camera_absolute_pc` | the camera is anywhere at all |
| detached | `_camera_local` | the camera is inside an unlocated system |

They are mutually exclusive, so the parsec field's invariant — *this is an
absolute galactic position* — holds unconditionally. An earlier version
stored the detached camera's AU offset in the parsec field, scaled so the
arithmetic worked out; it produced correct pictures while quietly making
one field mean two things. A detached camera really is a
`FramedPosition([0, 0, 3], SystemFrame[AU])`, so that is what it is.

Absolute navigation is then refused *structurally*. `move_to_pc()`,
`approach()`, `enter_system()`, `leave_system()`, `path_to_system()` and
reading `camera_pc` all go through one `_require_located()` guard rather
than a check copied to each call site. `camera_absolute_pc` is the
non-raising query, and answers `None` — "nowhere" — rather than the Sun.
`move_to_local()` is the operation that *is* defined while detached, and
works in a located frame too.

## What "stable entity id" guarantees

Precisely this: **an entity id is stable while the authoritative catalogue
key remains unchanged.** A selection therefore survives a scene rebuild, an
LOD change and a change of display label, because none of those touch the
key.

It does *not* survive a canonical rename by the catalogue:
`planet:nasa:K2-18_b` and `planet:nasa:EPIC_201912552_b` are different ids
for the same planet, and nothing in `data/identity.py` can know that.
Closing that gap needs a persistent internal entity id — minted locally,
never derived from a name — with catalogue identifiers and aliases hanging
off it: NASA canonical name, Gaia `source_id`, SIMBAD identifiers. That
belongs with the offline synchronised catalogue, because it needs somewhere
durable to live.

## Acceptance criteria

| Criterion | Test |
|---|---|
| unknown absolute system position never becomes Sun position | `test_detached_system_never_appears_at_solar_origin` |
| detached local system cannot claim Earth/system distance | `test_detached_system_has_no_universe_distance` |
| selection uses stable entity key | `test_selection_is_by_stable_key_not_by_index` |
| display-name/alias changes do not invalidate selection | `test_display_name_changes_do_not_invalidate_selection` |
| UNKNOWN is never converted into a numeric UI placeholder | `test_unknown_never_becomes_a_numeric_placeholder` |
| every displayed scientific parameter exposes provenance | `test_every_scientific_row_exposes_provenance` |
| planet selection does not mutate scientific orbit state | `test_building_a_panel_does_not_mutate_the_record` |
| time control uses physical TimeController | `test_the_clock_drives_the_physical_propagator` |
| catalogue epochs reach the clock through `Epoch`/`TimeScale` | `test_bkjd_epoch_gets_2454833_day_offset`, `test_btjd_epoch_gets_2457000_day_offset` |
| `phase_at()` is on the same time axis as the clock | `test_phase_at_uses_same_epoch_scale_as_time_controls`, `test_phase_is_identical_before_and_after_offset_normalisation` |
| an unstated scale keeps its uncertainty and is not called `BJD_TDB` | `test_jd_unspecified_keeps_scale_uncertainty`, `test_unspecified_jd_is_not_labelled_exact_bjd_tdb` |
| the starting-epoch policy is explicit and deterministic | `test_the_selected_planet_anchors_the_clock`, `test_a_periastron_epoch_outranks_a_transit_epoch` |
| a detached camera is a local `FramedPosition`, not a parsec carrier | `test_a_detached_camera_is_a_local_framed_position` |
| absolute navigation is refused while detached | `test_every_absolute_navigation_method_rejects_detached_mode` |
| entity ids claim catalogue-key stability, not rename-proofness | `test_entity_ids_are_catalog_key_stable_not_rename_proof` |
| constrained and assumed phases are visually distinct | `test_an_assumed_phase_is_shown_as_assumed` |
| selection survives LOD and scene rebuilds | `test_selection_identity_survives_lod_transitions` |
| async panel updates carry a selection/generation token | `test_a_stale_async_result_is_rejected` |
| full test suite remains green | the full pytest suite |
| headless GL CI still produces verified frames | the `opengl` job |

## Deferred

Review section 6 asks for `active_coordinate_frame` to be separated from a
`presentation_state`, so a crossfade can begin before or after the numerical
frame switches. It explicitly does not block Explorer B, and it is not done
here: the frame is still both. It should be done before the final timed
free-flight camera, and the presentation state must never feed back into
coordinate interpretation when it is.
