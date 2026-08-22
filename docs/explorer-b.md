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

`TimeControls` owns a barycentric Julian date and hands it to
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
