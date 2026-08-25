# Explorer C4a Local Audit — HR Diagram Integration

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote baseline:** `a16fb5928ae0ccdf9f29a06ebaf69cc74d80880e`  
**Local C4a commit:** `7ccaf4c0105f061f1cdd9467fc7b1bb4645524ff`  
**Reported suite:** `1027 passed, 1 skipped`  
**Reported C4a suite:** `22 tests`  
**Verdict:** **C4a is architecturally strong and has no P0 physics blocker, but I would make two provenance/presentation fixes before push.**

---

## 1. What is already right

The basic architecture is the right one:

```text
StarRecord / stellar physics
        ↓
HRPlacement
        ↓
plot
```

rather than:

```text
plot
    ↓
recalculate stellar science
```

Having `HRPlacement` retain the selected star's actual `Parameter` objects is especially good because it keeps:

```text
value
unit
uncertainty
Status
provenance
reference
note
```

attached to the selected marker.

That makes the panel and marker consume the same scientific object rather than merely agreeing numerically.

---

## 2. Selected-marker provenance fix — PASS

The old remote implementation extracts:

```python
record.host.luminosity.value_in(u.L_sun)
```

and then decides whether to append `(derived)` by testing:

```python
record.host.luminosity.status.value == "DERIVED"
```

That loses most of the provenance model and uses a string comparison as a substitute for the enum.

The reported C4a replacement:

```text
HRPlacement retains Parameter
luminosity_is_derived is typed
```

is the correct repair.

Keep presentation logic based on `Status`, not string spelling.

---

## 3. Unknown selected stars — PASS

The old HR code simply omitted the selected marker if Teff or luminosity was unavailable.

That is visually ambiguous:

```text
star not plotted
```

could mean:

```text
selection bug
outside plot limits
missing data
invalid luminosity
```

The new policy:

```text
no invented coordinate
+
explicit reason annotation
```

is much better.

Keep unknown values as absent data, not plotting defaults.

---

## 4. Non-positive luminosity — PASS

Refusing:

```text
L <= 0
```

on a logarithmic luminosity axis is correct.

Do not clamp to epsilon.

That would turn:

```text
invalid / physically unusable input
```

into:

```text
a small but apparently real stellar luminosity
```

and would violate the project's missing-data rules.

---

## 5. One stellar-luminosity implementation — PASS

The old `population_arrays()` independently encoded:

\[
L/L_\odot = (R/R_\odot)^2(T/T_\odot)^4
\]

inside the plotting module.

Routing this through the same stellar-physics identity used elsewhere is the correct fix.

The strongest regression is not merely:

```text
outputs are close
```

but:

```text
population derivation is exactly the shared function's output
```

because the purpose of the test is architectural single-source-of-truth, not numerical tolerance.

---

## 6. `combined_status()` move — APPROVED

Moving the pessimistic status-combination rule into:

```text
provenance.py
```

is appropriate because it is not inspector-specific.

The dependency should remain:

```text
provenance semantics
    ↓
inspector
HR placement
future scientific views
```

rather than each feature inventing a private status lattice.

The inspector alias is acceptable for compatibility.

Do not create a second HR-specific combination rule.

---

# 7. Remaining issue 1 — background HR population still needs provenance separation

This is the most important remaining C4a issue.

The remote HR implementation's `population_arrays()` returns only numeric arrays:

```text
Teff
luminosity
radius
```

and its luminosity population mixes two fundamentally different sources:

```text
published NASA st_lum
```

with:

```text
luminosity derived from radius + Teff
```

The scatter then renders all valid points together as one undifferentiated population.

Your summary says the selected marker's provenance is fixed, but it does not say that the **background host population** now retains per-point luminosity provenance.

If `population_arrays()` still reduces the background to ordinary numeric arrays, then the chart still visually merges:

```text
catalogue luminosity
```

and:

```text
derived luminosity
```

even though C4a's scientific rule is that measured and derived quantities remain distinguishable.

### Required fix before push

Use one of these designs.

### Preferred

Create a population representation that carries status:

```python
@dataclass(frozen=True)
class HRPopulationPoint:
    hostname: str
    effective_temperature: Parameter
    luminosity: Parameter
```

or an array-oriented equivalent with a status mask.

Then draw, for example:

```text
catalog/published luminosity     one marker style
derived luminosity               a different marker style
```

Do not rely on colour alone.

### Minimal acceptable patch

Keep arrays, but also return a mask/status vector:

```text
luminosity_status
```

and perform separate scatter calls/legend entries for:

```text
published
derived
```

### Alternative

For C4a only, stop filling missing population luminosities and plot the background only where `st_lum` is published.

That is scientifically clean but loses useful context.

---

# 8. Remaining issue 2 — the hard-coded main-sequence guide needs provenance or explicit illustrative status

The currently pushed HR module contains a manually entered `_MAIN_SEQUENCE` table.

It is labelled:

```text
Main sequence (reference)
```

but the values have no scientific source/citation attached.

For a research-facing plot, a line of hand-entered stellar values should not look like another catalogued scientific series.

Before push, choose one:

### Best

Attach a documented source/reference to the guide and preserve it in the plot metadata/legend.

### Fastest safe MVP option

Relabel it explicitly:

```text
Illustrative main-sequence guide (not catalogue data)
```

and document that the points are approximate visualization context.

### Strictest

Set:

```python
show_main_sequence=False
```

by default until a sourced sequence/isochrone is added.

I would not leave the current unsourced wording `Main sequence (reference)` because a coordinator can reasonably ask:

> Reference to what?

and the software should already know the answer.

---

# 9. Main-sequence science should stay separate from the host population

When this is upgraded later, do not fit a "main sequence" from the exoplanet-host sample itself.

That sample is selection-biased and is not a stellar-population reference dataset.

Use a separately sourced:

```text
main-sequence calibration
isochrone
stellar-evolution grid
```

with its own citation and model metadata.

This is future work, not required beyond the disclosure/source fix above.

---

# 10. Error bars — recommended, not blocking C4a

The selected `Parameter` objects already carry asymmetric uncertainty.

The plot should eventually make use of:

```text
Teff uncertainty
luminosity uncertainty
```

with error bars or another explicit uncertainty representation.

This is especially valuable when luminosity is derived.

I do **not** consider it a C4a push blocker because the current milestone's goal is provenance-correct integration, and the panel still exposes uncertainty.

But I would add it before the coordinator-facing MVP freeze if the data are available.

---

# 11. Axis semantics — PASS

The existing stellar layer already distinguishes:

```text
Hertzsprung-Russell diagram:
    Teff vs luminosity

temperature-radius diagram:
    Teff vs radius
```

which fixes the original 2D program's naming problem.

Keep:

```text
hotter temperatures to the left
log luminosity on y
```

and explicit axis labels.

A logarithmic Teff x-axis is acceptable as long as the ticks are labelled in Kelvin and temperature direction is obvious.

---

# 12. Selection consistency — PASS based on reported tests

The C4a acceptance requirement that changing selection moves the selected marker without mutating stellar data is the right behavior.

The selected host should be identified by stable entity/catalog identity, not by population-array index.

Keep the selected marker entirely independent of background-population ordering and deduplication.

---

# 13. One future population-data hardening

The remote implementation de-duplicates the raw PS table with:

```python
drop_duplicates(subset=["hostname"])
```

which chooses one planet row as the representative stellar row.

That is probably adequate for the current snapshot, but it is not a strong long-term stellar-population model because stellar parameters can differ across rows/solutions.

Eventually, build the HR population from:

```text
one canonical StarRecord per host
```

or a dedicated stellar snapshot rather than from whichever planet row happens to appear first.

This is not a C4a blocker if the current tests verify host consistency.

---

# 14. Local verification — strong

Reported:

```text
1027 passed, 1 skipped
C4a: 22 tests
verify_gl.py: PASS
4 demos / 11 frames
```

The increase:

```text
1005 -> 1027
```

is consistent with a focused vertical slice.

The fact that two initially vacuous tests were rewritten before commit is also a good sign; keep watching for assertions that prove the setup changed rather than simply ending in `or True`-style success.

---

# 15. Push decision

Current status:

```text
selected-host science object         PASS
marker provenance                    PASS
unknown-marker disclosure            PASS
non-positive luminosity refusal       PASS
shared luminosity equation            PASS
combined status semantics             PASS
axis semantics                        PASS
selection behavior                    PASS

background-population provenance      FIX BEFORE PUSH
main-sequence guide provenance        FIX/DISCLOSE BEFORE PUSH
```

## **Do not push `7ccaf4c` yet.**

Amend the local C4a commit after closing those two presentation/provenance gaps.

Then rerun:

```text
C4a suite
full suite
verify_gl.py
demos
```

and send the revised C4a SHA and counts.

No remote-history rewrite is involved because C4a is still local.

---

# 16. After C4a

Once C4a is remotely green, proceed to:

```text
C4b — selected-host blackbody spectrum
```

Keep the same architecture:

```text
stellar Teff Parameter
    ↓
physics/radiation or stellar physics
    ↓
plot model
    ↓
UI
```

The blackbody plot should clearly say:

```text
idealized Planck continuum
```

rather than presenting itself as an observed stellar spectrum.
