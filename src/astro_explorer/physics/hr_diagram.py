"""Explorer C4a: where a star sits on the Hertzsprung-Russell diagram.

The plotting code for both stellar diagrams already existed. What did not
exist was a *model* between the stellar physics and the plot, and the gap
had the same shape as the one C3 closed for distances:

    the thing on screen recomputed, reformatted and silently dropped
    scientific values, so what the plot showed and what the panel said
    could disagree without either being obviously wrong.

Three concrete symptoms, all of them in the old ``draw_hr_diagram``:

* it read ``record.host.luminosity.value_in(u.L_sun)`` and plotted the
  number, losing the :class:`~astro_explorer.provenance.Status` that says
  whether anyone measured it;
* it decided "is this derived?" by comparing ``status.value`` to the string
  ``"DERIVED"``, which is a spelling test rather than a type test;
* a star with an unknown temperature, an unknown luminosity, or a
  non-positive one was **silently omitted**. Nothing appeared, and nothing
  said why nothing appeared - which reads as "this star is not interesting"
  rather than "this measurement does not exist".

So this module is the single crossing point, and it produces a
:class:`HRPlacement` rather than a pair of floats. The plot draws what it is
given and computes nothing.

Why the axes are stated rather than assumed
-------------------------------------------

An HR diagram is not a scatter plot of two stellar columns. It has a
convention, it is old, and it is backwards:

* the abscissa is **effective temperature, increasing to the left**. This
  is historical - it began as spectral type O B A F G K M - and a plot that
  puts hot stars on the right is a temperature-luminosity scatter, not an
  HR diagram;
* the ordinate is **luminosity relative to the Sun, logarithmic**. The main
  sequence spans about eight decades, so a linear axis shows one star and a
  band along the bottom.

:data:`TEFF_AXIS` and :data:`LUMINOSITY_AXIS` write both down. A
colour-magnitude diagram - apparent colour against magnitude, with
magnitudes increasing *downward* - is a different plot that looks similar,
and quietly substituting one for the other is the failure this names.

What this module refuses to do
------------------------------

It never computes a luminosity or a temperature. Both arrive as
:class:`~astro_explorer.provenance.Parameter` objects that some catalogue
published or that
:func:`~astro_explorer.physics.stellar.luminosity_from_radius_and_teff`
derived, and this module places them. That is why it takes parameters
rather than a ``StarRecord``: the record lives in the data layer, the
golden rule keeps physics out of it, and passing the parameters through
means the plot and the panel are looking at *the same objects* rather than
at two readings of the same source.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from ..provenance import Parameter, Status, combined_status, unknown
from .stellar import DiagramKind, luminosity_ratio_from_radius_and_teff

__all__ = [
    "TEFF_AXIS",
    "LUMINOSITY_AXIS",
    "TEFF_NOT_PUBLISHED",
    "LUMINOSITY_NOT_PUBLISHED",
    "LUMINOSITY_NOT_POSITIVE",
    "HRPlacement",
    "hr_placement",
    "HRPopulation",
    "hr_population",
    "MAIN_SEQUENCE_GUIDE",
    "MAIN_SEQUENCE_GUIDE_LABEL",
    "MAIN_SEQUENCE_GUIDE_DISCLOSURE",
    "DiagramKind",
]

#: The abscissa convention, in words, because a plot that gets it backwards
#: is still a perfectly readable plot of something else.
TEFF_AXIS = (
    "effective temperature in K, increasing to the *left* - the historical "
    "O B A F G K M ordering. Hot stars on the right would be a "
    "temperature-luminosity scatter, not an HR diagram."
)

#: The ordinate convention. Logarithmic is not a display preference: the
#: main sequence covers roughly eight decades of luminosity.
LUMINOSITY_AXIS = (
    "luminosity in L_sun on a base-10 logarithmic axis, so one decade is "
    "one unit of spacing everywhere on the axis"
)

TEFF_NOT_PUBLISHED = (
    "no effective temperature is published for this star, so it has no "
    "position along the temperature axis"
)
LUMINOSITY_NOT_PUBLISHED = (
    "no luminosity is published and none could be derived from a radius and "
    "a temperature, so this star has no position along the luminosity axis"
)
LUMINOSITY_NOT_POSITIVE = (
    "the luminosity is not positive, so it has no logarithm and cannot be "
    "placed on a logarithmic axis"
)


@dataclass(frozen=True)
class HRPlacement:
    """One star's place on the HR diagram, or the reason it has none.

    Holds the *parameters*, not copies of their numbers, so a caller can ask
    the placement anything it could have asked the star - the uncertainty,
    the provenance tag, the reference - and get the same answer the info
    panel gives, because it is the same object.

    A placement that cannot be drawn is still a placement. It carries the
    reasons in :attr:`blockers`, which is what lets a plot say "no published
    temperature" in the corner instead of silently showing one fewer point.
    """

    name: str
    effective_temperature: Parameter
    luminosity: Parameter
    blockers: tuple[str, ...] = ()

    @property
    def is_plottable(self) -> bool:
        """True when both coordinates exist. Says nothing about provenance."""
        return not self.blockers

    @property
    def teff_k(self) -> float | None:
        """The abscissa in kelvin, or None. Never a substitute."""
        if TEFF_NOT_PUBLISHED in self.blockers:
            return None
        return self.effective_temperature.value_in(u.K)

    @property
    def luminosity_solar(self) -> float | None:
        """The ordinate in solar luminosities, or None."""
        if not self.is_plottable and (
            LUMINOSITY_NOT_PUBLISHED in self.blockers
            or LUMINOSITY_NOT_POSITIVE in self.blockers
        ):
            return None
        return self.luminosity.value_in(u.L_sun)

    @property
    def log_luminosity(self) -> float | None:
        """``log10(L / L_sun)``, the quantity the axis is linear in.

        Exposed so a caller can check the axis spacing directly rather than
        reading it off a rendered figure: one decade is one unit here, at
        every luminosity.
        """
        value = self.luminosity_solar
        if value is None or value <= 0.0:
            return None
        return float(np.log10(value))

    @property
    def status(self) -> Status:
        """What the *placement* may claim, not what either input claims.

        Pessimistic in the usual way: a marker drawn from an assumed
        luminosity sits at a perfectly definite height on the axis, and
        nothing about a dot says it was invented.
        """
        return combined_status(self.effective_temperature, self.luminosity)

    @property
    def is_scientific(self) -> bool:
        """True when the marker may be read as a measurement."""
        return self.is_plottable and self.status.is_scientific

    @property
    def luminosity_is_derived(self) -> bool:
        """True when the luminosity was computed rather than published.

        A typed question, replacing the old string comparison against
        ``"DERIVED"``. Most exoplanet hosts land here: the archive publishes
        ``st_lum`` for some and a radius and temperature for the rest.
        """
        return self.luminosity.status is Status.DERIVED

    def label(self, digits: int = 3) -> str:
        """One line naming the star, its coordinates and their provenance."""
        if not self.is_plottable:
            return "{0}: not placeable".format(self.name or "unknown star")

        text = "{0}: {1:.0f} K, {2:.{3}g} L_sun".format(
            self.name or "unknown star", self.teff_k, self.luminosity_solar, digits
        )
        if self.luminosity_is_derived:
            text += " (luminosity derived)"
        elif self.status is Status.ASSUMED_FOR_VISUALIZATION:
            text += " (assumed for visualisation)"
        return text

    def describe(self) -> list[str]:
        """Lines that never hide a substitution or a missing measurement."""
        lines = [
            "Star:               {0}".format(self.name or "unknown"),
            "Effective temp.:    {0}".format(self.effective_temperature.format()),
            "Luminosity:         {0}".format(self.luminosity.format()),
        ]
        if self.is_plottable:
            lines.append(
                "log10(L/L_sun):     {0:+.3f}".format(self.log_luminosity)
            )
        for reason in self.blockers:
            lines.append("Not placed:         {0}".format(reason))
        return lines


# ---------------------------------------------------------------------------
# The background population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HRPopulation:
    """The host-star background, with each point's provenance kept.

    A scatter of every known exoplanet host gives a single selected star
    context, and thousands of rows cannot each afford a
    :class:`~astro_explorer.provenance.Parameter`. So this is array-oriented
    - but it is **not** three bare numeric arrays, because the luminosities
    in it come from two different places:

    * the archive's ``st_lum``, a published value;
    * :func:`~astro_explorer.physics.stellar.luminosity_ratio_from_radius_and_teff`,
      derived from a radius and a temperature.

    Drawing those as one undifferentiated cloud says every dot is the same
    kind of thing, which is the same claim the selected marker was making
    before C4a and is wrong for the same reason. ``luminosity_status``
    carries the distinction per point so the figure can keep them apart.

    The arrays are parallel and the same length; a row with no usable
    luminosity is ``UNKNOWN`` and excluded by :attr:`plottable`.
    """

    hostname: np.ndarray
    effective_temperature_k: np.ndarray
    luminosity_solar: np.ndarray
    luminosity_status: np.ndarray
    radius_solar: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.hostname),
            len(self.effective_temperature_k),
            len(self.luminosity_solar),
            len(self.luminosity_status),
            len(self.radius_solar),
        }
        if len(lengths) > 1:
            raise ValueError(
                "an HR population's arrays must be parallel; got lengths {0}".format(
                    sorted(lengths)
                )
            )

    def __len__(self) -> int:
        return int(len(self.hostname))

    @property
    def plottable(self) -> np.ndarray:
        """Mask of points that have both coordinates and a positive L."""
        return (
            np.isfinite(self.effective_temperature_k)
            & np.isfinite(self.luminosity_solar)
            & (self.luminosity_solar > 0.0)
        )

    def with_status(self, status: Status) -> np.ndarray:
        """Mask of plottable points whose luminosity has ``status``."""
        return self.plottable & (self.luminosity_status == status)

    @property
    def published(self) -> np.ndarray:
        """Mask of points whose luminosity the archive published."""
        return self.with_status(Status.MEASURED)

    @property
    def derived(self) -> np.ndarray:
        """Mask of points whose luminosity came from radius and temperature."""
        return self.with_status(Status.DERIVED)

    def counts(self) -> dict[Status, int]:
        """How many plottable points of each provenance, for a legend."""
        return {
            status: int(np.count_nonzero(self.with_status(status)))
            for status in (Status.MEASURED, Status.DERIVED)
        }


def hr_population(
    hostname,
    effective_temperature_k,
    radius_solar,
    published_log_luminosity,
) -> HRPopulation:
    """Build the background population, recording where each L came from.

    Takes plain arrays rather than a catalogue frame, so the physics layer
    decides the *tiering* and the caller does the dataframe plumbing. The
    rule is one line and it is the whole point of the type:

    ==============================  ==========================================
    the archive gave                the luminosity is
    ==============================  ==========================================
    ``st_lum``                      MEASURED - published, used unchanged
    a radius and a temperature      DERIVED - Stefan-Boltzmann, computed here
    neither                         UNKNOWN - no point is drawn
    ==============================  ==========================================

    ``published_log_luminosity`` is ``log10(L/L_sun)``, which is the form the
    archive publishes it in. Converting it here rather than at the call site
    keeps the one place that knows the column's units.
    """
    names = np.asarray(hostname, dtype=object)
    teff = np.asarray(effective_temperature_k, dtype=np.float64)
    radius = np.asarray(radius_solar, dtype=np.float64)
    log_lum = np.asarray(published_log_luminosity, dtype=np.float64)

    published = np.isfinite(log_lum)
    luminosity = np.where(published, np.power(10.0, np.where(published, log_lum, 0.0)), np.nan)

    # Fill the rest from radius and temperature, through the one
    # Stefan-Boltzmann implementation rather than a second copy of it.
    derivable = ~published & np.isfinite(teff) & np.isfinite(radius)
    if np.any(derivable):
        luminosity = np.where(
            derivable, luminosity_ratio_from_radius_and_teff(radius, teff), luminosity
        )

    status = np.full(names.shape, Status.UNKNOWN, dtype=object)
    status[published] = Status.MEASURED
    status[derivable & np.isfinite(luminosity)] = Status.DERIVED

    return HRPopulation(
        hostname=names,
        effective_temperature_k=teff,
        luminosity_solar=luminosity,
        luminosity_status=status,
        radius_solar=radius,
    )


# ---------------------------------------------------------------------------
# The main-sequence guide
# ---------------------------------------------------------------------------

#: Approximate main-sequence points as ``(Teff K, L/L_sun, R/R_sun)``.
#:
#: These are hand-entered textbook-scale values kept so that a single
#: selected star has some context, and they are **not** catalogue data. No
#: citation is attached because none can honestly be: they were not taken
#: from a specific published table, fitted, or derived from a stellar model
#: in this repository.
#:
#: They are deliberately *not* fitted from the exoplanet-host sample either.
#: That sample is selection-biased - it is the stars people chose to survey
#: for planets - so a "main sequence" drawn through it would be a property
#: of the survey rather than of stellar structure.
MAIN_SEQUENCE_GUIDE = (
    (2800.0, 0.0018, 0.18),
    (3300.0, 0.015, 0.32),
    (3800.0, 0.06, 0.50),
    (4400.0, 0.16, 0.70),
    (5200.0, 0.42, 0.85),
    (5772.0, 1.00, 1.00),
    (6200.0, 1.75, 1.15),
    (7000.0, 4.5, 1.40),
    (8500.0, 18.0, 1.80),
    (10000.0, 55.0, 2.40),
)

#: What the legend says. "Main sequence (reference)" invited the question
#: "reference to what?", and the software should already know the answer -
#: so the label answers it instead: this line is illustrative context, not a
#: series a reader may measure against.
MAIN_SEQUENCE_GUIDE_LABEL = "Illustrative main-sequence guide (not catalogue data)"

MAIN_SEQUENCE_GUIDE_DISCLOSURE = (
    "Approximate hand-entered main-sequence points, drawn only to give a "
    "single selected star some context. They carry no catalogue provenance "
    "and no citation, and they are not fitted from the exoplanet-host "
    "sample - that sample is selection-biased. Replacing them with a sourced "
    "isochrone or stellar-evolution grid, with its own reference and model "
    "metadata, is future work."
)


def hr_placement(
    name: str,
    effective_temperature: Parameter | None,
    luminosity: Parameter | None,
) -> HRPlacement:
    """Place one star, or say why it cannot be placed.

    Takes the star's own parameters - the same objects
    :func:`~astro_explorer.app.panel.build_star_panel` puts in the info
    panel - so the marker and the panel cannot disagree about a number.
    Nothing here recomputes either of them.

    Every reason is collected rather than the first, because a star with no
    temperature *and* no luminosity is missing two things and a plot that
    says one of them invites the reader to think the other is fine.
    """
    teff = effective_temperature if effective_temperature is not None else unknown(u.K)
    lum = luminosity if luminosity is not None else unknown(u.L_sun)

    blockers: list[str] = []
    if not teff.is_known:
        blockers.append(TEFF_NOT_PUBLISHED)

    if not lum.is_known:
        blockers.append(LUMINOSITY_NOT_PUBLISHED)
    else:
        value = lum.value_in(u.L_sun)
        if value is None or not np.isfinite(value):  # pragma: no cover - is_known
            blockers.append(LUMINOSITY_NOT_PUBLISHED)
        elif value <= 0.0:
            # A negative or zero luminosity is not a faint star; it is a bad
            # row. Plotting it would need a logarithm that does not exist,
            # and clamping it to a small positive number would invent a
            # brightness.
            blockers.append(LUMINOSITY_NOT_POSITIVE)

    return HRPlacement(
        name=name,
        effective_temperature=teff,
        luminosity=lum,
        blockers=tuple(blockers),
    )
