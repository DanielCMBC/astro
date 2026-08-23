"""Interactive time, driven by the physical propagator.

Review section 13.6. The explorer needs play, pause, rate and step - but
the clock underneath must stay the one the science uses. This class owns a
Julian date and hands it to :meth:`OrbitalElements.phase_at`; it does not
integrate anything itself and it has no notion of "one orbit per animation
cycle".

The educational normalised mode still exists in
:class:`~astro_explorer.physics.ephemeris.TimeController`, and is still
labelled non-physical wherever it appears. What this class adds is that the
*default* is a real date advancing at a real rate, so the thing on screen
answers "where is it" rather than "roughly what does it do".

The clock's canonical axis
--------------------------
Explorer B review section 2. The date this class holds is called
``epoch_jd``, not ``epoch_bjd``, because it is a *full Julian date* and
nothing here certifies it as ``BJD_TDB``. A catalogue time only reaches
this clock through :class:`~astro_explorer.physics.epoch.Epoch`, whose
:attr:`~astro_explorer.physics.epoch.Epoch.canonical_jd` applies the
mission offset - ``+2454833`` for Kepler's BKJD, ``+2457000`` for TESS's
BTJD - that reading the raw parameter would have skipped by thirteen and
seven years respectively.

What the offset cannot absorb travels with the clock instead of being
discarded: :attr:`TimeControls.source_scale` records which convention the
epoch was published in, and
:attr:`TimeControls.scale_uncertainty_days` records how far from
``BJD_TDB`` it could still be. Both are zero-cost when the scale is stated
and honest when it is not.

Starting-epoch policy
---------------------
Review section 10. :meth:`TimeControls.for_system` no longer takes the
first epoch it stumbles over in record order. The ranking is explicit:

1. the selected planet's own constrained epoch, when a planet is selected;
2. any planet's periastron epoch, the anchor that needs no omega;
3. any planet's transit epoch;
4. the caller's fallback date.

A periastron epoch outranks a transit epoch because ``M = 0`` at
periastron directly, whereas a transit time reaches the mean anomaly
through ``nu = pi/2 - omega`` and so inherits the argument-of-periastron
convention ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.epoch import Epoch, EpochKind, TimeScale
from ..physics.ephemeris import JD_UNIX_EPOCH, TimeMode

__all__ = ["TimeControls", "RATE_PRESETS"]

#: Simulated days per real second, as a UI would offer them.
RATE_PRESETS = (
    ("1 hour/s", 1.0 / 24.0),
    ("1 day/s", 1.0),
    ("1 week/s", 7.0),
    ("1 month/s", 30.0),
    ("1 year/s", 365.25),
)



#: How directly each dated epoch kind yields a mean anomaly. Lower is a
#: better clock anchor; anything unlisted sorts last.
_KIND_RANK = {
    EpochKind.PERIASTRON: 0,
    EpochKind.TRANSIT: 1,
    EpochKind.MEAN_ANOMALY_AT_EPOCH: 2,
}


@dataclass
class TimeControls:
    """A Julian date the user can drive.

    Attributes
    ----------
    epoch_jd:
        The instant currently being shown, as a full Julian date. Every
        propagation in the explorer is evaluated at this date, so what is on
        screen and what the panel reports can never disagree. It is
        deliberately *not* called ``epoch_bjd``: unless :attr:`source_scale`
        is already ``BJD_TDB``, nothing here has converted it to barycentric
        dynamical time.
    source_scale:
        The :class:`~astro_explorer.physics.epoch.TimeScale` the starting
        epoch was published in. ``UNKNOWN`` when the clock was started from
        a fallback date rather than a catalogue epoch.
    scale_uncertainty_days:
        How far the canonical date could still be from true ``BJD_TDB``,
        after the mission offset has been applied. Zero when the scale is
        stated barycentric; about 568 s when the archive did not say, and
        about 77 s for an ``HJD_UTC`` epoch, which has already had the
        Earth's orbital light time removed. Evaluated at the epoch's own
        date, so the leap-second term is the one that applied then.
    source_kind:
        Which orbital event the starting epoch marks, so a panel can say
        "started at a published periastron" rather than only quote a number.
    rate_days_per_second:
        How fast simulated time runs while playing.
    playing:
        Whether :meth:`advance` moves the clock.
    """

    epoch_jd: float = 2458882.344
    rate_days_per_second: float = 1.0
    playing: bool = False
    mode: TimeMode = TimeMode.SCALED

    #: Provenance of the starting epoch (review section 2). These say what
    #: the number on the clock actually is; they are never used to silently
    #: shift it a second time.
    source_scale: TimeScale = TimeScale.UNKNOWN
    scale_uncertainty_days: float = 0.0
    source_kind: EpochKind = EpochKind.UNKNOWN
    source_name: str = ""

    #: Optional bounds, so a UI slider has ends. None means unbounded.
    span_days: float | None = None
    _origin_jd: float = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._origin_jd is None:
            self._origin_jd = self.epoch_jd

    # -- transport -------------------------------------------------------
    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def advance(self, real_seconds: float) -> float:
        """Move the clock forward by wall-clock time, if playing."""
        if self.playing:
            self.epoch_jd += float(real_seconds) * self.rate_days_per_second
        return self.epoch_jd

    def step_days(self, days: float) -> float:
        """Nudge the clock, whether or not it is playing."""
        self.epoch_jd += float(days)
        return self.epoch_jd

    def step_fraction(self, period_days: float | None, fraction: float) -> float:
        """Step by a fraction of an orbital period.

        The natural unit for inspecting one planet: a quarter of *its* year
        rather than an arbitrary number of days.
        """
        if period_days is None or period_days <= 0.0:
            return self.epoch_jd
        return self.step_days(period_days * float(fraction))

    def seek(self, epoch_jd: float, *, rebase: bool = False) -> float:
        """Jump to a date, given as a full Julian date.

        ``rebase`` also moves the reference the elapsed time is measured
        from, which is what setting a *starting* epoch means; scrubbing
        within a session should leave it alone.
        """
        self.epoch_jd = float(epoch_jd)
        if rebase:
            self._origin_jd = self.epoch_jd
        return self.epoch_jd

    def seek_epoch(self, epoch: Epoch, *, rebase: bool = True) -> float:
        """Jump to a published :class:`Epoch`, offset and metadata included.

        The scale-aware counterpart of :meth:`seek`: it is impossible to
        pass a BKJD number in here and have it land thirteen years early,
        because the conversion happens from the epoch's own scale.
        """
        canonical = epoch.canonical_jd
        if canonical is None:
            return self.epoch_jd
        self.source_scale = epoch.scale
        self.scale_uncertainty_days = epoch.scale_uncertainty_days
        self.source_kind = epoch.kind
        return self.seek(canonical, rebase=rebase)

    def reset(self) -> float:
        self.epoch_jd = self._origin_jd
        return self.epoch_jd

    # -- presentation ----------------------------------------------------
    @property
    def rate_label(self) -> str:
        for label, rate in RATE_PRESETS:
            if abs(rate - self.rate_days_per_second) < 1e-9:
                return label
        return "{0:g} day/s".format(self.rate_days_per_second)

    @property
    def epoch_label(self) -> str:
        """The clock's date, named as precisely as its scale allows.

        Only an epoch published as ``BJD_TDB`` is displayed under that name.
        Anything else is a full Julian date and says so.
        """
        prefix = "BJD_TDB" if self.source_scale is TimeScale.BJD_TDB else "JD"
        return "{0} {1:.5f}".format(prefix, self.epoch_jd)

    def set_rate(self, days_per_second: float) -> None:
        self.rate_days_per_second = float(days_per_second)

    def offset_days(self) -> float:
        """Days elapsed since the clock was started or reset."""
        return self.epoch_jd - self._origin_jd

    def phase_fraction(self, period_days: float | None) -> float | None:
        """Where in its period a planet is, as a fraction, for a UI dial."""
        if period_days is None or period_days <= 0.0:
            return None
        return float(np.mod(self.offset_days(), period_days) / period_days)

    def describe(self) -> list[str]:
        lines = [
            "Epoch:             {0}".format(self.epoch_label),
            "Rate:              {0} ({1})".format(
                self.rate_label, "playing" if self.playing else "paused"
            ),
            "Elapsed:           {0:+.4f} d since the start epoch".format(self.offset_days()),
        ]
        if self.source_kind is not EpochKind.UNKNOWN:
            lines.append(
                "Started at:        {0}{1} ({2})".format(
                    self.source_kind.label,
                    " of {0}".format(self.source_name) if self.source_name else "",
                    self.source_scale.label,
                )
            )
        else:
            lines.append(
                "Started at:        no published epoch in this system; "
                "arbitrary fallback date"
            )
        if self.scale_uncertainty_days > 0.0:
            lines.append(
                "Time-system error: +/- {0:.0f} s; the published scale was not "
                "stated, so the offset to BJD_TDB is unresolved".format(
                    self.scale_uncertainty_days * 86400.0
                )
            )
        if self.mode is TimeMode.NORMALIZED:
            lines.append(
                "Mode:              normalised educational clock - NOT physical time"
            )
        else:
            lines.append("Mode:              physical ephemeris time")
        return lines

    # -- starting epoch --------------------------------------------------
    @staticmethod
    def _ranked_epochs(records, selected_name: str | None) -> list:
        """Published epochs of a system, best starting anchor first.

        Implements the section-10 policy. Every candidate is an
        :class:`Epoch`, so whatever wins already knows its own time scale
        and cannot reach the clock as a bare number.
        """
        candidates = []
        for index, record in enumerate(records):
            name = getattr(record, "name", "") or record.elements.name
            for epoch in record.elements.epochs:
                if not (epoch.is_known and epoch.is_dated):
                    continue
                # Lower sorts first: the selected planet outranks everything,
                # then the epoch kinds in order of how directly they give a
                # mean anomaly - periastron needs nothing, a transit needs
                # omega, a mean-anomaly reference date needs the published
                # angle beside it. Record order is the last tiebreak, so the
                # result is deterministic.
                rank = (
                    0 if selected_name is not None and name == selected_name else 1,
                    _KIND_RANK.get(epoch.kind, len(_KIND_RANK)),
                    index,
                )
                candidates.append((rank, name, epoch))

        candidates.sort(key=lambda item: item[0])
        return [(name, epoch) for _, name, epoch in candidates]

    @classmethod
    def for_system(
        cls,
        records,
        *,
        selected_name: str | None = None,
        fallback_jd: float = JD_UNIX_EPOCH,
    ) -> "TimeControls":
        """Start at an epoch the system actually publishes, if it has one.

        Starting at a published periastron or transit means the very first
        frame is a position the ephemeris vouches for, rather than an
        arbitrary date.

        The epoch travels as an :class:`Epoch`, so its mission offset is
        applied and its residual scale uncertainty is carried onto the clock
        rather than dropped (review sections 2 and 10). Pass
        ``selected_name`` to anchor on the planet the user is actually
        looking at, which is the most intuitive starting instant.
        """
        best = next(iter(cls._ranked_epochs(records, selected_name)), None)
        if best is None:
            # No published epoch anywhere in the system: any date is as good
            # as another, and the phase reports itself assumed regardless.
            return cls(epoch_jd=fallback_jd)

        name, epoch = best
        return cls(
            epoch_jd=epoch.canonical_jd,
            source_scale=epoch.scale,
            scale_uncertainty_days=epoch.scale_uncertainty_days,
            source_kind=epoch.kind,
            source_name=name,
        )
