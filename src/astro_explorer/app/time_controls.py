"""Interactive time, driven by the physical propagator.

Review section 13.6. The explorer needs play, pause, rate and step - but
the clock underneath must stay the one the science uses. This class owns a
barycentric Julian date and hands it to
:meth:`OrbitalElements.phase_at`; it does not integrate anything itself and
it has no notion of "one orbit per animation cycle".

The educational normalised mode still exists in
:class:`~astro_explorer.physics.ephemeris.TimeController`, and is still
labelled non-physical wherever it appears. What this class adds is that the
*default* is a real date advancing at a real rate, so the thing on screen
answers "where is it" rather than "roughly what does it do".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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


@dataclass
class TimeControls:
    """A barycentric Julian date the user can drive.

    Attributes
    ----------
    epoch_bjd:
        The instant currently being shown. Every propagation in the
        explorer is evaluated at this date, so what is on screen and what
        the panel reports can never disagree.
    rate_days_per_second:
        How fast simulated time runs while playing.
    playing:
        Whether :meth:`advance` moves the clock.
    """

    epoch_bjd: float = 2458882.344
    rate_days_per_second: float = 1.0
    playing: bool = False
    mode: TimeMode = TimeMode.SCALED

    #: Optional bounds, so a UI slider has ends. None means unbounded.
    span_days: float | None = None
    _origin_bjd: float = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._origin_bjd is None:
            self._origin_bjd = self.epoch_bjd

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
            self.epoch_bjd += float(real_seconds) * self.rate_days_per_second
        return self.epoch_bjd

    def step_days(self, days: float) -> float:
        """Nudge the clock, whether or not it is playing."""
        self.epoch_bjd += float(days)
        return self.epoch_bjd

    def step_fraction(self, period_days: float | None, fraction: float) -> float:
        """Step by a fraction of an orbital period.

        The natural unit for inspecting one planet: a quarter of *its* year
        rather than an arbitrary number of days.
        """
        if period_days is None or period_days <= 0.0:
            return self.epoch_bjd
        return self.step_days(period_days * float(fraction))

    def seek(self, epoch_bjd: float, *, rebase: bool = False) -> float:
        """Jump to a date.

        ``rebase`` also moves the reference the elapsed time is measured
        from, which is what setting a *starting* epoch means; scrubbing
        within a session should leave it alone.
        """
        self.epoch_bjd = float(epoch_bjd)
        if rebase:
            self._origin_bjd = self.epoch_bjd
        return self.epoch_bjd

    def reset(self) -> float:
        self.epoch_bjd = self._origin_bjd
        return self.epoch_bjd

    # -- presentation ----------------------------------------------------
    @property
    def rate_label(self) -> str:
        for label, rate in RATE_PRESETS:
            if abs(rate - self.rate_days_per_second) < 1e-9:
                return label
        return "{0:g} day/s".format(self.rate_days_per_second)

    def set_rate(self, days_per_second: float) -> None:
        self.rate_days_per_second = float(days_per_second)

    def offset_days(self) -> float:
        """Days elapsed since the clock was started or reset."""
        return self.epoch_bjd - self._origin_bjd

    def phase_fraction(self, period_days: float | None) -> float | None:
        """Where in its period a planet is, as a fraction, for a UI dial."""
        if period_days is None or period_days <= 0.0:
            return None
        return float(np.mod(self.offset_days(), period_days) / period_days)

    def describe(self) -> list[str]:
        lines = [
            "Epoch:             BJD {0:.5f}".format(self.epoch_bjd),
            "Rate:              {0} ({1})".format(
                self.rate_label, "playing" if self.playing else "paused"
            ),
            "Elapsed:           {0:+.4f} d since the start epoch".format(self.offset_days()),
        ]
        if self.mode is TimeMode.NORMALIZED:
            lines.append(
                "Mode:              normalised educational clock - NOT physical time"
            )
        else:
            lines.append("Mode:              physical ephemeris time")
        return lines

    @classmethod
    def for_system(cls, records, *, fallback_bjd: float = JD_UNIX_EPOCH) -> "TimeControls":
        """Start at an epoch the system actually publishes, if it has one.

        Starting at a published periastron or transit means the very first
        frame is a position the ephemeris vouches for, rather than an
        arbitrary date.
        """
        import astropy.units as u

        for record in records:
            elements = record.elements
            for parameter in (elements.epoch_periastron, elements.epoch_transit):
                if parameter.is_known:
                    return cls(epoch_bjd=parameter.value_in(u.day))

        # No published epoch anywhere in the system: any date is as good as
        # another, and the phase will be reported as assumed regardless.
        return cls(epoch_bjd=fallback_bjd)
