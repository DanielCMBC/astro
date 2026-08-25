"""Provenance-aware scientific parameters (roadmap sections 3.2 and 12).

Every important scientific quantity carries its value, unit, asymmetric
uncertainty, source and an explicit :class:`Status`.  A value that was
invented so that something could be drawn is never indistinguishable from a
measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

import astropy.units as u

__all__ = [
    "Status",
    "Parameter",
    "combined_status",
    "unknown",
    "measured",
    "derived",
    "assumed",
]


class Status(str, Enum):
    """Epistemic status of a scientific parameter.

    The four states are mandated by roadmap sections 3.3, 3.4, 4.4 and 12.
    """

    MEASURED = "MEASURED"
    """Published value taken from a catalogue or paper."""

    DERIVED = "DERIVED"
    """Computed from other measured values by a documented relation."""

    ASSUMED_FOR_VISUALIZATION = "ASSUMED_FOR_VISUALIZATION"
    """Not known; a placeholder chosen only so the scene can be drawn."""

    UNKNOWN = "UNKNOWN"
    """Not known and not substituted."""

    @property
    def is_scientific(self) -> bool:
        """True when the value may be quoted as science."""
        return self in (Status.MEASURED, Status.DERIVED)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_LABELS = {
    Status.MEASURED: "measured",
    Status.DERIVED: "derived",
    Status.ASSUMED_FOR_VISUALIZATION: "assumed for visualisation",
    Status.UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class Parameter:
    """A single scientific quantity with full provenance.

    Parameters
    ----------
    value:
        Magnitude in ``unit``.  ``None`` (or NaN) means "not available".
    unit:
        An :mod:`astropy.units` unit, or ``u.dimensionless_unscaled``.
    error_plus, error_minus:
        Asymmetric uncertainties, always stored as non-negative magnitudes in
        ``unit``.  IPAC/NASA tables give the lower error as a negative number;
        the constructor normalises it.
    status:
        See :class:`Status`.
    provenance:
        Short machine-readable tag, e.g. ``"nasa:pscomppars.pl_orbsmax"`` or
        ``"kepler3(pl_orbper, st_mass)"``.
    reference:
        Human-readable publication or catalogue reference.
    retrieved:
        When the value entered the local store.
    note:
        Free text shown next to assumptions.
    """

    value: float | None
    unit: u.UnitBase = u.dimensionless_unscaled
    error_plus: float | None = None
    error_minus: float | None = None
    status: Status = Status.UNKNOWN
    provenance: str = ""
    reference: str | None = None
    retrieved: date | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise NaN -> None so is_known has a single meaning everywhere.
        value = self.value
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            else:
                if not math.isfinite(value):
                    value = None
        object.__setattr__(self, "value", value)

        for name in ("error_plus", "error_minus"):
            err = getattr(self, name)
            if err is None:
                continue
            try:
                err = abs(float(err))
            except (TypeError, ValueError):
                err = None
            else:
                if not math.isfinite(err):
                    err = None
            object.__setattr__(self, name, err)

        # A parameter without a value can only be UNKNOWN.
        if value is None and self.status is not Status.UNKNOWN:
            object.__setattr__(self, "status", Status.UNKNOWN)

        if isinstance(self.unit, str):
            object.__setattr__(self, "unit", u.Unit(self.unit))

    # -- state -----------------------------------------------------------
    @property
    def is_known(self) -> bool:
        """True when a numeric value is present."""
        return self.value is not None

    @property
    def is_scientific(self) -> bool:
        """True when the value may be quoted as science (not an assumption)."""
        return self.is_known and self.status.is_scientific

    @property
    def is_assumed(self) -> bool:
        return self.status is Status.ASSUMED_FOR_VISUALIZATION

    # -- conversion ------------------------------------------------------
    @property
    def quantity(self) -> u.Quantity | None:
        """The value as an astropy :class:`~astropy.units.Quantity`."""
        if self.value is None:
            return None
        return self.value * self.unit

    def to(self, unit: u.UnitBase) -> "Parameter":
        """Return a copy converted to ``unit``, scaling the uncertainties."""
        if self.value is None:
            return replace(self, unit=u.Unit(unit))
        factor = (1.0 * self.unit).to_value(unit)
        return replace(
            self,
            value=self.value * factor,
            error_plus=None if self.error_plus is None else self.error_plus * factor,
            error_minus=None if self.error_minus is None else self.error_minus * factor,
            unit=u.Unit(unit),
        )

    def value_in(self, unit: u.UnitBase, default: float | None = None) -> float | None:
        """Magnitude in ``unit``, or ``default`` when unknown."""
        if self.value is None:
            return default
        return float((self.value * self.unit).to_value(unit))

    def require(self, unit: u.UnitBase) -> float:
        """Magnitude in ``unit``; raise when the parameter is unknown.

        Use this at call sites where silently inventing a number would
        corrupt the science.
        """
        if self.value is None:
            raise ValueError(
                "parameter is {0}; refusing to substitute a value ({1})".format(
                    self.status.value, self.provenance or "no provenance"
                )
            )
        return float((self.value * self.unit).to_value(unit))

    # -- presentation ----------------------------------------------------
    @property
    def status_label(self) -> str:
        return _LABELS[self.status]

    def format(self, digits: int = 4, *, with_status: bool = True) -> str:
        """Render for the UI, never hiding an assumption."""
        if self.value is None:
            return "unknown"

        err_digits = max(digits - 1, 1)
        text = "{0:.{1}g}".format(self.value, digits)
        plus, minus = self.error_plus, self.error_minus
        if plus is not None or minus is not None:
            if plus is not None and minus is not None and math.isclose(plus, minus, rel_tol=1e-9):
                text += " +/- {0:.{1}g}".format(plus, err_digits)
            else:
                plus_text = "?" if plus is None else "{0:.{1}g}".format(plus, err_digits)
                minus_text = "?" if minus is None else "{0:.{1}g}".format(minus, err_digits)
                text += " (+{0} / -{1})".format(plus_text, minus_text)

        unit_text = self.unit.to_string()
        if unit_text and self.unit is not u.dimensionless_unscaled:
            text += " " + unit_text
        if with_status and self.status is not Status.MEASURED:
            text += "  [{0}]".format(self.status_label)
        return text

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.format()

    def as_dict(self) -> dict[str, Any]:
        """Flat record for persistence (roadmap section 12)."""
        return {
            "value": self.value,
            "unit": self.unit.to_string(),
            "error_plus": self.error_plus,
            "error_minus": self.error_minus,
            "status": self.status.value,
            "provenance": self.provenance,
            "reference": self.reference,
            "retrieved": None if self.retrieved is None else self.retrieved.isoformat(),
            "note": self.note,
        }


def combined_status(*parameters: "Parameter") -> Status:
    """The status a value computed from ``parameters`` is entitled to.

    Deliberately pessimistic, in this order: anything unknown makes the
    result unknown; anything assumed for visualisation makes the result
    assumed, however solid the arithmetic; otherwise the result is derived,
    because it was computed rather than observed.

    The middle rule is the one with teeth. ``a(1-e)`` is an exact identity,
    so it is tempting to report a derived periapsis from an assumed
    semimajor axis - and that would turn a number invented so a picture
    could be drawn into a quotable orbital distance. The same applies to a
    star's place on an HR diagram: an assumed luminosity plots at a
    perfectly definite height.

    Lives here, beside :class:`Status`, because two copies of this rule
    would eventually disagree about the same pair of inputs.
    """
    if any(not p.is_known for p in parameters):
        return Status.UNKNOWN
    if any(p.status is Status.ASSUMED_FOR_VISUALIZATION for p in parameters):
        return Status.ASSUMED_FOR_VISUALIZATION
    return Status.DERIVED


def _today() -> date:
    return datetime.now(timezone.utc).date()


def unknown(
    unit: u.UnitBase = u.dimensionless_unscaled,
    *,
    provenance: str = "",
    note: str = "",
) -> Parameter:
    """A parameter that is explicitly not available."""
    return Parameter(None, unit, status=Status.UNKNOWN, provenance=provenance, note=note)


def measured(
    value: float | None,
    unit: u.UnitBase = u.dimensionless_unscaled,
    *,
    error_plus: float | None = None,
    error_minus: float | None = None,
    provenance: str = "",
    reference: str | None = None,
    retrieved: date | None = None,
) -> Parameter:
    """A published value.  A missing ``value`` degrades to :func:`unknown`."""
    return Parameter(
        value,
        unit,
        error_plus=error_plus,
        error_minus=error_minus,
        status=Status.MEASURED,
        provenance=provenance,
        reference=reference,
        retrieved=retrieved or _today(),
    )


def derived(
    value: float | None,
    unit: u.UnitBase = u.dimensionless_unscaled,
    *,
    error_plus: float | None = None,
    error_minus: float | None = None,
    provenance: str = "",
    note: str = "",
) -> Parameter:
    """A value computed from other values by a documented relation."""
    return Parameter(
        value,
        unit,
        error_plus=error_plus,
        error_minus=error_minus,
        status=Status.DERIVED,
        provenance=provenance,
        note=note,
        retrieved=_today(),
    )


def assumed(
    value: float,
    unit: u.UnitBase = u.dimensionless_unscaled,
    *,
    provenance: str = "",
    note: str = "",
) -> Parameter:
    """A rendering placeholder.  Never quotable as science."""
    return Parameter(
        value,
        unit,
        status=Status.ASSUMED_FOR_VISUALIZATION,
        provenance=provenance or "visualization-default",
        note=note,
        retrieved=_today(),
    )
