"""Unit bridges and distance formatting (roadmap section 10).

Everything here goes through :mod:`astropy.units`; no conversion factor is
written out by hand.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from ..provenance import Parameter, derived, unknown

__all__ = [
    "pc_to_ly",
    "ly_to_pc",
    "au_to_pc",
    "pc_to_au",
    "au_to_km",
    "km_to_au",
    "describe_distance",
]


def pc_to_ly(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.pc).to_value(u.lyr))


def ly_to_pc(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.lyr).to_value(u.pc))


def au_to_pc(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.au).to_value(u.pc))


def pc_to_au(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.pc).to_value(u.au))


def au_to_km(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.au).to_value(u.km))


def km_to_au(value):
    return np.asarray(value, dtype=np.float64) * float((1.0 * u.km).to_value(u.au))


def _significant_digits(param: Parameter, cap: int = 6) -> int:
    """Digits justified by the quoted uncertainty (roadmap section 10.1).

    Displaying a catalogue distance of 12.34 +/- 0.5 pc to nine decimals is
    fake precision, so the number of digits is bounded by the error bar.
    """
    if param.value is None:
        return cap
    error = param.error_plus or param.error_minus
    if not error or error <= 0:
        return cap
    magnitude = np.log10(abs(param.value)) if param.value else 0.0
    digits = int(np.floor(magnitude - np.log10(error))) + 2
    return max(2, min(cap, digits))


def describe_distance(distance: Parameter) -> list[str]:
    """Render a distance in pc, ly, km and light-travel time.

    Roadmap section 10.1: show several units, but never invent precision the
    catalogue does not support.
    """
    if not distance.is_known:
        return ["Distance: unknown"]

    digits = _significant_digits(distance)
    in_pc = distance.to(u.pc)
    in_ly = distance.to(u.lyr)
    in_km = distance.to(u.km)

    lines = [
        "Distance: {0}".format(in_pc.format(digits)),
        "          {0}".format(in_ly.format(digits)),
        "          {0}".format(in_km.format(min(digits, 4))),
        "Light travel time: {0}".format(
            derived(in_ly.value, u.yr, provenance="light_travel_time").format(digits)
        ),
    ]
    if distance.status.value != "MEASURED":
        lines.append("          [{0}] {1}".format(distance.status_label, distance.note).rstrip())
    return lines


def orbital_distance_summary(elements) -> list[str]:
    """Periapsis / apoapsis / current distance lines (roadmap section 10.2)."""
    lines = []
    peri = elements.periapsis
    apo = elements.apoapsis
    lines.append("Periapsis: {0}".format(peri.format() if peri.is_known else "unknown"))
    lines.append("Apoapsis:  {0}".format(apo.format() if apo.is_known else "unknown"))
    return lines


__all__ += ["orbital_distance_summary"]
