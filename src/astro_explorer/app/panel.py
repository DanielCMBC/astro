"""The information panel model: read-only, provenance-carrying rows.

Review section 13. A panel is a *model*, not a widget - it holds text and
status, and any front end (Tkinter, Qt, a GL overlay, a test) renders it.
That keeps the rule that display concerns never reach back into the
science.

Two invariants the whole file exists to hold:

**UNKNOWN never becomes a numeric placeholder.** A row with no value has
``value is None`` and shows the word "unknown". It does not show ``0``,
``-``, ``N/A`` dressed as a number, or a dash that a reader might mistake
for a measurement.

**Every scientific row exposes its provenance.** Where the number came
from, whether it was measured, derived or assumed, and which publication -
carried on the row itself rather than in a footnote that can drift away
from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import astropy.units as u

from ..data.identity import EntityId
from ..physics.orbital_semantics import OrbitValidity
from ..physics.phase import PhaseSolution, PhaseStatus
from ..provenance import Parameter, Status
from ..text import display_text

__all__ = [
    "Emphasis",
    "ParameterRow",
    "PanelSection",
    "InfoPanel",
    "build_star_panel",
    "build_planet_panel",
    "build_system_panel",
]


class Emphasis(str, Enum):
    """How a row should be distinguished, without naming a colour.

    The model says what a row *means*; a front end decides how that looks.
    """

    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    """A display assumption. Must be visually distinct from a measurement."""

    UNKNOWN = "UNKNOWN"
    PLAIN = "PLAIN"
    """Non-scientific text: a name, a discovery facility, a heading."""

    @classmethod
    def for_status(cls, status: Status) -> "Emphasis":
        return {
            Status.MEASURED: cls.MEASURED,
            Status.DERIVED: cls.DERIVED,
            Status.ASSUMED_FOR_VISUALIZATION: cls.ASSUMED,
            Status.UNKNOWN: cls.UNKNOWN,
        }[status]

    @property
    def is_scientific_claim(self) -> bool:
        """True only where the number may be quoted as science."""
        return self in (Emphasis.MEASURED, Emphasis.DERIVED)


@dataclass(frozen=True)
class ParameterRow:
    """One labelled value, with everything needed to judge it."""

    label: str
    value_text: str = "unknown"
    unit: str = ""
    uncertainty_text: str = ""
    emphasis: Emphasis = Emphasis.PLAIN
    source: str = ""
    reference: str = ""
    note: str = ""
    #: The raw magnitude, or None. A UI must never render this directly;
    #: it exists so a plot or an export can use the number behind the text.
    value: float | None = None

    @property
    def is_known(self) -> bool:
        return self.value is not None

    @property
    def is_assumption(self) -> bool:
        return self.emphasis is Emphasis.ASSUMED

    @property
    def has_provenance(self) -> bool:
        """Every scientific row must be able to say where it came from."""
        if not self.emphasis.is_scientific_claim and not self.is_assumption:
            return True
        return bool(self.source or self.reference or self.note)

    def rendered(self) -> str:
        """One line, for a text front end."""
        parts = [self.value_text]
        if self.uncertainty_text:
            parts.append(self.uncertainty_text)
        if self.unit:
            parts.append(self.unit)
        text = " ".join(parts)
        if self.emphasis in (Emphasis.DERIVED, Emphasis.ASSUMED):
            text += "  [{0}]".format(self.emphasis.value.lower())
        return "{0:<22} {1}".format(self.label + ":", text)


def _text_row(label: str, value, note: str = "") -> ParameterRow:
    """A non-scientific row: a name, a method, a facility."""
    cleaned = display_text(value)
    return ParameterRow(
        label=label,
        value_text=cleaned,
        emphasis=Emphasis.PLAIN,
        value=None,
        note=note,
    )


def parameter_row(
    label: str,
    parameter: Parameter,
    *,
    unit: u.UnitBase | None = None,
    digits: int = 4,
) -> ParameterRow:
    """Turn a :class:`Parameter` into a row without losing anything.

    An unknown parameter produces the word "unknown" and a ``None`` value -
    never a zero, never a dash that could pass for a number.
    """
    shown = parameter.to(unit) if unit is not None and parameter.unit != unit else parameter

    if not shown.is_known:
        return ParameterRow(
            label=label,
            value_text="unknown",
            unit="",
            emphasis=Emphasis.UNKNOWN,
            source=shown.provenance,
            note=shown.note,
            value=None,
        )

    uncertainty = ""
    if shown.error_plus is not None or shown.error_minus is not None:
        plus, minus = shown.error_plus, shown.error_minus
        if plus is not None and minus is not None and abs(plus - minus) < 1e-12 * max(abs(plus), 1.0):
            uncertainty = "+/- {0:.{1}g}".format(plus, max(digits - 1, 1))
        else:
            uncertainty = "(+{0} / -{1})".format(
                "?" if plus is None else "{0:.{1}g}".format(plus, max(digits - 1, 1)),
                "?" if minus is None else "{0:.{1}g}".format(minus, max(digits - 1, 1)),
            )

    unit_text = shown.unit.to_string()
    if shown.unit is u.dimensionless_unscaled:
        unit_text = ""

    return ParameterRow(
        label=label,
        value_text="{0:.{1}g}".format(shown.value, digits),
        unit=unit_text,
        uncertainty_text=uncertainty,
        emphasis=Emphasis.for_status(shown.status),
        source=shown.provenance,
        reference=shown.reference or "",
        note=shown.note,
        value=float(shown.value),
    )


@dataclass(frozen=True)
class PanelSection:
    """A titled group of rows."""

    heading: str
    rows: list[ParameterRow] = field(default_factory=list)
    caveat: str = ""

    @property
    def assumptions(self) -> list[ParameterRow]:
        return [row for row in self.rows if row.is_assumption]


@dataclass(frozen=True)
class InfoPanel:
    """Everything a front end needs to display one selected entity."""

    entity_id: str
    title: str
    subtitle: str = ""
    sections: list[PanelSection] = field(default_factory=list)
    #: Bumped by the caller whenever the selection changes, so a late async
    #: result can be discarded rather than overwriting a newer panel.
    generation: int = 0

    @property
    def rows(self) -> list[ParameterRow]:
        return [row for section in self.sections for row in section.rows]

    @property
    def assumptions(self) -> list[ParameterRow]:
        """Display assumptions, kept separate from measurements."""
        return [row for row in self.rows if row.is_assumption]

    @property
    def unknowns(self) -> list[ParameterRow]:
        return [row for row in self.rows if row.emphasis is Emphasis.UNKNOWN]

    def render(self) -> list[str]:
        lines = [self.title]
        if self.subtitle:
            lines.append(self.subtitle)
        for section in self.sections:
            lines.append("")
            lines.append(section.heading.upper())
            lines.extend("  " + row.rendered() for row in section.rows)
            if section.caveat:
                lines.append("  " + section.caveat)
        if self.assumptions:
            lines.append("")
            lines.append("DISPLAY ASSUMPTIONS (not measurements)")
            for row in self.assumptions:
                lines.append("  {0}: {1}".format(row.label, row.note or "assumed for display"))
        return lines


# ==========================================================================
# Builders
# ==========================================================================


def build_star_panel(star, *, generation: int = 0) -> InfoPanel:
    """The panel for a selected host star."""
    identity = star.entity_id
    sections = [
        PanelSection(
            "Star",
            [
                _text_row("Name", star.name),
                _text_row("Spectral type", star.spectral_type),
                parameter_row("Effective temp.", star.effective_temperature, unit=u.K),
                parameter_row("Radius", star.radius, unit=u.R_sun),
                parameter_row("Mass", star.mass, unit=u.M_sun),
                parameter_row("Luminosity", star.luminosity, unit=u.L_sun),
                parameter_row("Metallicity", star.metallicity),
                parameter_row("Age", star.age, unit=u.Gyr),
            ],
        )
    ]

    if star.position is not None:
        distance = star.position.distance
        rows = [
            parameter_row("Distance", distance, unit=u.pc),
            parameter_row("Right ascension", star.position.ra, unit=u.deg),
            parameter_row("Declination", star.position.dec, unit=u.deg),
        ]
        if distance.is_known:
            rows.insert(1, parameter_row("Distance", distance.to(u.lyr), unit=u.lyr))
        sections.append(
            PanelSection(
                "Position",
                rows,
                caveat=(
                    ""
                    if distance.is_known
                    else "No usable parallax or catalogue distance: this system "
                    "has no position in the neighbourhood view."
                ),
            )
        )

    zone = star.habitable_zone
    sections.append(
        PanelSection(
            "Habitable zone",
            [
                parameter_row("Inner edge", zone.inner, unit=u.au),
                parameter_row("Outer edge", zone.outer, unit=u.au),
            ],
            caveat=zone.model,
        )
    )

    return InfoPanel(
        entity_id=str(identity) if identity else star.name,
        title=star.name or "unknown star",
        subtitle=display_text(star.spectral_type, ""),
        sections=sections,
        generation=generation,
    )


def build_planet_panel(
    record, phase: PhaseSolution | None = None, *, generation: int = 0
) -> InfoPanel:
    """The panel for a selected planet.

    Read-only: nothing here touches the record, so selecting a planet cannot
    perturb the orbit it is describing.
    """
    elements = record.elements
    identity = record.entity_id

    sections = [
        PanelSection(
            "Planet",
            [
                _text_row("Name", record.name),
                _text_row("Host", record.host.name),
                _text_row("Discovery method", record.discovery_method),
                _text_row(
                    "Discovery year",
                    int(round(record.discovery_year.value))
                    if record.discovery_year.is_known
                    else None,
                ),
                _text_row("Facility", record.discovery_facility),
            ],
        ),
        PanelSection(
            "Physical",
            [
                parameter_row("Radius", record.radius_earth, unit=u.R_earth),
                parameter_row("Mass", record.mass_earth, unit=u.M_earth),
                parameter_row("Bulk density", record.bulk_density),
                parameter_row("Surface gravity", record.surface_gravity),
                parameter_row("Equilibrium temp.", record.equilibrium_temperature, unit=u.K),
                parameter_row("Insolation", record.insolation),
            ],
        ),
        PanelSection(
            "Orbit",
            [
                parameter_row("Semi-major axis", elements.semimajor_axis, unit=u.au),
                parameter_row("Eccentricity", elements.eccentricity),
                parameter_row("Period", elements.period, unit=u.day),
                parameter_row("Periapsis", elements.periapsis, unit=u.au),
                parameter_row("Apoapsis", elements.apoapsis, unit=u.au),
                parameter_row("Inclination", elements.inclination, unit=u.deg),
                parameter_row(
                    "Arg. periapsis", elements.argument_of_periastron, unit=u.deg
                ),
                parameter_row(
                    "Ascending node", elements.longitude_of_ascending_node, unit=u.deg
                ),
            ],
            caveat=(
                "Argument of periastron convention: {0}".format(
                    elements.periastron_convention.label
                )
            ),
        ),
    ]

    validity_rows = [
        ParameterRow(
            label="Orbit validity",
            value_text=", ".join(elements.validity.names) or "NONE",
            emphasis=Emphasis.PLAIN,
            note="; ".join(elements.validity.describe()),
        )
    ]
    if phase is not None:
        validity_rows.append(
            ParameterRow(
                label="Phase status",
                value_text=phase.status.value,
                emphasis=(
                    Emphasis.MEASURED
                    if phase.status is PhaseStatus.CONSTRAINED
                    else Emphasis.DERIVED
                    if phase.status is PhaseStatus.PARTIALLY_CONSTRAINED
                    else Emphasis.ASSUMED
                    if phase.status is PhaseStatus.ASSUMED
                    else Emphasis.UNKNOWN
                ),
                source=phase.provenance.value,
                note=phase.status.label,
            )
        )
    sections.append(PanelSection("Provenance", validity_rows))

    return InfoPanel(
        entity_id=str(identity) if identity else record.name,
        title=record.name or "unknown planet",
        subtitle="orbiting {0}".format(record.host.name),
        sections=sections,
        generation=generation,
    )


def build_system_panel(host_name: str, star, records, *, generation: int = 0) -> InfoPanel:
    """A summary of a whole system: one row per planet."""
    rows = []
    for record in records:
        axis = record.semimajor_axis
        rows.append(
            ParameterRow(
                label=record.name,
                value_text=(
                    "{0:.5g}".format(axis.value_in(u.au)) if axis.is_known else "unknown"
                ),
                unit="AU" if axis.is_known else "",
                emphasis=Emphasis.for_status(axis.status),
                source=axis.provenance,
                reference=axis.reference or "",
                value=axis.value_in(u.au),
            )
        )

    identity = star.entity_id if star is not None else None
    return InfoPanel(
        entity_id=str(identity) if identity else host_name,
        title=host_name,
        subtitle="{0} planet(s)".format(len(records)),
        sections=[PanelSection("Planets by semi-major axis", rows)],
        generation=generation,
    )
