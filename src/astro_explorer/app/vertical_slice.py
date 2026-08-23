"""The one-star-one-planet scientific vertical slice.

This module wires the whole chain together for a single host system::

    local snapshot
        -> PlanetRecord / StarRecord      (units + provenance)
        -> OrbitalElements                (measured / derived / assumed)
        -> M(t) -> Kepler -> E            (physics/kepler.py)
        -> r_perifocal -> Rz Rx Rz -> r   (physics/orientation.py)
        -> SystemFrame position in AU     (coordinates/system_frame.py)
        -> SceneDescription               (rendering/scene_builder.py)
        -> OpenGL                         (rendering/gl_backend.py)

The reference system is **HD 80606 b**: at ``e = 0.93183`` it is one of the
most eccentric known planets, which makes it an unusually sharp probe. A
first-order Kepler solver, a mistaken rotation order, or a propagator that
advances true anomaly instead of mean anomaly all produce visibly and
measurably wrong results at that eccentricity, where they would pass
unnoticed on a near-circular orbit.

A second, ordinary system (WASP-39 b or K2-18 b) is carried alongside as a
sanity case, and because both have *incomplete* orientation - which exercises
the unknown-versus-zero policy the slice is meant to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd

from ..coordinates.inspector import (
    NoteRow,
    planet_distance_rows,
    star_coordinate_rows,
)
from ..coordinates.system_frame import SystemFrame
from ..data.nasa_archive import SolutionPolicy
from ..data.schema import PlanetRecord, StarRecord, build_planet_record
from ..physics.ephemeris import JD_UNIX_EPOCH
from ..physics.orbital_elements import PhaseKnowledge
from ..physics.phase import PhaseSolution, PhaseStatus
from ..physics.state_vectors import (
    StateVector,
    expected_specific_energy,
    gravitational_parameter,
    specific_orbital_energy,
    state_at_mean_anomaly,
)
from ..provenance import Status

__all__ = [
    "REFERENCE_SNAPSHOT",
    "PRIMARY_TARGET",
    "SANITY_TARGETS",
    "SystemSlice",
    "load_reference_catalog",
    "build_slice",
]

#: The validated local snapshot the slice reads from. Fetched once from the
#: NASA archive using the documented default-solution query and committed, so
#: the slice is reproducible with the network disabled.
REFERENCE_SNAPSHOT = Path("data/reference_systems/vertical_slice_ps_default.csv")

#: The eccentricity stress test.
PRIMARY_TARGET = "HD 80606 b"

#: Ordinary systems carried as sanity cases.
SANITY_TARGETS = ("WASP-39 b", "K2-18 b")


def load_reference_catalog(path=None, resources=None) -> pd.DataFrame:
    """Read the committed reference snapshot.

    Looks the file up through the resource manager when one is supplied, so
    a frozen build finds it the same way it finds the spectra.
    """
    if path is None:
        candidates = []
        if resources is not None:
            candidates.extend(Path(root) / REFERENCE_SNAPSHOT for root in resources.roots())
        candidates.append(Path(__file__).resolve().parents[3] / REFERENCE_SNAPSHOT)
        candidates.append(Path.cwd() / REFERENCE_SNAPSHOT)
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "reference snapshot not found; looked in:\n  "
                + "\n  ".join(str(c) for c in candidates)
            )

    frame = pd.read_csv(path)
    frame.attrs["solution_policy"] = SolutionPolicy.DEFAULT_SOLUTION.value
    frame.attrs["source_table"] = "ps"
    frame.attrs["offline_snapshot"] = True
    frame.attrs["path"] = str(path)
    return frame


@dataclass
class SystemSlice:
    """One host system, propagated and ready to render.

    Holds the frame, the records and the gravitational parameter. Everything
    it returns is either a provenance-carrying scientific object or a
    render primitive - never a bare number whose meaning depends on context.
    """

    frame: SystemFrame
    star: StarRecord
    planets: list[PlanetRecord]
    policy: SolutionPolicy = SolutionPolicy.DEFAULT_SOLUTION
    source: str = ""
    _mu: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._mu is None:  # derived once; the star mass may legitimately be unknown
            planet_mass = 0.0
            if self.planets:
                planet_mass = self.planets[0].mass_earth.value_in(u.M_sun) or 0.0
            self._mu = gravitational_parameter(self.star.mass, planet_mass)

    # -- identification ---------------------------------------------------
    @property
    def mu(self) -> float | None:
        """``G(M* + Mp)`` in AU^3/day^2, or None if the stellar mass is unknown."""
        return self._mu

    def planet(self, name: str) -> PlanetRecord | None:
        return next((p for p in self.planets if p.name == name), None)

    # -- propagation ------------------------------------------------------
    def phase(self, record: PlanetRecord, time_jd: float) -> PhaseSolution:
        """Full phase provenance for a planet (review section 9).

        Assumed phases are permitted here because the system view draws
        them; what matters is that the returned solution says so.
        """
        return record.elements.phase_at(time_jd, allow_assumed=True)

    def mean_anomaly(self, record: PlanetRecord, time_jd: float) -> tuple[float | None, bool]:
        """``(M, phase_is_assumed)`` - the compact form of :meth:`phase`."""
        solution = self.phase(record, time_jd)
        return solution.mean_anomaly, solution.is_assumed

    def placements(self, time_jd: float) -> dict[str, tuple[float, bool]]:
        """``name -> (mean anomaly, phase_is_assumed)`` for every placeable planet.

        The flag matters. A planet with a published epoch is where the
        ephemeris says it is; one without is being advanced at the right
        *rate* from an arbitrary zero, which is a picture of the motion and
        not a claim about tonight's sky.
        """
        found: dict[str, tuple[float, bool]] = {}
        for record in self.planets:
            solution = self.phase(record, time_jd)
            if solution.is_placeable:
                found[record.name] = (solution.mean_anomaly, solution.is_assumed)
        return found

    def phase_solutions(self, time_jd: float) -> dict[str, PhaseSolution]:
        """Full phase provenance per planet."""
        return {record.name: self.phase(record, time_jd) for record in self.planets}

    def mean_anomalies(self, time_jd: float, *, include_assumed: bool = True) -> dict[str, float]:
        """Mean anomalies for every planet whose phase can be drawn.

        ``include_assumed=False`` keeps every planet whose *timing* was
        observed - fully constrained or conjunction-normalised - and drops
        only those advanced from an arbitrary zero. A transit epoch is a
        real observation even when the orientation it is read through is
        normalised.
        """
        solutions = self.phase_solutions(time_jd)
        return {
            name: solution.mean_anomaly
            for name, solution in solutions.items()
            if solution.is_placeable
            and (include_assumed or solution.status.is_observationally_anchored)
        }

    def state(self, record: PlanetRecord, time_jd: float) -> StateVector | None:
        """Full inertial state in the system frame, in AU and AU/day.

        This is the scientific output of the slice. It goes through
        ``M -> E -> r_perifocal -> R_z(Omega) R_x(i) R_z(omega)``, using the
        display-normalised elements so an unknown node is a documented zero
        rather than an omission.
        """
        solution = self.phase(record, time_jd)
        anomaly = solution.mean_anomaly
        if anomaly is None or not record.elements.semimajor_axis.is_known:
            return None

        display = record.elements.for_display()
        return state_at_mean_anomaly(
            display.semimajor_axis.value_in(u.au),
            display.eccentricity.value_in(u.dimensionless_unscaled, 0.0),
            anomaly,
            inclination=display.inclination.value_in(u.rad, 0.0),
            argument_of_periapsis=display.argument_of_periastron.value_in(u.rad, 0.0),
            longitude_of_ascending_node=display.longitude_of_ascending_node.value_in(u.rad, 0.0),
            mu=self.mu,
        )

    def framed_position(self, record: PlanetRecord, time_jd: float):
        """The planet's position as a :class:`FramedPosition` in AU."""
        state = self.state(record, time_jd)
        if state is None:
            return None
        return self.frame.place_planet(state.position)

    # -- coordinates and distances (Explorer C3) --------------------------
    def inspect_star(self) -> list:
        """Coordinate and distance rows for the host star.

        Empty when the catalogue gave the host no sky position at all. It is
        not a row list of zeros: a star with no position is not a star at
        the origin.
        """
        return star_coordinate_rows(self.star.position)

    def inspect_planet(self, record: PlanetRecord, time_jd: float) -> list:
        """Coordinate and distance rows for one planet at ``time_jd``.

        The position handed to the inspector is ``state.position`` - the
        propagator's float64 vector in AU - and never anything that has been
        through the scene builder. That is the whole C3 contract, and it is
        enforced here at the one point where the two layers meet: this
        method has a render-ready position available in
        :meth:`framed_position` and deliberately does not use it.
        """
        state = self.state(record, time_jd)
        rows = planet_distance_rows(
            record.elements,
            None if state is None else state.position,
            host=self.star.position,
        )
        # Every instantaneous row above is only as meaningful as the phase it
        # was evaluated at, so the qualifier travels with them rather than
        # being left for the caller to remember.
        solution = self.phase(record, time_jd)
        rows.append(
            NoteRow(
                "Phase provenance",
                solution.status.label,
                status=Status.MEASURED
                if solution.status is PhaseStatus.CONSTRAINED
                else Status.ASSUMED_FOR_VISUALIZATION
                if solution.status is PhaseStatus.ASSUMED
                else Status.DERIVED
                if solution.status is PhaseStatus.PARTIALLY_CONSTRAINED
                else Status.UNKNOWN,
            )
        )
        return rows

    def describe_coordinates(self, record: PlanetRecord, time_jd: float) -> list[str]:
        """The C3 inspector as text, host rows then planet rows."""
        lines = ["COORDINATES AND DISTANCES", "  Host: {0}".format(self.star.name)]
        lines += ["    " + row.format() for row in self.inspect_star()]
        if not self.star.position or not self.star.position.has_distance:
            lines.append(
                "    Absolute position unavailable: this system is drawn in "
                "its own frame and claims no distance from Earth."
            )
        lines += ["", "  Planet: {0}".format(record.name)]
        lines += ["    " + row.format() for row in self.inspect_planet(record, time_jd)]
        return lines

    # -- diagnostics ------------------------------------------------------
    def energy_check(self, record: PlanetRecord, time_jd: float) -> dict | None:
        """Compare the propagated energy against ``-mu / 2a``.

        Surfaced rather than hidden: if this ever drifts, the propagator is
        wrong and the whole slice is untrustworthy.
        """
        state = self.state(record, time_jd)
        if state is None or not state.has_velocity or self.mu is None:
            return None
        axis = record.elements.semimajor_axis.value_in(u.au)
        measured = float(specific_orbital_energy(state, self.mu))
        expected = expected_specific_energy(axis, self.mu)
        return {
            "measured": measured,
            "expected": expected,
            "relative_error": abs(measured / expected - 1.0),
        }

    def describe_orbit(self, record: PlanetRecord, time_jd: float) -> list[str]:
        """Provenance-first report of the orbit and the propagated state."""
        elements = record.elements
        display = elements.for_display()

        lines = [
            "Planet:              {0}".format(record.name),
            "Host star:           {0}".format(self.star.name),
            "Frame:               {0}, origin = host star, CPU float64".format(
                self.frame.describe()
            ),
            "",
            "ORBITAL ELEMENTS",
            "  Semi-major axis a: {0}".format(elements.semimajor_axis.format()),
            "  Eccentricity e:    {0}".format(elements.eccentricity.format()),
            "  Period P:          {0}".format(elements.period.format()),
        ]

        # The unknown-versus-zero distinction, spelled out.
        for label, published, shown in (
            ("Inclination i", elements.inclination, display.inclination),
            (
                "Asc. node O",
                elements.longitude_of_ascending_node,
                display.longitude_of_ascending_node,
            ),
        ):
            if published.status is Status.MEASURED:
                lines.append("  {0:<19}{1}".format(label + ":", published.to(u.deg).format()))
            else:
                lines.append("  {0:<19}UNKNOWN".format(label + ":"))
                lines.append(
                    "  {0:<19}display normalisation {1} [{2}]".format(
                        "", shown.to(u.deg).format(with_status=False), shown.status_label
                    )
                )

        # The argument of periastron needs its convention stated alongside
        # it, because the raw number is meaningless without one.
        raw_omega = elements.argument_of_periastron
        resolved = elements.argument_of_periapsis_planet
        if raw_omega.is_known:
            lines.append("  {0:<19}{1}  (raw, as catalogued)".format(
                "Arg. periapsis w:", raw_omega.to(u.deg).format()
            ))
            lines.append("  {0:<19}{1}".format(
                "  convention:", elements.periastron_convention.label
            ))
            lines.append("  {0:<19}{1}  [{2}]".format(
                "  used for planet:", resolved.to(u.deg).format(with_status=False),
                resolved.status_label,
            ))
            if elements.periastron_convention_is_assumed:
                lines.append("  {0:<19}{1}".format(
                    "  caveat:", elements.periastron_convention.caveat
                ))
        else:
            lines.append("  {0:<19}UNKNOWN".format("Arg. periapsis w:"))
            lines.append("  {0:<19}display normalisation {1} [{2}]".format(
                "", display.argument_of_periastron.to(u.deg).format(with_status=False),
                display.argument_of_periastron.status_label,
            ))

        lines.append("")
        lines.append("ORBIT VALIDITY")
        lines.extend("  " + line for line in elements.validity.describe())
        for epoch in elements.epochs:
            lines.append("  " + epoch.describe())
        anchor = elements.mean_anomaly_anchor
        if anchor.is_known:
            # Listed even when undated: the angle is a real measurement, and
            # saying nothing would look like it was never published.
            lines.append("  " + anchor.describe())
        if not elements.epochs and not anchor.is_known:
            lines.append("  no epoch published; the orbital phase is not constrained")
        lines.append("  Reference:         {0}".format(_clean_reference(elements.reference)))

        lines += [
            "",
            "PROPAGATED STATE at JD {0:.4f}".format(time_jd),
        ]

        state = self.state(record, time_jd)
        if state is None:
            lines.append("  not computable from the published elements")
            return lines

        position = state.position
        lines.append(
            "  r (AU):            [{0:+.8f}, {1:+.8f}, {2:+.8f}]".format(*position)
        )
        lines.append("  |r|:               {0:.8f} AU".format(float(state.radius)))
        if state.has_velocity:
            velocity = state.velocity
            speed_kms = float((float(state.speed) * u.au / u.day).to_value(u.km / u.s))
            lines.append(
                "  v (AU/day):        [{0:+.8f}, {1:+.8f}, {2:+.8f}]".format(*velocity)
            )
            lines.append("  |v|:               {0:.6f} km/s".format(speed_kms))
        else:
            lines.append("  v: not computable - stellar mass unpublished")

        lines += [
            "  Periapsis:         {0}".format(elements.periapsis.format()),
            "  Apoapsis:          {0}".format(elements.apoapsis.format()),
        ]

        check = self.energy_check(record, time_jd)
        if check is not None:
            lines.append(
                "  Energy check:      eps = {0:.9e}, -mu/2a = {1:.9e}, "
                "rel. error {2:.2e}".format(
                    check["measured"], check["expected"], check["relative_error"]
                )
            )

        residual = record.kepler_residual
        if residual is not None:
            lines.append("  Kepler III residual: {0:+.4%}".format(residual))

        return lines

    def describe_system(self, time_jd: float) -> list[str]:
        """System information panel (review section 15).

        One row per planet, ordered outwards, stating what is actually
        known about each orbit rather than only its numbers.
        """
        lines = [
            "SYSTEM: {0}".format(self.frame.host_name),
            "  {0}".format(" | ".join(self.star.describe()[1:4]).replace("  ", " ")),
            "",
            "  {0:<14} {1:>10} {2:>10} {3:>7} {4:>9}  {5}".format(
                "planet", "a (AU)", "P (d)", "e", "r (AU)", "validity"
            ),
        ]

        for record in self.planets:
            elements = record.elements
            axis = elements.semimajor_axis.value_in(u.au)
            period = elements.period.value_in(u.day)
            ecc = elements.eccentricity.value

            state = self.state(record, time_jd)
            radius = "{0:9.5f}".format(float(state.radius)) if state is not None else "        -"

            lines.append(
                "  {0:<14} {1:>10} {2:>10} {3:>7} {4}  {5}".format(
                    record.name,
                    "{0:.5f}".format(axis) if axis is not None else "-",
                    "{0:.4f}".format(period) if period is not None else "-",
                    "{0:.4f}".format(ecc) if ecc is not None else "unk",
                    radius,
                    ",".join(
                        name.replace("_VALID", "").replace("ORIENTATION_", "ORIENT:")
                        for name in elements.validity.names
                    )
                    or "NONE",
                )
            )

        solutions = self.phase_solutions(time_jd)
        tally = {status: 0 for status in PhaseStatus}
        for solution in solutions.values():
            tally[solution.status] += 1

        lines += ["", "  {0} planet(s) by phase provenance:".format(len(self.planets))]
        for status in (
            PhaseStatus.CONSTRAINED,
            PhaseStatus.PARTIALLY_CONSTRAINED,
            PhaseStatus.ASSUMED,
            PhaseStatus.UNKNOWN,
        ):
            if tally[status]:
                lines.append(
                    "    {0:>2}  {1:<22} {2}".format(
                        tally[status], status.value, status.label
                    )
                )

        # Name the specific mapping wherever it is not the trivial one, so
        # "partially constrained" is never left as a bare adjective.
        mappings = sorted(
            {
                solution.provenance
                for solution in solutions.values()
                if solution.status is PhaseStatus.PARTIALLY_CONSTRAINED
            },
            key=lambda item: item.value,
        )
        for provenance in mappings:
            lines.append("    via {0}: {1}".format(provenance.value, provenance.label))
        unknown_nodes = sum(
            1 for r in self.planets if not r.elements.longitude_of_ascending_node.is_known
        )
        if unknown_nodes:
            lines.append(
                "  Ascending node unknown for {0}/{1} planets; all normalised to 0 deg "
                "for display, so relative node alignment is not measured.".format(
                    unknown_nodes, len(self.planets)
                )
            )
        return lines

    def describe_provenance(self) -> list[str]:
        """Where every number came from."""
        return [
            "Source:            {0}".format(self.source or "unknown"),
            "Solution policy:   {0}".format(self.policy.label),
            "                   {0}".format(self.policy.caveat),
            "Host reference:    {0}".format(_clean_reference(self.star.reference)),
            "Planet reference:  {0}".format(
                _clean_reference(self.planets[0].reference) if self.planets else "n/a"
            ),
        ]


def _clean_reference(reference) -> str:
    """Strip the archive's HTML wrapper off a reference string."""
    if not reference:
        return "unknown"
    import re

    text = re.sub(r"<[^>]*>", "", str(reference)).strip()
    return text or "unknown"


def build_slice(
    host_name: str,
    catalog: pd.DataFrame | None = None,
    *,
    policy: SolutionPolicy = SolutionPolicy.DEFAULT_SOLUTION,
    resources=None,
) -> SystemSlice:
    """Build a :class:`SystemSlice` for one host from a local snapshot."""
    if catalog is None:
        catalog = load_reference_catalog(resources=resources)

    rows = catalog[catalog["hostname"] == host_name]
    if rows.empty:
        raise KeyError(
            "host {0!r} not in the snapshot; available: {1}".format(
                host_name, sorted(catalog["hostname"].unique())
            )
        )

    records = [build_planet_record(row.to_dict(), policy=policy) for _, row in rows.iterrows()]
    records.sort(
        key=lambda r: (
            r.semimajor_axis.value_in(u.au) is None,
            r.semimajor_axis.value_in(u.au) or 0.0,
        )
    )

    star = records[0].host
    # The system can be rendered without knowing where it sits in the galaxy;
    # requiring a distance would exclude every host with an unusable parallax.
    host_pc = None
    if star.position is not None and star.position.has_distance:
        host_pc = star.position.cartesian_pc()

    return SystemSlice(
        frame=SystemFrame.for_host(host_name, host_pc),
        star=star,
        planets=records,
        policy=policy,
        source=str(catalog.attrs.get("path", "in-memory catalogue")),
    )
