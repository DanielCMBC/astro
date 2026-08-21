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

from ..coordinates.system_frame import SystemFrame
from ..data.nasa_archive import SolutionPolicy
from ..data.schema import PlanetRecord, StarRecord, build_planet_record
from ..physics.ephemeris import JD_UNIX_EPOCH
from ..physics.orbital_elements import PhaseKnowledge
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
    def mean_anomaly(self, record: PlanetRecord, time_bjd: float) -> tuple[float | None, bool]:
        """``(M, phase_is_assumed)`` for a planet at a barycentric Julian date.

        Returns ``(None, True)`` when the orbit has no epoch at all, so the
        caller draws the path without placing a body on it.
        """
        anomaly = record.elements.mean_anomaly_at(time_bjd)
        if anomaly is not None:
            return anomaly, not record.elements.can_compute_current_position

        motion = record.elements.mean_motion_rad_per_day
        if motion is None:
            return None, True
        # The rate is physical even when the absolute phase is not.
        return float(np.mod(motion * (time_bjd - JD_UNIX_EPOCH), 2.0 * np.pi)), True

    def mean_anomalies(self, time_bjd: float) -> dict[str, float]:
        """Mean anomalies for every planet whose phase is defined."""
        result: dict[str, float] = {}
        for record in self.planets:
            anomaly, _assumed = self.mean_anomaly(record, time_bjd)
            if anomaly is not None:
                result[record.name] = anomaly
        return result

    def state(self, record: PlanetRecord, time_bjd: float) -> StateVector | None:
        """Full inertial state in the system frame, in AU and AU/day.

        This is the scientific output of the slice. It goes through
        ``M -> E -> r_perifocal -> R_z(Omega) R_x(i) R_z(omega)``, using the
        display-normalised elements so an unknown node is a documented zero
        rather than an omission.
        """
        anomaly, _assumed = self.mean_anomaly(record, time_bjd)
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

    def framed_position(self, record: PlanetRecord, time_bjd: float):
        """The planet's position as a :class:`FramedPosition` in AU."""
        state = self.state(record, time_bjd)
        if state is None:
            return None
        return self.frame.place_planet(state.position)

    # -- diagnostics ------------------------------------------------------
    def energy_check(self, record: PlanetRecord, time_bjd: float) -> dict | None:
        """Compare the propagated energy against ``-mu / 2a``.

        Surfaced rather than hidden: if this ever drifts, the propagator is
        wrong and the whole slice is untrustworthy.
        """
        state = self.state(record, time_bjd)
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

    def describe_orbit(self, record: PlanetRecord, time_bjd: float) -> list[str]:
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
                "Arg. periapsis w",
                elements.argument_of_periastron,
                display.argument_of_periastron,
            ),
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

        lines += [
            "  Phase knowledge:   {0}".format(elements.phase_knowledge.value),
            "  Orientation fully measured: {0}".format(elements.orientation_known),
            "",
            "PROPAGATED STATE at BJD {0:.4f}".format(time_bjd),
        ]

        state = self.state(record, time_bjd)
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

        check = self.energy_check(record, time_bjd)
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
