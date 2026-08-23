"""Application state: the single source of truth the UI and renderer read.

Roadmap section 6.  State owns the catalogue, the selection, the time model
and the derived scientific objects.  The UI reads from here; the renderer
receives a :class:`~astro_explorer.rendering.renderer.SceneDescription`
built from here.  Neither computes science of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from enum import Enum

import astropy.units as u
import pandas as pd

from ..classification import AtmosphericEvidence, classify, classify_vector
from ..data.nasa_archive import SolutionPolicy
from ..data.schema import PlanetRecord, build_planet_record
from ..physics.ephemeris import TimeController, TimeMode
from ..spectroscopy.molecular_evidence import DetectionStatus, EvidenceTable

__all__ = ["ViewMode", "AppState"]


class ViewMode(str, Enum):
    """Top-level UI modes (roadmap section 20)."""

    UNIVERSE = "UNIVERSE"
    SYSTEM = "SYSTEM"
    PLANET = "PLANET"
    SCIENCE = "SCIENCE"
    DYNAMICS = "DYNAMICS"
    DATA = "DATA"


@dataclass
class AppState:
    """Everything the application knows right now."""

    catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    policy: SolutionPolicy = SolutionPolicy.COMPOSITE
    evidence: EvidenceTable = field(default_factory=EvidenceTable)
    time: TimeController = field(default_factory=TimeController)
    view: ViewMode = ViewMode.SYSTEM
    data_source_label: str = ""
    offline: bool = False

    selected_host: str | None = None
    selected_planet: str | None = None

    _record_cache: dict = field(default_factory=dict, repr=False)
    _started: float = field(default_factory=monotonic, repr=False)

    # -- catalogue -------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return not self.catalog.empty

    def hosts(self) -> list[str]:
        if self.catalog.empty or "hostname" not in self.catalog:
            return []
        return sorted(self.catalog["hostname"].dropna().unique().tolist())

    def planets_of(self, host: str) -> list[str]:
        if self.catalog.empty:
            return []
        rows = self.catalog[self.catalog["hostname"] == host]
        return sorted(rows["pl_name"].dropna().tolist())

    def record(self, planet_name: str) -> PlanetRecord | None:
        """Provenance-aware record for a planet, cached."""
        if planet_name in self._record_cache:
            return self._record_cache[planet_name]
        if self.catalog.empty:
            return None
        rows = self.catalog[self.catalog["pl_name"] == planet_name]
        if rows.empty:
            return None
        record = build_planet_record(rows.iloc[0].to_dict(), policy=self.policy)
        self._record_cache[planet_name] = record
        return record

    def system_records(self, host: str) -> list[PlanetRecord]:
        """Every planet of a host, ordered by semimajor axis when known."""
        records = [self.record(name) for name in self.planets_of(host)]
        records = [r for r in records if r is not None]
        return sorted(
            records,
            key=lambda r: (
                r.semimajor_axis.value_in(u.au) is None,
                r.semimajor_axis.value_in(u.au) or 0.0,
            ),
        )

    @property
    def selected_record(self) -> PlanetRecord | None:
        if not self.selected_planet:
            return None
        return self.record(self.selected_planet)

    def select(self, planet_name: str) -> PlanetRecord | None:
        record = self.record(planet_name)
        if record is not None:
            self.selected_planet = planet_name
            self.selected_host = record.host.name
        return record

    # -- time ------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self._started

    def reset_clock(self) -> None:
        self._started = monotonic()

    def set_time_mode(self, mode: TimeMode, *, scale_days_per_second: float | None = None) -> None:
        self.time.mode = mode
        if scale_days_per_second is not None:
            self.time.scale_days_per_second = scale_days_per_second
        self.reset_clock()

    def mean_anomalies(self, host: str, *, now_jd: float | None = None) -> dict[str, float]:
        """Current mean anomaly per planet, omitting those with no phase."""
        elapsed = self.elapsed_seconds
        result: dict[str, float] = {}
        for record in self.system_records(host):
            anomaly, _assumed = self.time.mean_anomaly(
                record.elements, elapsed, now_jd=now_jd
            )
            if anomaly is not None:
                result[record.name] = anomaly
        return result

    # -- evidence and classification -------------------------------------
    def atmospheric_evidence(self, planet_name: str) -> AtmosphericEvidence:
        """Collapse the evidence table into one classification axis."""
        entries = self.evidence.for_planet(planet_name)
        if not entries:
            return AtmosphericEvidence.UNKNOWN
        statuses = {entry.detection_status for entry in entries}
        if DetectionStatus.DETECTED in statuses:
            return AtmosphericEvidence.DETECTED
        if statuses == {DetectionStatus.NOT_DETECTED}:
            return AtmosphericEvidence.ABSENT
        if DetectionStatus.TENTATIVE in statuses or DetectionStatus.DISPUTED in statuses:
            return AtmosphericEvidence.CONSTRAINED
        return AtmosphericEvidence.UNKNOWN

    def classification(self, planet_name: str):
        record = self.record(planet_name)
        if record is None:
            return None, None
        traditional = classify(
            record.radius_earth, record.equilibrium_temperature, record.mass_earth
        )
        vector = classify_vector(
            radius_earth=record.radius_earth,
            mass_earth=record.mass_earth,
            equilibrium_temperature=record.equilibrium_temperature,
            period_days=record.elements.period,
            eccentricity=record.elements.eccentricity,
            insolation_earth=record.insolation,
            atmosphere=self.atmospheric_evidence(planet_name),
        )
        return traditional, vector

    # -- reporting -------------------------------------------------------
    def data_report(self) -> list[str]:
        """Lines for the DATA view (roadmap section 20)."""
        lines = [
            "Source:            {0}".format(self.data_source_label or "not loaded"),
            "Solution policy:   {0}".format(self.policy.label),
            "                   {0}".format(self.policy.caveat),
            "Mode:              {0}".format("offline snapshot" if self.offline else "live"),
            "Planets loaded:    {0}".format(
                0 if self.catalog.empty else int(self.catalog["pl_name"].nunique())
            ),
            "Time model:        {0}".format(self.time.describe()),
            "Evidence records:  {0}".format(len(self.evidence)),
        ]
        return lines
