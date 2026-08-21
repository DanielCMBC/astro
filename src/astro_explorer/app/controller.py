"""Application controller: loading, synchronising, and answering questions.

The controller is the only place that decides where data comes from.  It
enforces the offline-first rule of roadmap section 11: startup reads the
local snapshot, and a network synchronisation is an explicit, optional
action whose failure changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u

from ..assets.manager import ResourceManager
from ..coordinates.frames import SkyPosition, separation_pc
from ..coordinates.transforms import describe_distance
from ..data.nasa_archive import SolutionPolicy
from ..data.repository import CatalogRepository
from ..data.synchronizer import SyncResult, synchronize
from ..provenance import unknown
from ..spectroscopy.ipac import read_planet_spectra
from ..spectroscopy.molecular_evidence import load_evidence
from ..spectroscopy.normalization import load_signatures
from .state import AppState

__all__ = ["Controller"]


@dataclass
class Controller:
    """Wires the resource manager, repository and state together."""

    resources: ResourceManager
    repository: CatalogRepository
    state: AppState

    @classmethod
    def create(cls, project_root=None) -> "Controller":
        resources = ResourceManager(project_root)
        repository = CatalogRepository(resources.user_data_dir() / "store")
        return cls(resources=resources, repository=repository, state=AppState())

    # -- loading ---------------------------------------------------------
    def load_local(self) -> bool:
        """Load the offline snapshot.  Returns True when data is available."""
        frame = self.repository.load()
        if frame is None:
            return False

        info = self.repository.snapshot_info()
        self.state.catalog = frame
        self.state.policy = info.solution_policy if info else SolutionPolicy.COMPOSITE
        self.state.offline = True
        self.state.data_source_label = "local snapshot, synchronised {0}".format(
            info.retrieved if info else "unknown"
        )
        self.state._record_cache.clear()
        self.load_evidence()
        return True

    def load_evidence(self) -> None:
        """Load molecular evidence, preferring the structured table."""
        path = self.resources.find("molecular_evidence")
        if path is None:
            path = self.resources.find("legacy_molecules")
        if path is not None:
            self.state.evidence = load_evidence(path)

    def signatures(self):
        path = self.resources.find("signatures")
        return load_signatures(path) if path else []

    def synchronize(self, policy: SolutionPolicy | None = None) -> SyncResult:
        """Fetch, validate and atomically replace the local snapshot."""
        policy = policy or self.state.policy
        result = synchronize(self.repository, policy)
        if result.succeeded:
            self.load_local()
            self.state.offline = False
            self.state.data_source_label = "{0}, synchronised {1}".format(
                policy.label, result.snapshot.retrieved if result.snapshot else "just now"
            )
        return result

    def adopt_frame(self, frame, policy: SolutionPolicy, label: str) -> None:
        """Use an already-loaded frame, e.g. a legacy cache file."""
        self.state.catalog = frame
        self.state.policy = policy
        self.state.data_source_label = label
        self.state._record_cache.clear()
        self.load_evidence()

    # -- queries ---------------------------------------------------------
    def spectra_for(self, planet_name: str):
        """Local atmospheric spectra, kept separate per measurement."""
        return read_planet_spectra(planet_name, self.resources.spectra_directories())

    def distance_lines(self, planet_name: str) -> list[str]:
        """Earth-to-system distance block (roadmap section 10.1)."""
        record = self.state.record(planet_name)
        if record is None or record.host.position is None:
            return ["Distance: unknown"]
        return describe_distance(record.host.position.distance)

    def separation_between(self, planet_name: str, other_host: str):
        """Distance from a selected planet's system to another star (10.3).

        The planet's orbital offset is included when it is known, although at
        interstellar scales it is almost always negligible.
        """
        record = self.state.record(planet_name)
        if record is None or record.host.position is None:
            return unknown(u.pc, provenance="separation", note="host position unknown")

        others = [r for r in self.state.system_records(other_host) if r.host.position]
        if not others:
            return unknown(u.pc, provenance="separation", note="target host position unknown")

        return separation_pc(record.host.position, others[0].host.position)

    def data_report(self) -> list[str]:
        lines = list(self.state.data_report())
        info = self.repository.snapshot_info()
        if info is not None:
            lines.append("")
            lines.extend(info.describe())
        lines.append("")
        lines.extend(self.resources.report())
        return lines
