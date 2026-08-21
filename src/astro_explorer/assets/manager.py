"""Resource resolution that survives PyInstaller (roadmap section 3.11).

``exoplanet_analyzer.spec`` bundles ``tables`` and ``atmospheric_signatures.json``,
but the application located them relative to the process working directory.
Launch the frozen executable from anywhere else and the data disappears.

:class:`ResourceManager` resolves declared resources against, in order:

1. an explicit override (an environment variable or a constructor argument);
2. the PyInstaller bundle directory ``sys._MEIPASS``;
3. the directory containing the running module;
4. the project root, when running from a source checkout;
5. the current working directory - last, because it is the least reliable.

It never walks the filesystem looking for data.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Resource", "ResourceManager", "DECLARED_RESOURCES", "ResourceNotFound"]


class ResourceNotFound(FileNotFoundError):
    """Raised when a required resource cannot be located in any root."""


@dataclass(frozen=True)
class Resource:
    """A file or directory the application declares that it needs."""

    key: str
    relative_path: str
    is_directory: bool = False
    required: bool = False
    description: str = ""


#: Everything the application expects to find on disk.  The PyInstaller spec
#: is generated from this list, so declaring a resource here is enough to get
#: it bundled.
DECLARED_RESOURCES = (
    Resource(
        "spectra",
        "tables",
        is_directory=True,
        required=False,
        description="Local NASA atmospheric spectroscopy tables (.tbl)",
    ),
    Resource(
        "signatures",
        "atmospheric_signatures.json",
        description="Indicative molecular band centres for plot overlays",
    ),
    Resource(
        "molecular_evidence",
        "molecular_evidence.csv",
        description="Structured molecular detection evidence with provenance",
    ),
    Resource(
        "legacy_molecules",
        "planet_molecules.csv",
        description="Legacy flat molecule list, read only if the evidence table is absent",
    ),
    Resource(
        "asset_manifest",
        "assets/manifest.json",
        description="Texture and concept-art provenance manifest",
    ),
    Resource(
        "shaders",
        "src/astro_explorer/rendering/shaders",
        is_directory=True,
        description="GLSL sources for the modern OpenGL renderer",
    ),
)


class ResourceManager:
    """Locates declared resources without relying on the working directory."""

    #: Environment variable that overrides the search entirely.
    ENV_OVERRIDE = "ASTRO_EXPLORER_DATA"

    def __init__(self, project_root: Path | str | None = None, extra_roots=()):
        self._explicit_root = Path(project_root) if project_root else None
        self._extra_roots = [Path(root) for root in extra_roots]
        self._cache: dict[str, Path | None] = {}

    # -- roots -----------------------------------------------------------
    @property
    def bundle_root(self) -> Path | None:
        """``sys._MEIPASS`` when running from a PyInstaller bundle."""
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else None

    @property
    def module_root(self) -> Path:
        """The directory containing the installed package."""
        return Path(__file__).resolve().parent.parent

    @property
    def project_root(self) -> Path:
        """The repository root when running from a source checkout.

        ``src/astro_explorer/assets/manager.py`` is three levels below the
        repository root.
        """
        if self._explicit_root:
            return self._explicit_root
        return Path(__file__).resolve().parents[3]

    def roots(self) -> list[Path]:
        """Search roots, most trustworthy first."""
        found: list[Path] = []

        override = os.environ.get(self.ENV_OVERRIDE)
        if override:
            found.append(Path(override))
        if self._explicit_root:
            found.append(self._explicit_root)
        found.extend(self._extra_roots)
        if self.bundle_root:
            found.append(self.bundle_root)
        found.append(self.module_root)
        found.append(self.project_root)
        found.append(Path.cwd())

        unique: list[Path] = []
        seen: set[Path] = set()
        for root in found:
            try:
                resolved = root.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return unique

    # -- lookup ----------------------------------------------------------
    def find(self, key: str) -> Path | None:
        """Resolve a declared resource, or None when it is absent."""
        if key in self._cache:
            return self._cache[key]

        resource = next((r for r in DECLARED_RESOURCES if r.key == key), None)
        if resource is None:
            raise KeyError("undeclared resource {0!r}".format(key))

        result: Path | None = None
        for root in self.roots():
            candidate = root / resource.relative_path
            if candidate.exists():
                exists_correctly = (
                    candidate.is_dir() if resource.is_directory else candidate.is_file()
                )
                if exists_correctly:
                    result = candidate
                    break
            # Some data files ship inside the tables/ directory rather than
            # beside it - planet_molecules.csv is one.
            if not resource.is_directory:
                nested = root / "tables" / resource.relative_path
                if nested.is_file():
                    result = nested
                    break

            # A bundled directory may be flattened to the bundle root.
            if resource.is_directory:
                flattened = root / Path(resource.relative_path).name
                if flattened.is_dir():
                    result = flattened
                    break

        self._cache[key] = result
        return result

    def require(self, key: str) -> Path:
        """Resolve a resource or raise, naming every root that was tried."""
        path = self.find(key)
        if path is None:
            roots = "\n  ".join(str(r) for r in self.roots())
            raise ResourceNotFound(
                "required resource {0!r} not found. Searched:\n  {1}".format(key, roots)
            )
        return path

    def spectra_directories(self) -> list[Path]:
        """Every directory that might hold ``.tbl`` spectra."""
        directories: list[Path] = []
        seen: set[Path] = set()
        for root in self.roots():
            for candidate in (root / "tables", root):
                if not candidate.is_dir():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                if any(resolved.glob("*.tbl")):
                    seen.add(resolved)
                    directories.append(resolved)
        return directories

    def user_data_dir(self) -> Path:
        """Writable location for the local snapshot and caches."""
        override = os.environ.get("ASTRO_EXPLORER_HOME")
        if override:
            path = Path(override)
        elif sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home())
            path = Path(base) / "AstroExplorer"
        elif sys.platform == "darwin":
            path = Path.home() / "Library" / "Application Support" / "AstroExplorer"
        else:
            base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
            path = Path(base) / "astro-explorer"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def report(self) -> list[str]:
        """Diagnostic lines for the DATA view."""
        lines = ["Resource roots:"]
        lines.extend("  " + str(root) for root in self.roots())
        lines.append("Declared resources:")
        for resource in DECLARED_RESOURCES:
            path = self.find(resource.key)
            lines.append(
                "  {0:<20} {1}".format(resource.key, path if path else "NOT FOUND")
            )
        return lines
