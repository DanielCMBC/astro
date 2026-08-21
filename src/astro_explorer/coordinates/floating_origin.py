"""Hierarchical coordinate frames and the floating origin.

Roadmap sections 4.2, 9.2, 9.3 and 9.4.

The prototype multiplied orbital coordinates by ``0.005`` and called it "AU
to parsecs".  The true factor is ``4.8481368e-6``, so systems were roughly
a thousand times oversized.  The deeper problem is that no single float32
coordinate space can hold both parsec-scale and kilometre-scale geometry:
float32 has about seven significant decimal digits, and one parsec expressed
in kilometres already needs fourteen.

The fix is structural.  Geometry lives in one of three frames, CPU maths is
float64, and positions are rebased around the active camera before they are
narrowed to float32 for the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import astropy.units as u
import numpy as np

__all__ = [
    "Scale",
    "FloatingOrigin",
    "SceneGraph",
    "AU_TO_PC",
    "PC_TO_AU",
    "KM_TO_AU",
    "AU_TO_KM",
]

#: Exact conversions, taken from Astropy rather than typed in by hand.
AU_TO_PC = float((1.0 * u.au).to_value(u.pc))
PC_TO_AU = float((1.0 * u.pc).to_value(u.au))
AU_TO_KM = float((1.0 * u.au).to_value(u.km))
KM_TO_AU = float((1.0 * u.km).to_value(u.au))

#: float32 keeps ~7 significant decimal digits.  Positions larger than this
#: many display units lose sub-unit precision and must be rebased first.
FLOAT32_SAFE_MAGNITUDE = 1.0e6


class Scale(str, Enum):
    """The three nested frames of roadmap section 9."""

    GALAXY = "GALAXY"
    """Origin at the Sun or the galactic centre; unit parsec."""

    SYSTEM = "SYSTEM"
    """Origin at the host star; unit AU."""

    PLANET = "PLANET"
    """Origin at the planet; unit kilometre."""

    @property
    def unit(self) -> u.UnitBase:
        return {Scale.GALAXY: u.pc, Scale.SYSTEM: u.au, Scale.PLANET: u.km}[self]

    def to_parsec(self, value):
        """Convert a magnitude in this frame's unit to parsecs."""
        return np.asarray(value, dtype=np.float64) * float((1.0 * self.unit).to_value(u.pc))

    def from_parsec(self, value):
        """Convert parsecs to a magnitude in this frame's unit."""
        return np.asarray(value, dtype=np.float64) * float((1.0 * u.pc).to_value(self.unit))


@dataclass
class FloatingOrigin:
    """Rebases world coordinates around the active camera.

    All book-keeping is float64 in parsecs.  :meth:`to_render_space` is the
    single boundary at which float32 appears, exactly as roadmap section 6
    requires ("convert to unitless GPU values only at the rendering
    boundary").
    """

    #: Current origin, in parsecs, float64.
    origin_pc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    #: Frame whose unit render space is expressed in.
    scale: Scale = Scale.GALAXY

    #: Rebase when the camera drifts further than this from the origin,
    #: measured in the active frame's unit.
    rebase_threshold: float = 1.0e4

    def __post_init__(self) -> None:
        self.origin_pc = np.asarray(self.origin_pc, dtype=np.float64).reshape(3)

    # -- origin management -----------------------------------------------
    def set_origin_pc(self, position_pc) -> None:
        """Move the origin to an absolute position in parsecs."""
        self.origin_pc = np.asarray(position_pc, dtype=np.float64).reshape(3)

    def set_origin(self, position, scale: Scale | None = None) -> None:
        """Move the origin using coordinates in ``scale``'s unit."""
        scale = scale or self.scale
        self.origin_pc = np.asarray(scale.to_parsec(position), dtype=np.float64).reshape(3)

    def maybe_rebase(self, camera_position) -> bool:
        """Rebase onto the camera if it has drifted too far.

        Returns True when the origin moved, so the caller knows cached
        render-space buffers are stale.
        """
        camera = np.asarray(camera_position, dtype=np.float64).reshape(3)
        if np.linalg.norm(camera) <= self.rebase_threshold:
            return False
        self.origin_pc = self.origin_pc + self.scale.to_parsec(camera)
        return True

    # -- transforms ------------------------------------------------------
    def world_to_local(self, position_pc, scale: Scale | None = None) -> np.ndarray:
        """Absolute parsecs to origin-relative coordinates in ``scale``'s unit."""
        scale = scale or self.scale
        absolute = np.atleast_2d(np.asarray(position_pc, dtype=np.float64))
        relative_pc = absolute - self.origin_pc
        local = scale.from_parsec(relative_pc)
        return local[0] if np.ndim(position_pc) == 1 else local

    def local_to_world(self, position, scale: Scale | None = None) -> np.ndarray:
        """Origin-relative coordinates in ``scale``'s unit to absolute parsecs."""
        scale = scale or self.scale
        local = np.atleast_2d(np.asarray(position, dtype=np.float64))
        absolute = scale.to_parsec(local) + self.origin_pc
        return absolute[0] if np.ndim(position) == 1 else absolute

    def to_render_space(self, position_pc, scale: Scale | None = None) -> np.ndarray:
        """The one place float32 is allowed to appear.

        Raises
        ------
        ValueError
            If a rebased coordinate is still too large for float32 to
            represent without losing sub-unit precision.  Silently emitting
            a degenerate position is exactly the class of bug this module
            exists to prevent.
        """
        local = np.asarray(self.world_to_local(position_pc, scale), dtype=np.float64)
        finite = local[np.isfinite(local)]
        if finite.size and np.max(np.abs(finite)) > FLOAT32_SAFE_MAGNITUDE:
            raise ValueError(
                "coordinate magnitude {0:.3g} exceeds the float32 safe range; "
                "rebase the floating origin or switch to a smaller frame".format(
                    float(np.max(np.abs(finite)))
                )
            )
        return local.astype(np.float32)


@dataclass
class SceneGraph:
    """Places a planet correctly relative to its host and the galaxy.

    This is the object that makes the 0.005 bug impossible to reintroduce:
    a planet's AU-scale offset is never added to a parsec-scale star
    position in the same float32 buffer.  In SYSTEM scale the star sits at
    the origin and the planet's AU coordinates are used directly; only in
    GALAXY scale is the (negligible but correctly converted) AU offset added
    in parsecs.
    """

    origin: FloatingOrigin = field(default_factory=FloatingOrigin)

    def host_render_position(self, host_position_pc) -> np.ndarray:
        """Render-space position of a host star."""
        return self.origin.to_render_space(host_position_pc)

    def planet_render_position(
        self,
        host_position_pc,
        planet_offset_au,
    ) -> np.ndarray:
        """Render-space position of a planet orbiting ``host_position_pc``.

        ``planet_offset_au`` is the star-centred orbital position in AU, as
        produced by :mod:`astro_explorer.physics.orbital_elements`.
        """
        host_pc = np.asarray(host_position_pc, dtype=np.float64).reshape(3)
        offset_au = np.asarray(planet_offset_au, dtype=np.float64).reshape(3)

        if self.origin.scale is Scale.SYSTEM:
            # The origin is (or should be) the host star: work purely in AU
            # and never round-trip the orbit through parsecs.
            host_local_au = np.asarray(
                self.origin.world_to_local(host_pc, Scale.SYSTEM), dtype=np.float64
            )
            local = host_local_au + offset_au
            if np.max(np.abs(local)) > FLOAT32_SAFE_MAGNITUDE:
                raise ValueError("system-frame coordinate exceeds the float32 safe range")
            return local.astype(np.float32)

        if self.origin.scale is Scale.PLANET:
            host_local_km = np.asarray(
                self.origin.world_to_local(host_pc, Scale.PLANET), dtype=np.float64
            )
            local = host_local_km + offset_au * AU_TO_KM
            if np.max(np.abs(local)) > FLOAT32_SAFE_MAGNITUDE:
                raise ValueError("planet-frame coordinate exceeds the float32 safe range")
            return local.astype(np.float32)

        # GALAXY scale: the orbit is far below one parsec, but convert it
        # with the exact factor rather than an invented one.
        absolute_pc = host_pc + offset_au * AU_TO_PC
        return self.origin.to_render_space(absolute_pc, Scale.GALAXY)

    def enter_system(self, host_position_pc) -> None:
        """Switch to the host-star frame (roadmap section 9.2)."""
        self.origin.set_origin_pc(host_position_pc)
        self.origin.scale = Scale.SYSTEM
        self.origin.rebase_threshold = 1.0e4  # AU

    def enter_planet(self, host_position_pc, planet_offset_au) -> None:
        """Switch to the planet frame (roadmap section 9.3)."""
        offset_pc = np.asarray(planet_offset_au, dtype=np.float64).reshape(3) * AU_TO_PC
        self.origin.set_origin_pc(np.asarray(host_position_pc, dtype=np.float64).reshape(3) + offset_pc)
        self.origin.scale = Scale.PLANET
        self.origin.rebase_threshold = 1.0e7  # km

    def enter_galaxy(self, origin_position_pc=None) -> None:
        """Switch to the galaxy frame (roadmap section 9.1)."""
        self.origin.set_origin_pc(
            np.zeros(3) if origin_position_pc is None else origin_position_pc
        )
        self.origin.scale = Scale.GALAXY
        self.origin.rebase_threshold = 1.0e4  # pc
