"""Astronomical frames built on Astropy (roadmap sections 4.6, 4.7, 9).

Two prototype bugs are fixed structurally rather than by patching:

* a non-positive or unusable parallax must never become a placeholder
  distance such as one billion parsecs (section 4.6).  :func:`distance_from`
  returns an UNKNOWN parameter and the object is excluded from geometry that
  needs a distance;
* RA/Dec/distance are converted through :class:`~astropy.coordinates.SkyCoord`
  rather than by hand, so proper motion, radial velocity and epoch
  transformations can be added later without rewriting the maths
  (section 4.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Distance, Galactic, Galactocentric, SkyCoord

from ..provenance import Parameter, Status, derived, measured, unknown

__all__ = [
    "Frame",
    "SkyPosition",
    "distance_from_parallax",
    "sky_position",
    "cartesian_pc",
    "separation_pc",
]


class Frame(str, Enum):
    """Supported galaxy-scale frames (roadmap section 9.1)."""

    ICRS = "icrs"
    GALACTIC = "galactic"
    GALACTOCENTRIC = "galactocentric"

    @property
    def astropy_frame(self):
        return {
            Frame.ICRS: ICRS,
            Frame.GALACTIC: Galactic,
            Frame.GALACTOCENTRIC: Galactocentric,
        }[self]


#: Parallaxes at or below this value carry no usable distance information.
MIN_USEFUL_PARALLAX_MAS = 1e-3


def distance_from_parallax(
    parallax_mas: float | Parameter | None,
    *,
    catalog_distance_pc: float | Parameter | None = None,
) -> Parameter:
    """Distance from parallax, or the catalogue distance, or UNKNOWN.

    Roadmap section 4.6.  The order of preference is:

    1. a usable positive parallax (``d = 1 / parallax``);
    2. a trusted catalogue distance;
    3. UNKNOWN - never an arbitrary placeholder.

    A negative parallax is a real and common Gaia measurement outcome for
    distant faint sources; it means the astrometric solution is consistent
    with zero, not that the star is a billion parsecs away.
    """
    value = parallax_mas.value_in(u.mas) if isinstance(parallax_mas, Parameter) else parallax_mas

    if value is not None and np.isfinite(value) and value > MIN_USEFUL_PARALLAX_MAS:
        distance = Distance(parallax=value * u.mas)
        error = None
        if isinstance(parallax_mas, Parameter) and parallax_mas.error_plus:
            # d = 1/p, so dd/d = dp/p to first order.
            error = float(distance.to_value(u.pc)) * (parallax_mas.error_plus / abs(value))
        return derived(
            float(distance.to_value(u.pc)),
            u.pc,
            error_plus=error,
            error_minus=error,
            provenance="parallax_inversion",
            note="1/parallax; unreliable at low signal-to-noise",
        )

    if catalog_distance_pc is not None:
        if isinstance(catalog_distance_pc, Parameter):
            if catalog_distance_pc.is_known:
                return catalog_distance_pc.to(u.pc)
        elif np.isfinite(catalog_distance_pc) and catalog_distance_pc > 0:
            return measured(
                float(catalog_distance_pc), u.pc, provenance="catalog_distance"
            )

    return unknown(
        u.pc,
        provenance="parallax_inversion",
        note="parallax not usable and no catalogue distance available",
    )


@dataclass(frozen=True)
class SkyPosition:
    """A celestial position whose distance may legitimately be unknown."""

    name: str
    ra: Parameter
    dec: Parameter
    distance: Parameter
    frame: Frame = Frame.ICRS

    @property
    def has_distance(self) -> bool:
        """True when 3D geometry is defined for this object."""
        return self.distance.is_known and self.distance.status is not Status.ASSUMED_FOR_VISUALIZATION

    @property
    def skycoord(self) -> SkyCoord | None:
        """Astropy SkyCoord, or None when RA/Dec are missing.

        The coordinate carries a distance only when one is actually known,
        so any 3D operation on a distance-less object fails loudly instead
        of silently placing it somewhere.
        """
        ra = self.ra.value_in(u.deg)
        dec = self.dec.value_in(u.deg)
        if ra is None or dec is None:
            return None
        if self.has_distance:
            return SkyCoord(
                ra=ra * u.deg,
                dec=dec * u.deg,
                distance=self.distance.value_in(u.pc) * u.pc,
                frame="icrs",
            )
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    def cartesian_pc(self, frame: Frame = Frame.ICRS) -> np.ndarray | None:
        """Cartesian position in parsecs, or None without a distance."""
        coord = self.skycoord
        if coord is None or not self.has_distance:
            return None
        transformed = coord.transform_to(frame.astropy_frame()) if frame is not Frame.ICRS else coord
        cart = transformed.cartesian
        return np.array(
            [
                cart.x.to_value(u.pc),
                cart.y.to_value(u.pc),
                cart.z.to_value(u.pc),
            ],
            dtype=np.float64,
        )

    def light_years(self) -> Parameter:
        """Distance expressed in light years, preserving status."""
        if not self.distance.is_known:
            return unknown(u.lyr, provenance=self.distance.provenance)
        return self.distance.to(u.lyr)

    def light_travel_time(self) -> Parameter:
        """Light travel time from Earth (roadmap section 10.1)."""
        if not self.distance.is_known:
            return unknown(u.yr, provenance="light_travel_time")
        return derived(
            self.distance.value_in(u.lyr),
            u.yr,
            provenance="light_travel_time(distance)",
        )


def sky_position(
    name: str,
    ra_deg: float | Parameter | None,
    dec_deg: float | Parameter | None,
    *,
    parallax_mas: float | Parameter | None = None,
    catalog_distance_pc: float | Parameter | None = None,
) -> SkyPosition:
    """Build a :class:`SkyPosition`, resolving the distance policy."""

    def as_param(value, unit) -> Parameter:
        if isinstance(value, Parameter):
            return value.to(unit)
        if value is None or not np.isfinite(value):
            return unknown(unit)
        return measured(float(value), unit, provenance="catalog")

    return SkyPosition(
        name=name,
        ra=as_param(ra_deg, u.deg),
        dec=as_param(dec_deg, u.deg),
        distance=distance_from_parallax(parallax_mas, catalog_distance_pc=catalog_distance_pc),
    )


def cartesian_pc(ra_deg, dec_deg, distance_pc, frame: Frame = Frame.ICRS) -> np.ndarray:
    """Vectorised RA/Dec/distance to Cartesian parsecs via Astropy.

    Rows whose distance is not finite and positive come back as NaN, so a
    caller that forgets to filter gets NaN rather than a plausible-looking
    wrong position.
    """
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=np.float64))
    dist = np.atleast_1d(np.asarray(distance_pc, dtype=np.float64))

    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dist) & (dist > 0)
    out = np.full((ra.size, 3), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out

    coord = SkyCoord(
        ra=ra[valid] * u.deg,
        dec=dec[valid] * u.deg,
        distance=dist[valid] * u.pc,
        frame="icrs",
    )
    if frame is not Frame.ICRS:
        coord = coord.transform_to(frame.astropy_frame())
    cart = coord.cartesian
    out[valid, 0] = cart.x.to_value(u.pc)
    out[valid, 1] = cart.y.to_value(u.pc)
    out[valid, 2] = cart.z.to_value(u.pc)
    return out


def separation_pc(a: SkyPosition, b: SkyPosition) -> Parameter:
    """3D separation between two objects (roadmap section 10.3).

    Returns UNKNOWN if either object lacks a distance, because the
    separation genuinely is not computable in that case.
    """
    pos_a = a.cartesian_pc()
    pos_b = b.cartesian_pc()
    if pos_a is None or pos_b is None:
        return unknown(
            u.pc,
            provenance="separation",
            note="at least one object has no reliable distance",
        )
    return derived(
        float(np.linalg.norm(pos_a - pos_b)),
        u.pc,
        provenance="separation({0}, {1})".format(a.name, b.name),
    )
