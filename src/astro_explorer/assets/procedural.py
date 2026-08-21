"""Procedural planet appearance derived from measured parameters.

Roadmap sections 4.9 and 4.10.  The prototype drew flat coloured circles,
which is fine for a proof of concept but says nothing about the planet.  The
appearance here is computed from equilibrium temperature, bulk density and
irradiation, and every result is tagged
:attr:`~astro_explorer.assets.manifest.AssetType.SCIENTIFIC_PROCEDURAL`, so
the UI must label it "actual appearance unknown".

Star colours are separated into a physically-derived display colour and a
stylised, visibility-enhanced variant, because saturating a red dwarf so it
is visible on screen is a rendering choice, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
import numpy as np

from ..provenance import Parameter
from .manifest import AssetType

__all__ = [
    "MaterialClass",
    "PlanetMaterial",
    "planet_material",
    "blackbody_rgb",
    "star_display_color",
]


class MaterialClass(str, Enum):
    """Which shader family a body should be drawn with (roadmap section 15)."""

    ROCKY = "ROCKY"
    ICY = "ICY"
    GAS_GIANT = "GAS_GIANT"
    HOT_GIANT = "HOT_GIANT"
    ULTRA_HOT_GIANT = "ULTRA_HOT_GIANT"
    UNKNOWN = "UNKNOWN"

    @property
    def shader(self) -> str:
        return {
            MaterialClass.ROCKY: "rocky",
            MaterialClass.ICY: "rocky",
            MaterialClass.GAS_GIANT: "gas_giant",
            MaterialClass.HOT_GIANT: "gas_giant",
            MaterialClass.ULTRA_HOT_GIANT: "gas_giant",
            MaterialClass.UNKNOWN: "rocky",
        }[self]


@dataclass(frozen=True)
class PlanetMaterial:
    """Display parameters for one planet, with their provenance."""

    material_class: MaterialClass
    base_color: tuple[float, float, float]
    emissive: float = 0.0
    banding: float = 0.0
    roughness: float = 0.8
    asset_type: AssetType = AssetType.SCIENTIFIC_PROCEDURAL
    basis: str = ""

    @property
    def badge(self) -> str:
        return self.asset_type.badge

    @property
    def shader(self) -> str:
        return self.material_class.shader


#: Indicative colours, chosen for legibility rather than measured.
_ROCK_COLD = (0.55, 0.55, 0.58)
_ROCK_WARM = (0.62, 0.48, 0.38)
_ROCK_HOT = (0.72, 0.34, 0.22)
_ICE = (0.78, 0.86, 0.92)
_GAS_COLD = (0.72, 0.68, 0.58)
_GAS_WARM = (0.80, 0.62, 0.42)
_GAS_HOT = (0.86, 0.42, 0.26)


def _lerp(a, b, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def planet_material(
    radius_earth: Parameter | float | None,
    equilibrium_temperature: Parameter | float | None,
    density: Parameter | float | None = None,
) -> PlanetMaterial:
    """Choose a material class and colour from measured parameters."""

    def numeric(value, unit):
        if isinstance(value, Parameter):
            return value.value_in(unit)
        return value

    radius = numeric(radius_earth, u.R_earth)
    teq = numeric(equilibrium_temperature, u.K)
    rho = numeric(density, u.g / u.cm**3)

    if radius is None:
        return PlanetMaterial(
            MaterialClass.UNKNOWN,
            (0.5, 0.5, 0.5),
            basis="radius unknown; generic placeholder",
        )

    is_giant = radius >= 6.0 or (rho is not None and rho < 2.0 and radius >= 3.0)

    if is_giant:
        if teq is None:
            return PlanetMaterial(
                MaterialClass.GAS_GIANT, _GAS_COLD, banding=0.6,
                basis="radius {0:.3g} R_earth; temperature unknown".format(radius),
            )
        if teq >= 2000.0:
            return PlanetMaterial(
                MaterialClass.ULTRA_HOT_GIANT,
                _GAS_HOT,
                emissive=float(np.clip((teq - 1800.0) / 1500.0, 0.0, 1.0)),
                banding=0.2,
                basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
            )
        if teq >= 1000.0:
            return PlanetMaterial(
                MaterialClass.HOT_GIANT,
                _lerp(_GAS_WARM, _GAS_HOT, (teq - 1000.0) / 1000.0),
                emissive=float(np.clip((teq - 1000.0) / 2000.0, 0.0, 0.4)),
                banding=0.45,
                basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
            )
        return PlanetMaterial(
            MaterialClass.GAS_GIANT,
            _lerp(_GAS_COLD, _GAS_WARM, (teq - 100.0) / 900.0),
            banding=0.7,
            basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
        )

    if teq is None:
        return PlanetMaterial(
            MaterialClass.ROCKY, _ROCK_COLD,
            basis="radius {0:.3g} R_earth; temperature unknown".format(radius),
        )
    if teq < 170.0:
        return PlanetMaterial(
            MaterialClass.ICY, _ICE, roughness=0.35,
            basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
        )
    if teq < 700.0:
        return PlanetMaterial(
            MaterialClass.ROCKY,
            _lerp(_ROCK_COLD, _ROCK_WARM, (teq - 170.0) / 530.0),
            basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
        )
    return PlanetMaterial(
        MaterialClass.ROCKY,
        _lerp(_ROCK_WARM, _ROCK_HOT, (teq - 700.0) / 1300.0),
        emissive=float(np.clip((teq - 1200.0) / 1500.0, 0.0, 0.8)),
        roughness=0.9,
        basis="radius {0:.3g} R_earth, T_eq {1:.0f} K".format(radius, teq),
    )


# Planckian-locus fit (Kim et al. 2002), valid for 1667 K <= T <= 25000 K.
def _planckian_xy(temperature: float) -> tuple[float, float]:
    t = float(np.clip(temperature, 1667.0, 25000.0))
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390

    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def blackbody_rgb(temperature: float) -> tuple[float, float, float]:
    """Physically-derived display colour for a blackbody of temperature T.

    Goes through the Planckian locus in CIE xy, then XYZ to linear sRGB with
    gamma encoding.  This is still a *display* colour: tone mapping and the
    sRGB gamut mean no screen shows a star's true radiance.
    """
    x, y = _planckian_xy(temperature)
    if y <= 0:
        return (1.0, 1.0, 1.0)

    xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
    linear = _XYZ_TO_SRGB @ xyz
    linear = np.clip(linear, 0.0, None)
    peak = float(np.max(linear))
    if peak > 0:
        linear = linear / peak

    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return tuple(float(np.clip(channel, 0.0, 1.0)) for channel in encoded)


@dataclass(frozen=True)
class StarColor:
    """Both colours a star needs, kept apart (roadmap section 4.10)."""

    scientific: tuple[float, float, float]
    stylized: tuple[float, float, float]
    temperature_k: float

    @property
    def caveat(self) -> str:
        return (
            "Display colour derived from effective temperature; tone mapping "
            "and display gamut mean it remains a visualisation."
        )


def star_display_color(
    effective_temperature: Parameter | float | None,
    *,
    saturation_boost: float = 0.35,
) -> StarColor | None:
    """Scientific and stylised star colours, returned as separate fields.

    A cool M dwarf is a deep orange-red that nearly disappears against a
    dark background; the stylised colour lifts it towards white for
    visibility.  The two are never conflated.
    """
    temperature = (
        effective_temperature.value_in(u.K)
        if isinstance(effective_temperature, Parameter)
        else effective_temperature
    )
    if temperature is None or not np.isfinite(temperature) or temperature <= 0:
        return None

    scientific = blackbody_rgb(temperature)
    boosted = tuple(
        float(np.clip(channel + (1.0 - channel) * saturation_boost, 0.0, 1.0))
        for channel in scientific
    )
    return StarColor(scientific=scientific, stylized=boosted, temperature_k=float(temperature))


__all__ += ["StarColor"]
