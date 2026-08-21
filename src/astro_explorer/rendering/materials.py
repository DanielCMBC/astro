"""Material and uniform definitions for the shader families.

Roadmap section 15: there is no universal planet shader.  Each material
class maps to one program and one set of uniforms.  Uniform *values* come
from the science layer; this module only declares their names, defaults and
which program consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MaterialDefinition", "MATERIALS", "material_for", "atmosphere_uniforms"]


@dataclass(frozen=True)
class MaterialDefinition:
    """One shader program plus its default uniform values."""

    material_id: str
    program: str
    uniforms: dict = field(default_factory=dict)
    blend: bool = False
    cull_back_faces: bool = True

    def with_values(self, **overrides) -> dict:
        """Uniform dictionary with ``overrides`` applied.

        Unknown uniform names are rejected: a typo that silently does
        nothing is a rendering bug that is very hard to see.
        """
        unknown = set(overrides) - set(self.uniforms)
        if unknown:
            raise KeyError(
                "material {0!r} has no uniform(s) {1}".format(
                    self.material_id, ", ".join(sorted(unknown))
                )
            )
        return self.uniforms | overrides


MATERIALS = {
    "star": MaterialDefinition(
        material_id="star",
        program="star",
        uniforms={
            "u_limb_darkening": 0.6,  # Eddington approximation
            "u_granulation": 0.25,
            "u_exposure": 1.6,
            "u_time": 0.0,
        },
    ),
    "rocky": MaterialDefinition(
        material_id="rocky",
        program="rocky",
        uniforms={
            "u_roughness": 0.85,
            "u_ambient": 0.03,
            "u_terrain_strength": 0.35,
            "u_use_texture": False,
        },
    ),
    "icy": MaterialDefinition(
        material_id="icy",
        program="rocky",
        uniforms={
            "u_roughness": 0.25,
            "u_ambient": 0.05,
            "u_terrain_strength": 0.15,
            "u_use_texture": False,
        },
    ),
    "gas_giant": MaterialDefinition(
        material_id="gas_giant",
        program="gas_giant",
        uniforms={
            "u_banding": 0.75,
            "u_turbulence": 0.55,
            "u_limb_haze": 0.35,
            "u_day_night_contrast": 0.25,
            "u_ambient": 0.04,
            "u_time": 0.0,
        },
    ),
    "hot_giant": MaterialDefinition(
        material_id="hot_giant",
        program="gas_giant",
        uniforms={
            "u_banding": 0.45,
            "u_turbulence": 0.65,
            "u_limb_haze": 0.5,
            "u_day_night_contrast": 0.7,
            "u_ambient": 0.05,
            "u_time": 0.0,
        },
    ),
    "ultra_hot_giant": MaterialDefinition(
        material_id="ultra_hot_giant",
        program="gas_giant",
        uniforms={
            # Cloud condensation is suppressed above roughly 2000 K, so the
            # banding is deliberately weak here.
            "u_banding": 0.15,
            "u_turbulence": 0.35,
            "u_limb_haze": 0.65,
            "u_day_night_contrast": 0.95,
            "u_ambient": 0.06,
            "u_time": 0.0,
        },
    ),
    "atmosphere": MaterialDefinition(
        material_id="atmosphere",
        program="atmosphere",
        blend=True,
        cull_back_faces=False,
        uniforms={
            "u_enabled": False,
            "u_rayleigh": (0.19, 0.45, 1.0),
            "u_mie": 0.02,
            "u_mie_anisotropy": 0.76,
            "u_optical_depth": 0.6,
            "u_scale_height_fraction": 1.0,
            "u_shell_scale": 1.025,
        },
    ),
    "orbit": MaterialDefinition(
        material_id="orbit",
        program="orbit",
        blend=True,
        cull_back_faces=False,
        uniforms={
            "u_color": (1.0, 1.0, 1.0, 0.35),
            "u_dashed": False,
            "u_dash_period": 0.05,
        },
    ),
}

#: Maps :class:`~astro_explorer.assets.procedural.MaterialClass` values onto
#: material ids, so the science layer's classification chooses the shader.
_CLASS_TO_MATERIAL = {
    "ROCKY": "rocky",
    "ICY": "icy",
    "GAS_GIANT": "gas_giant",
    "HOT_GIANT": "hot_giant",
    "ULTRA_HOT_GIANT": "ultra_hot_giant",
    "UNKNOWN": "rocky",
}


def material_for(material_class) -> MaterialDefinition:
    """Material definition for a procedural material class."""
    key = getattr(material_class, "value", str(material_class))
    return MATERIALS[_CLASS_TO_MATERIAL.get(key, "rocky")]


def atmosphere_uniforms(
    *,
    has_evidence: bool,
    scale_height_km: float | None = None,
    planet_radius_km: float | None = None,
) -> dict:
    """Uniforms for the atmospheric shell.

    Roadmap section 15: the shell is only enabled when there is actual
    evidence of an atmosphere.  With no evidence the shader discards every
    fragment, so a bare rock is drawn as a bare rock.
    """
    definition = MATERIALS["atmosphere"]
    if not has_evidence:
        return definition.with_values(u_enabled=False)

    fraction = 1.0
    shell = 1.025
    if scale_height_km and planet_radius_km and planet_radius_km > 0:
        # Draw the shell a few scale heights thick, capped so a puffy
        # atmosphere does not swallow the planet.
        fraction = float(min(3.0, max(0.2, scale_height_km / planet_radius_km * 10.0)))
        shell = 1.0 + min(0.15, 5.0 * scale_height_km / planet_radius_km)

    return definition.with_values(
        u_enabled=True,
        u_scale_height_fraction=fraction,
        u_shell_scale=shell,
    )
