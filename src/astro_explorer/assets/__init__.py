"""Asset resolution, provenance manifest and procedural materials."""

from .manager import DECLARED_RESOURCES, Resource, ResourceManager, ResourceNotFound
from .manifest import AssetManifest, AssetRecord, AssetType, sha256_file
from .procedural import (
    MaterialClass,
    PlanetMaterial,
    StarColor,
    blackbody_rgb,
    planet_material,
    star_display_color,
)

__all__ = [
    "DECLARED_RESOURCES",
    "AssetManifest",
    "AssetRecord",
    "AssetType",
    "MaterialClass",
    "PlanetMaterial",
    "Resource",
    "ResourceManager",
    "ResourceNotFound",
    "StarColor",
    "blackbody_rgb",
    "planet_material",
    "sha256_file",
    "star_display_color",
]
