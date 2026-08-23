"""Stable entity keys, separate from display names.

Review sections 7 and 13. A catalogue name is a much better selection key
than an array index or a render handle, because it survives a scene
rebuild. It is still not stable *enough*: display names and aliases change
between catalogue releases, and a selection that breaks when a planet is
renamed is a selection that breaks.

So identity and naming are separated:

``entity_id``
    ``planet:nasa:HD_80606_b`` - deterministic, derived from the catalogue's
    canonical key, and the thing selection, picking and panels all use.

``display_name``
    ``HD 80606 b`` - what a human reads, and the only thing a label shows.

The id is a pure function of the catalogue key, so it can be recomputed at
any time rather than stored and synchronised. When a Gaia ``source_id`` or
an internal database key becomes authoritative, ``Catalog.GAIA`` and a new
``key`` are all that changes; nothing downstream cares.

What "stable" actually means here
---------------------------------
Explorer B review section 7. The guarantee is precisely this:

    an entity id is stable while the authoritative catalogue key remains
    unchanged.

So a selection survives a scene rebuild, a level-of-detail change and a
change of *display* label, because none of those touch the key. It does
**not** survive a true canonical rename by the catalogue: ``planet:nasa:K2-18_b``
and ``planet:nasa:EPIC_201912552_b`` are different ids for the same planet,
and nothing in this module can know that.

Closing that gap needs a layer this module does not have and should not
grow: a persistent internal entity id, minted locally and never derived
from a name, with catalogue identifiers and aliases - NASA canonical name,
Gaia ``source_id``, SIMBAD identifiers - hanging off it as attributes. That
belongs with the offline synchronised catalogue in
:mod:`astro_explorer.data.synchronizer`, because it needs somewhere durable
to live. Until then the honest claim is the one above, not "rename-proof".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..text import clean_text

__all__ = [
    "Catalog",
    "EntityKind",
    "EntityId",
    "normalise_key",
    "planet_id",
    "star_id",
    "parse_entity_id",
]


class Catalog(str, Enum):
    """Which naming authority a key belongs to."""

    NASA = "nasa"
    GAIA = "gaia"
    SIMBAD = "simbad"
    LOCAL = "local"


class EntityKind(str, Enum):
    STAR = "star"
    PLANET = "planet"
    SYSTEM = "system"


#: Anything that is not alphanumeric, a dot or a plus becomes an underscore.
#: Dots and pluses survive because they carry meaning in designations such as
#: ``2MASS J0437+2331`` and ``Kepler-11 b``.
_UNSAFE = re.compile(r"[^A-Za-z0-9.+-]+")


def normalise_key(name: str) -> str:
    """A deterministic, filesystem- and URL-safe form of a catalogue name.

    Case is preserved: exoplanet designations are case-significant
    (``K2-18 b`` and ``K2-18 B`` are a planet and a stellar companion).
    """
    text = clean_text(name)
    if not text:
        return ""
    return _UNSAFE.sub("_", text).strip("_")


@dataclass(frozen=True, order=True)
class EntityId:
    """A stable identity: ``kind:catalog:key``."""

    kind: EntityKind
    catalog: Catalog
    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("an entity id needs a non-empty key")

    def __str__(self) -> str:
        return "{0}:{1}:{2}".format(self.kind.value, self.catalog.value, self.key)

    @property
    def is_planet(self) -> bool:
        return self.kind is EntityKind.PLANET

    @property
    def is_star(self) -> bool:
        return self.kind is EntityKind.STAR


def planet_id(name: str, catalog: Catalog = Catalog.NASA) -> EntityId | None:
    """``HD 80606 b`` -> ``planet:nasa:HD_80606_b``. None when unnamed."""
    key = normalise_key(name)
    return EntityId(EntityKind.PLANET, catalog, key) if key else None


def star_id(hostname: str, catalog: Catalog = Catalog.NASA) -> EntityId | None:
    """``HD 80606`` -> ``star:nasa:HD_80606``. None when unnamed."""
    key = normalise_key(hostname)
    return EntityId(EntityKind.STAR, catalog, key) if key else None


def parse_entity_id(text: str) -> EntityId:
    """Inverse of :meth:`EntityId.__str__`."""
    parts = str(text).split(":")
    if len(parts) != 3:
        raise ValueError("expected 'kind:catalog:key', got {0!r}".format(text))
    kind, catalog, key = parts
    return EntityId(EntityKind(kind), Catalog(catalog), key)
