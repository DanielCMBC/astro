"""Text normalisation at the data/UI boundary (review section 11).

Pandas represents a missing *string* column as float ``NaN``, which is
truthy. So the idiomatic-looking

.. code-block:: python

    str(row.get("st_spectype") or "").strip()

leaks the literal word ``nan`` into the interface - which is how
"Spectral type: nan" reached the system panel.

Every catalogue string therefore crosses into the application through
:func:`clean_text`, which collapses ``None``, ``NaN``, empty and
whitespace-only values, plus the archive's own null spellings, to the empty
string. Presentation code then decides what to show instead.

This module lives at the package root deliberately: both :mod:`.data` and
:mod:`.ui` need it, and the layering forbids the UI importing from the data
package.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["clean_text", "display_text", "NULL_SPELLINGS"]

#: Spellings that mean "no value" in the archive's CSV output, in the
#: various cases they arrive in.
NULL_SPELLINGS = frozenset({"nan", "null", "none", "n/a", "na", "--", "-", "<na>", "nat"})


def clean_text(value: Any) -> str:
    """Catalogue text with every flavour of missing value collapsed to ``""``.

    Handles ``None``, float ``NaN``, pandas' ``NA``/``NaT`` sentinels, the
    empty string, whitespace-only strings and the archive's null spellings.
    Any other value is returned stripped.
    """
    if value is None:
        return ""

    # float("nan"), numpy.float64("nan") and pandas NA all fail an equality
    # test with themselves; that is the most portable way to catch them.
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if value != value:  # noqa: PLR0124 - NaN/NA sentinel check
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return "" if text.lower() in NULL_SPELLINGS else text


def display_text(value: Any, fallback: str = "unknown") -> str:
    """:func:`clean_text`, with a placeholder for the empty result.

    Use this at the point of display so the fallback wording is a
    presentation choice rather than something baked into the data layer.
    """
    return clean_text(value) or fallback
