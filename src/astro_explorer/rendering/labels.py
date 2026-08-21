"""Body labels composited onto a rendered frame (review section 15).

Placement is computed by
:meth:`~astro_explorer.rendering.renderer.SceneDescription.project_labels`,
which needs only the camera; this module turns those screen positions into
pixels. Drawing text is deliberately kept out of the GL backend: a text
atlas would tie the renderer to a font, and the same placements have to
serve a Qt overlay later.

Collision handling is intentionally simple - nearest label wins, farther
ones that would overlap are dropped. A crowded system like TRAPPIST-1 is
better read with three legible labels than seven illegible ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LabelStyle", "draw_labels", "resolve_collisions"]


@dataclass(frozen=True)
class LabelStyle:
    """How labels are drawn onto a frame."""

    color: tuple[int, int, int] = (222, 232, 248)
    halo: tuple[int, int, int] = (4, 6, 12)
    offset: tuple[int, int] = (9, -6)
    font_size: int = 13
    #: Minimum gap between two label anchors, in pixels.
    min_separation: int = 26
    #: Draw a short leader from the body to its text.
    leader: bool = True


def resolve_collisions(placements, min_separation: int):
    """Keep the nearest label wherever two would overlap.

    ``placements`` must already be sorted nearest-first, which is what
    ``project_labels`` returns.
    """
    kept: list = []
    for placement in placements:
        x, y = placement[1], placement[2]
        if any(
            abs(x - other[1]) < min_separation and abs(y - other[2]) < min_separation
            for other in kept
        ):
            continue
        kept.append(placement)
    return kept


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_labels(
    image: np.ndarray,
    placements,
    style: LabelStyle | None = None,
    *,
    header: list[str] | None = None,
) -> np.ndarray:
    """Composite labels, and an optional header block, onto a frame.

    Returns a new ``(H, W, 3)`` uint8 array; the input is not modified.
    Requires Pillow; raises ImportError if it is absent, because a caller
    asking for labels should be told rather than silently given none.
    """
    from PIL import Image, ImageDraw

    style = style or LabelStyle()
    picture = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(picture)
    font = _font(style.font_size)

    for placement in resolve_collisions(placements, style.min_separation):
        label, x, y = placement[0], placement[1], placement[2]
        screen_radius = placement[4] if len(placement) > 4 else 0.0
        # Clear the body itself, so a large host star is not written over.
        text_x = x + style.offset[0] + max(0.0, screen_radius)
        text_y = y + style.offset[1]

        if style.leader:
            draw.line(
                [(x, y), (text_x - 2, text_y + style.font_size * 0.6)],
                fill=style.halo,
                width=1,
            )
        # A one-pixel halo keeps text readable over both the black sky and a
        # bright limb, without needing an outline font.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((text_x + dx, text_y + dy), label, font=font, fill=style.halo)
        draw.text((text_x, text_y), label, font=font, fill=style.color)

    if header:
        small = _font(max(style.font_size - 2, 8))
        for index, line in enumerate(header):
            y = 10 + index * (style.font_size + 3)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((12 + dx, y + dy), line, font=small, fill=style.halo)
            draw.text((12, y), line, font=small, fill=style.color)

    return np.asarray(picture, dtype=np.uint8)
