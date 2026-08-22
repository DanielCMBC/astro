"""Object picking by analytic ray-sphere intersection.

Roadmap Phase 3 requires object picking.  Doing it analytically rather than
with a colour-ID framebuffer keeps it testable without a GL context and
avoids a per-frame readback stall.

Bodies are spheres, so a closed-form intersection is exact.  A minimum
screen-space pick radius is applied so that a body which is only a couple of
pixels across is still clickable - without it, distant stars would be
impossible to select.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PickResult", "pick", "ray_sphere_intersection"]


def ray_sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float | None:
    """Distance along the ray to the nearest hit, or None.

    Uses the numerically stable form of the quadratic solution, which avoids
    the catastrophic cancellation the textbook formula suffers when the ray
    origin is far from a small sphere - exactly the regime of a camera
    parsecs away from a planet.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)

    offset = origin - center
    b = np.dot(offset, direction)
    c = np.dot(offset, offset) - radius * radius

    if c > 0.0 and b > 0.0:
        return None  # sphere is behind the ray

    discriminant = b * b - c
    if discriminant < 0.0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    # Stable root selection.
    q = -b + sqrt_disc if b < 0.0 else -b - sqrt_disc
    roots = [t for t in (q, c / q if q != 0.0 else np.inf) if np.isfinite(t) and t > 0.0]
    if not roots:
        return None
    return float(min(roots))


@dataclass(frozen=True)
class PickResult:
    """What the cursor is over."""

    identifier: str
    kind: str  # "star" or "planet"
    distance: float
    position_local: np.ndarray


def _effective_radius(
    radius: float,
    depth: float,
    camera,
    viewport_height: int,
    min_pixels: float,
) -> float:
    """Grow a tiny body to at least ``min_pixels`` on screen for picking.

    ``depth`` is the distance along the view direction, not the radial
    distance to the camera. Perspective scaling is governed by forward
    depth, and the two diverge towards the edge of a wide field of view -
    where a radial measure would inflate off-axis bodies more than
    on-axis ones and bias selection towards the edges of the screen.
    """
    if viewport_height <= 0 or depth <= 0.0:
        return radius
    world_per_pixel = 2.0 * depth * np.tan(0.5 * camera.fov_y_rad) / viewport_height
    return max(radius, min_pixels * world_per_pixel)


def pick(
    scene,
    camera,
    pixel_x: float,
    pixel_y: float,
    viewport_width: int,
    viewport_height: int,
    *,
    min_pick_pixels: float = 6.0,
) -> PickResult | None:
    """Nearest star or planet under a pixel, or None.

    Planets are tested before stars at equal distance, because a planet is
    the more specific selection when one is silhouetted against its host.
    """
    origin, direction = camera.ray_through_pixel(
        pixel_x, pixel_y, viewport_width, viewport_height
    )

    best: PickResult | None = None

    def consider(identifier: str, kind: str, position, radius: float) -> None:
        nonlocal best
        center = np.asarray(position, dtype=np.float64)
        offset = center - origin
        distance_to_center = float(np.linalg.norm(offset))

        # Reject anything behind the camera before any radius inflation.
        # The legacy prototype measured perpendicular distance to an
        # *infinite* line, so a star directly behind the viewer scored as
        # well as one in front of it. Projecting onto the view direction
        # first makes that impossible.
        along_view = float(np.dot(offset, direction))
        if along_view + float(radius) <= 0.0:
            return
        effective = _effective_radius(
            float(radius), along_view, camera, viewport_height, min_pick_pixels
        )
        hit = ray_sphere_intersection(origin, direction, center, effective)
        if hit is None:
            return
        if best is None or hit < best.distance:
            best = PickResult(identifier, kind, hit, center.astype(np.float32))

    for planet in scene.planets:
        consider(planet.identifier, "planet", planet.position_local, planet.radius_display)
    for star in scene.stars:
        consider(star.identifier, "star", star.position_local, star.radius_display)

    return best
