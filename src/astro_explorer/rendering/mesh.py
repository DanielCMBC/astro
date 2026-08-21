"""Mesh generation for the modern pipeline (roadmap section 4.8).

The prototype drew spheres with ``gluSphere`` and immediate-mode
``glBegin``/``glEnd``.  Those are fixed-function constructs with no place in
a core-profile context.  This module builds plain vertex and index arrays
that a VAO/VBO/EBO can be filled from; it does not touch OpenGL itself, so
it is fully testable without a GL context.

Vertex layout is interleaved ``position(3) normal(3) uv(2)``, 8 floats per
vertex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh", "uv_sphere", "icosphere", "orbit_line", "lod_for_distance"]

VERTEX_STRIDE_FLOATS = 8


@dataclass(frozen=True)
class Mesh:
    """Interleaved vertex data plus a triangle index buffer."""

    vertices: np.ndarray  # (N, 8) float32
    indices: np.ndarray  # (M, 3) uint32
    name: str = ""

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def vertex_bytes(self) -> bytes:
        return np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes()

    @property
    def index_bytes(self) -> bytes:
        return np.ascontiguousarray(self.indices, dtype=np.uint32).tobytes()

    @property
    def vertex_format(self) -> str:
        """ModernGL buffer format string for the interleaved layout."""
        return "3f 3f 2f"

    @property
    def attributes(self) -> tuple[str, str, str]:
        return ("in_position", "in_normal", "in_uv")


def uv_sphere(rings: int = 32, sectors: int = 64, radius: float = 1.0) -> Mesh:
    """A latitude/longitude sphere with normals and equirectangular UVs.

    The seam is duplicated (``sectors + 1`` columns) so the texture wraps
    without a smeared column, which a shared-vertex sphere cannot do.
    """
    rings = max(2, int(rings))
    sectors = max(3, int(sectors))

    phi = np.linspace(0.0, np.pi, rings + 1)  # polar angle
    theta = np.linspace(0.0, 2.0 * np.pi, sectors + 1)  # azimuth
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.cos(phi_grid)
    z = np.sin(phi_grid) * np.sin(theta_grid)

    normals = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    positions = normals * radius
    u_coord = (theta_grid / (2.0 * np.pi)).reshape(-1, 1)
    v_coord = (1.0 - phi_grid / np.pi).reshape(-1, 1)

    vertices = np.hstack([positions, normals, u_coord, v_coord]).astype(np.float32)

    row_stride = sectors + 1
    ring_index, sector_index = np.meshgrid(
        np.arange(rings), np.arange(sectors), indexing="ij"
    )
    top_left = (ring_index * row_stride + sector_index).ravel()
    top_right = top_left + 1
    bottom_left = top_left + row_stride
    bottom_right = bottom_left + 1

    indices = np.concatenate(
        [
            np.stack([top_left, bottom_left, top_right], axis=-1),
            np.stack([top_right, bottom_left, bottom_right], axis=-1),
        ]
    ).astype(np.uint32)

    return Mesh(vertices=vertices, indices=indices, name="uv_sphere_{0}x{1}".format(rings, sectors))


_ICOSAHEDRON_VERTICES = None


def _base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    """A geodesic sphere with near-uniform triangles.

    Preferred over :func:`uv_sphere` for untextured bodies and for LOD, since
    it has no pole pinch and no wasted vertices near the poles.
    """
    vertices, faces = _base_icosahedron()

    for _ in range(max(0, int(subdivisions))):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        vertex_list = list(vertices)

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            point = (vertex_list[i] + vertex_list[j]) / 2.0
            point = point / np.linalg.norm(point)
            vertex_list.append(point)
            index = len(vertex_list) - 1
            midpoint_cache[key] = index
            return index

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

        vertices = np.array(vertex_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    normals = vertices
    positions = vertices * radius
    # Spherical UVs; acceptable here because icospheres are used untextured
    # or with procedural shaders that sample by direction.
    u_coord = (0.5 + np.arctan2(normals[:, 2], normals[:, 0]) / (2.0 * np.pi)).reshape(-1, 1)
    v_coord = (0.5 - np.arcsin(np.clip(normals[:, 1], -1.0, 1.0)) / np.pi).reshape(-1, 1)

    interleaved = np.hstack([positions, normals, u_coord, v_coord]).astype(np.float32)
    return Mesh(
        vertices=interleaved,
        indices=faces.astype(np.uint32),
        name="icosphere_{0}".format(subdivisions),
    )


def orbit_line(positions_au: np.ndarray) -> np.ndarray:
    """Pack orbit-path points into a float32 line-strip buffer.

    The renderer draws orbits as a single batched line strip per orbit
    rather than one draw call per segment (roadmap section 25: avoid
    per-object Python draw overhead).
    """
    points = np.asarray(positions_au, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("orbit_line expects an (N, 3) array of positions")
    return np.ascontiguousarray(points)


#: Screen-space size thresholds, in pixels, at which the sphere LOD drops.
LOD_THRESHOLDS = ((256.0, 4), (96.0, 3), (32.0, 2), (8.0, 1))


def lod_for_distance(radius_display: float, distance: float, viewport_height: int, fov_y_rad: float) -> int:
    """Pick an icosphere subdivision level from projected screen size."""
    if distance <= 0.0:
        return LOD_THRESHOLDS[0][1]
    projected = (radius_display / distance) / np.tan(0.5 * fov_y_rad) * viewport_height
    for threshold, level in LOD_THRESHOLDS:
        if projected >= threshold:
            return level
    return 0
